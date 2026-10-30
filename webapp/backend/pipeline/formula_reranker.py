from __future__ import annotations

import math
import os
import re

from .clipscore import ClipScorer
from .schema import CaptionCandidate, DetectedObject, FormulaBreakdown, HallucinationFlag, RankedCaption, RerankResult
from .vocab import clean_word, is_object_noun, singularize


def _float_env(*names: str, default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            try:
                return float(value)
            except ValueError:
                pass
    return default


def _normalize(values: list[float], valid: list[bool] | None = None) -> list[float]:
    """Min-max normalize `values` to [0, 1], using only entries where `valid[i]` is True to
    compute the min/max (entries missing their raw score don't skew the range for everyone
    else). Entries that are invalid, or where all valid values are equal (or there's < 2 of
    them, e.g. a single candidate), get the neutral value 0.5 -- this must not change the
    relative ranking, since it's the same additive contribution for every candidate."""
    if valid is None:
        valid = [True] * len(values)
    usable = [v for v, ok in zip(values, valid) if ok]
    if len(usable) < 2 or max(usable) == min(usable):
        return [0.5] * len(values)
    lo, hi = min(usable), max(usable)
    return [(v - lo) / (hi - lo) if ok else 0.5 for v, ok in zip(values, valid)]


def _belief_alpha(sim: float, confidence: float) -> float:
    """The Jeffrey's-rule revision exponent from Sabir et al., "Belief Revision based Caption
    Re-ranker with Visual Semantic Information" (COLING 2022):

        alpha = [(1 - sim(w, c)) / (1 + sim(w, c))] ** (1 - P(c))

    sim is clamped away from -1 (the ratio's pole) and confidence into [0, 1] defensively --
    real inputs (CLIP cosine similarity, YOLO box confidence) already live in range, this only
    guards against floating-point edge cases."""
    sim = min(max(sim, -0.999999), 0.999999)
    confidence = min(max(confidence, 0.0), 1.0)
    ratio = (1.0 - sim) / (1.0 + sim)
    return ratio ** (1.0 - confidence)


class FormulaReranker:
    """Deterministic composite reranker -- no network call, no LLM. Implements the S_total
    formula from "De xuat Co che Re-ranking Tong hop cho Mo hinh Sinh chu thich anh":

        S_total(c) = alpha*S_base(c) + beta*S_align(c) + gamma*S_belief(c)

    Each term is min-max normalized across the K candidates of the SAME image before combining
    (see _normalize) -- the three raw scores live on very different scales (log-prob, cosine
    similarity, a probability), so combining them raw would make the weights meaningless.
    alpha/beta/gamma are read from FORMULA_ALPHA/BETA/GAMMA in .env (default 1/3 each) and
    renormalized to sum to 1.

    - S_base: mean per-word log-prob from the checkpoint decoder's own teacher-forcing pass
      (candidate.metadata["token_log_probs"], populated in generate.py). Candidates from a
      different source (no such metadata) get the neutral 0.5 for this term.
    - S_align: candidate.clip_score (raw cosine similarity, already computed).
    - S_belief: Belief Revision score, following Sabir et al. (COLING 2022) -- see _s_belief.
    """

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        clip_scorer: ClipScorer | None = None,
        belief_neg_confidence: float | None = None,
    ) -> None:
        self.alpha = alpha if alpha is not None else _float_env("FORMULA_ALPHA", default=1.0 / 3.0)
        self.beta = beta if beta is not None else _float_env("FORMULA_BETA", default=1.0 / 3.0)
        self.gamma = gamma if gamma is not None else _float_env("FORMULA_GAMMA", default=1.0 / 3.0)
        total = self.alpha + self.beta + self.gamma
        if total > 0:
            self.alpha, self.beta, self.gamma = (
                self.alpha / total,
                self.beta / total,
                self.gamma / total,
            )
        # Used by _s_belief to compute sim(w, c) between a caption and a detected object's
        # label (see ClipScorer.text_similarity). None means "no live CLIP model available" --
        # _s_belief then relies entirely on a cached candidate.metadata["vc_sim"] and degrades
        # to the neutral 0.5 for anything not already cached there.
        self.clip_scorer = clip_scorer
        # Informativeness assigned to the "absence" of an object the caption names but the
        # detector never found at all (as opposed to the paper's negative-evidence method, which
        # uses a *low-confidence* detection as evidence of absence). We have no such confidence
        # value for something never detected, so default to 1.0 confidence-of-absence (i.e.
        # P(c)=0 in the formula below) -- treat "never detected" as maximally informative.
        # Tunable via FORMULA_BELIEF_NEG_CONFIDENCE if this proves too aggressive.
        self.belief_neg_confidence = (
            belief_neg_confidence
            if belief_neg_confidence is not None
            else _float_env("FORMULA_BELIEF_NEG_CONFIDENCE", default=0.0)
        )

    def rank(
        self,
        candidates: list[CaptionCandidate],
        objects: list[DetectedObject],
        start_rank: int = 1,
    ) -> list[RankedCaption]:
        if not candidates:
            return []

        base_raw, base_ok = self._s_base(candidates)
        align_raw = [c.clip_score if c.clip_score is not None else 0.0 for c in candidates]
        belief_raw, belief_ok = zip(*(self._s_belief(c, objects) for c in candidates))

        s_base = _normalize(base_raw, base_ok)
        s_align = _normalize(align_raw)
        s_belief = _normalize(list(belief_raw), list(belief_ok))

        scored: list[tuple[CaptionCandidate, float, FormulaBreakdown]] = []
        for i, candidate in enumerate(candidates):
            total = (
                self.alpha * s_base[i]
                + self.beta * s_align[i]
                + self.gamma * s_belief[i]
            )
            breakdown = FormulaBreakdown(s_base=s_base[i], s_align=s_align[i], s_belief=s_belief[i])
            scored.append((candidate, total, breakdown))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RankedCaption(caption=candidate, rank=rank, rerank_score=score, formula_breakdown=breakdown)
            for rank, (candidate, score, breakdown) in enumerate(scored, start=start_rank)
        ]

    def rerank(self, candidates: list[CaptionCandidate], objects: list[DetectedObject]) -> RerankResult:
        """Three-signal ranking plus a YOLO-only hallucination guess. The rerank step then adds
        S_obj/S_retr and replaces the flags with evidence-based ones (pipeline/retrieval.py)."""
        ranked = self.rank(candidates, objects)
        best = ranked[0] if ranked else None
        return RerankResult(
            ranked=ranked,
            hallucinations=heuristic_hallucinations(candidates, objects),
            model_name="công thức ba tín hiệu (S_base, S_align, S_belief)",
            selected_caption=best.caption.text if best else None,
        )

    @staticmethod
    def _s_base(candidates: list[CaptionCandidate]) -> tuple[list[float], list[bool]]:
        raw: list[float] = []
        ok: list[bool] = []
        for candidate in candidates:
            log_probs = (candidate.metadata or {}).get("token_log_probs")
            if log_probs:
                raw.append(sum(log_probs) / len(log_probs))
                ok.append(True)
            else:
                raw.append(0.0)
                ok.append(False)
        return raw, ok

    def _s_belief(self, candidate: CaptionCandidate, objects: list[DetectedObject]) -> tuple[float, bool]:
        """Belief Revision score, following Sabir et al., "Belief Revision based Caption
        Re-ranker with Visual Semantic Information" (COLING 2022, arXiv:2209.08163). The
        detected objects are treated as visual evidence that revises the caption's own
        probability via Jeffrey's rule of conditioning:

            P(w|c)  = P(w) ** alpha                      (positive evidence: object c is present)
            P(w|!c) = 1 - (1 - P(w)) ** alpha             (negative evidence: c is named but absent)
            alpha   = [(1 - sim(w,c)) / (1 + sim(w,c))] ** (1 - P(c))     (see _belief_alpha)

        - P(w): the caption's own prior probability. The paper initializes this from an
          external LM (GPT-2); we reuse the geometric mean of the checkpoint decoder's own
          per-word probabilities (candidate.metadata["token_log_probs"], the same source
          _s_base reads) since that IS this project's "common observation" prior.
        - sim(w,c): semantic relatedness between the caption and a detected object's label.
          The paper uses SBERT-STS; we reuse the CLIP text encoder already loaded for S_align
          (see ClipScorer.text_similarity) instead of loading a second embedding model. Cached
          per (candidate, label) into candidate.metadata["vc_sim"] so it round-trips through
          report JSON and needs no live CLIP model to replay.
        - P(c): the YOLO detector's confidence for that object -- the "informativeness" term.
          A confident detection lets the similarity term dominate the revision; a barely-
          confident one leaves P(w) close to unrevised (alpha -> 1).

        Every detected object is applied in turn as positive evidence, each revision's output
        feeding in as the next one's prior -- Jeffrey's rule chained over multiple pieces of
        evidence, matching how the paper revises against several visual-context labels (VC1,
        VC2, VC3, ...) for the same image.

        Object-noun words the caption names but no detected box matches (the project's
        hallucination signal) are then applied as *negative* evidence via P(w|!c), using the
        word itself as its own visual-context label (sim=1, an exact textual match) and
        self.belief_neg_confidence as P(c) -- see its docstring in __init__ for why "never
        detected at all" isn't quite the paper's low-confidence-detection negative evidence and
        how that's approximated here.

        Falls back to (0.0, False) -- caller applies the neutral 0.5 via _normalize -- when the
        candidate has no token_log_probs, or a similarity is neither cached nor computable
        (no live clip_scorer), so it never gets penalized against candidates that could compute
        every term.
        """
        log_probs = (candidate.metadata or {}).get("token_log_probs")
        if not log_probs:
            return 0.0, False

        # exp(mean log-prob): geometric mean per-word probability, kept strictly inside (0, 1)
        # since both belief-revision formulas divide/exponentiate around 1 - P(w).
        p_w = math.exp(sum(log_probs) / len(log_probs))
        p_w = min(max(p_w, 1e-6), 1.0 - 1e-6)

        detected_heads: set[str] = set()
        for obj in objects:
            sim = self._label_similarity(candidate, obj.label)
            if sim is None:
                return 0.0, False
            p_w = p_w ** _belief_alpha(sim, obj.confidence)
            detected_heads.add(singularize(clean_word(obj.label.split()[-1])))

        for word in candidate.text.split():
            cleaned = clean_word(word)
            if not cleaned or not is_object_noun(word) or singularize(cleaned) in detected_heads:
                continue
            alpha = _belief_alpha(sim=1.0, confidence=self.belief_neg_confidence)
            p_w = 1.0 - (1.0 - p_w) ** alpha

        return p_w, True

    def _label_similarity(self, candidate: CaptionCandidate, label: str) -> float | None:
        cache = candidate.metadata.setdefault("vc_sim", {})
        key = label.lower()
        if key in cache:
            return cache[key]
        if self.clip_scorer is None:
            return None
        sim = self.clip_scorer.text_similarity(candidate.text, label)
        cache[key] = sim
        return sim

def heuristic_hallucinations(
    candidates: list[CaptionCandidate],
    objects: list[DetectedObject],
) -> list[HallucinationFlag]:
    """Best-effort hallucination guess: flag common object nouns named in a caption that are
    absent from the YOLO-detected label set for the same image."""
    detected = {obj.label.lower().rstrip("s") for obj in objects}
    flags: list[HallucinationFlag] = []
    for idx, candidate in enumerate(candidates):
        words = re.findall(r"[a-zA-Z]+", candidate.text.lower())
        flagged_already: set[str] = set()
        for word in words:
            singular = singularize(word)
            if singular not in flagged_already and is_object_noun(word) and singular not in detected:
                flagged_already.add(singular)
                flags.append(
                    HallucinationFlag(
                        candidate_index=idx,
                        word=word,
                        note=f"\"{word}\" khong co trong danh sach object YOLO da detect duoc.",
                    )
                )
    return flags
