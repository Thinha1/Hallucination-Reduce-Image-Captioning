"""Caption retrieval, object evidence, the fixed logit-blocking rule and the two
retrieval-based reranking signals of the proposed method.

Mirrors the evaluation code in demo/ (research/scripts/sretr_experiment.py,
sobj_threshold_eval.py and research/reports/scripts/thesis/time_inference.py), so the web app
picks the same caption the thesis numbers were computed with:

    E(I)    = YOLO objects  U  {COCO objects named by >= r of the K retrieved train captions}
    z'(w)   = z(w) - delta * 1[w is a COCO object word and obj(w) not in E(I)]
    S_retr  = mean CLIP ViT-B/32 cosine between a candidate and the K retrieved captions
    S_obj   = -(number of COCO object mentions in the candidate that are not in E(I))
    S_final = S_formula + w_retr * minmax(S_retr) + w_obj * minmax(S_obj)   (min-max per image)
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import chair_metric
from .schema import CaptionCandidate, DetectedObject, FormulaBreakdown, HallucinationFlag, RankedCaption

# Plural forms the WordNet lemmatizer leaves untouched (same table as time_inference.py).
_IRREGULAR = {"men": "man", "women": "woman", "kids": "kid", "guys": "guy", "mice": "mouse"}


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    return np.full_like(values, 0.5) if hi - lo < 1e-9 else (values - lo) / (hi - lo)


class ObjectLexicon:
    """COCO object words (CHAIR synonym list) -> canonical object, shared by the evidence set,
    the logit-blocking rule, S_obj and the hallucination flags."""

    def __init__(self) -> None:
        self.objects, self.inverse = chair_metric.load_synonyms()
        self._token_maps: dict[int, tuple[list[int], list[str]]] = {}

    def objects_in(self, text: str) -> list[str]:
        """Canonical COCO objects mentioned in `text`, one entry per mention (CHAIR tokenizer)."""
        return chair_metric.caption_to_words(text, self.objects, self.inverse)[1]

    def detected(self, objects: list[DetectedObject], min_confidence: float) -> set[str]:
        return {self.inverse[o.label] for o in objects if o.confidence >= min_confidence and o.label in self.inverse}

    def token_objects(self, vocab: dict[str, int]) -> tuple[list[int], list[str]]:
        """(token ids, their canonical object) for every captioner vocabulary word naming a COCO object."""
        key = id(vocab)
        if key not in self._token_maps:
            pairs = {}
            for word, idx in vocab.items():
                lemma = _IRREGULAR.get(word) or chair_metric._LEMMATIZER.lemmatize(word, pos="n")
                if lemma in self.objects:
                    pairs[idx] = self.inverse[lemma]
            ids = sorted(pairs)
            self._token_maps[key] = (ids, [pairs[i] for i in ids])
        return self._token_maps[key]

    def surface_word(self, text: str, obj: str) -> str:
        """The word as written in `text` for a canonical object, so the UI can highlight it."""
        for token in re.findall(r"[A-Za-z]+", text):
            lemma = chair_metric._LEMMATIZER.lemmatize(token.lower(), pos="n")
            if self.inverse.get(lemma) == obj:
                return token.lower()
        return obj


@dataclass
class RetrievedCaption:
    index: int
    caption: str
    similarity: float
    objects: list[str]


@dataclass
class RetrievalResult:
    neighbours: list[RetrievedCaption]
    votes: dict[str, int]  # canonical object -> number of retrieved captions naming it


@dataclass
class Evidence:
    detected: set[str]
    retrieved: set[str]  # objects named by >= min_votes retrieved captions

    @property
    def objects(self) -> set[str]:
        return self.detected | self.retrieved


class RetrievalBank:
    """Train/restval caption bank encoded with CLIP ViT-B/32. Loaded on demand (the UI has a node
    for it) because it holds about 1 GB in memory."""

    def __init__(
        self,
        model_name: str,
        captions_path: Path,
        embeddings_path: Path,
        lexicon: ObjectLexicon,
        k: int = 5,
        min_votes: int = 3,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.captions_path = Path(captions_path)
        self.embeddings_path = Path(embeddings_path)
        self.lexicon = lexicon
        self.k = k
        self.min_votes = min_votes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.captions: list[str] | None = None
        self.embeddings: np.ndarray | None = None  # [N, 512] float16, L2-normalized
        self._bank: torch.Tensor | None = None
        self._model = None
        self._processor = None
        self._caption_objects: dict[int, list[str]] = {}
        self._lock = threading.Lock()
        self.loading = False
        self.load_ms: float | None = None
        self.error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._bank is not None

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "num_captions": len(self.captions) if self.captions is not None else None,
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
            "size_mb": round(self.embeddings_path.stat().st_size / 2**20) if self.embeddings_path.is_file() else None,
            "model_name": Path(self.model_name).name,
            "k": self.k,
            "min_votes": self.min_votes,
            "device": self.device,
            "load_ms": self.load_ms,
            "error": self.error,
        }

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            self.loading = True
            self.error = None
            start = time.perf_counter()
            try:
                from transformers import CLIPModel, CLIPProcessor

                self.captions = json.loads(self.captions_path.read_text(encoding="utf-8"))
                self.embeddings = np.load(self.embeddings_path)
                self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self._bank = torch.from_numpy(self.embeddings).to(self.device, dtype)
                self.load_ms = round((time.perf_counter() - start) * 1000, 1)
            except Exception as exc:
                self.error = str(exc)
                raise
            finally:
                self.loading = False

    @torch.no_grad()
    def _encode_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=[image], return_tensors="pt").to(self.device)
        with torch.autocast(self.device, enabled=self.device == "cuda"):
            features = self._model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy().astype(np.float16)
        return torch.from_numpy(features).to(self._bank.device, self._bank.dtype)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.autocast(self.device, enabled=self.device == "cuda"):
            features = self._model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy().astype(np.float16)

    def _objects_of(self, index: int) -> list[str]:
        if index not in self._caption_objects:
            self._caption_objects[index] = sorted(set(self.lexicon.objects_in(self.captions[index])))
        return self._caption_objects[index]

    def retrieve(self, image_path: str | Path) -> RetrievalResult:
        if not self.loaded:
            raise RuntimeError("Kho truy hồi chưa được nạp.")
        image = self._encode_image(Path(image_path))
        sims, idx = (image @ self._bank.T).topk(self.k, dim=1)
        neighbours = [
            RetrievedCaption(index=int(j), caption=self.captions[int(j)], similarity=float(s), objects=self._objects_of(int(j)))
            for s, j in zip(sims[0].tolist(), idx[0].tolist())
        ]
        votes = Counter(o for n in neighbours for o in n.objects)
        return RetrievalResult(neighbours=neighbours, votes=dict(votes.most_common()))

    def evidence(self, objects: list[DetectedObject], retrieval: RetrievalResult, min_confidence: float) -> Evidence:
        return Evidence(
            detected=self.lexicon.detected(objects, min_confidence),
            retrieved={o for o, n in retrieval.votes.items() if n >= self.min_votes},
        )


def logit_bias(
    lexicon: ObjectLexicon, vocab: dict[str, int], evidence: Evidence, delta: float, device: str
) -> tuple[torch.Tensor | None, list[str]]:
    """Per-vocabulary bias for the fixed blocking rule, plus the blocked objects (for display).
    delta <= 0 disables blocking."""
    token_ids, token_objects = lexicon.token_objects(vocab)
    blocked = sorted({o for o in token_objects if o not in evidence.objects})
    if delta <= 0:
        return None, blocked
    mask = torch.tensor([o not in evidence.objects for o in token_objects], device=device)
    bias = torch.zeros(len(vocab), device=device)
    bias[torch.tensor(token_ids, device=device)[mask]] = -delta
    return bias, blocked


def apply_retrieval_signals(
    ranked: list[RankedCaption],
    candidates: list[CaptionCandidate],
    bank: RetrievalBank,
    retrieval: RetrievalResult,
    evidence: Evidence,
    w_obj: float,
    w_retr: float,
) -> tuple[list[RankedCaption], list[HallucinationFlag]]:
    """Adds S_obj and S_retr on top of the three-signal formula score and re-ranks. Ties keep the
    candidate order (the plain beam candidate first), like np.argmax in the evaluation scripts."""
    texts = [c.text for c in candidates]
    by_text = {r.caption.text: r for r in ranked}
    base = np.array([by_text[t].rerank_score for t in texts])

    neighbour_emb = bank.embeddings[[n.index for n in retrieval.neighbours]].astype(np.float32)
    s_retr = (bank.encode_texts(texts).astype(np.float32) @ neighbour_emb.T).mean(axis=1)
    mentions = [bank.lexicon.objects_in(t) for t in texts]
    s_obj = np.array([-sum(o not in evidence.objects for o in m) for m in mentions], dtype=np.float64)
    n_retr, n_obj = minmax(s_retr), minmax(s_obj)
    final = base + w_retr * n_retr + w_obj * n_obj

    order = sorted(range(len(texts)), key=lambda i: (-final[i], i))
    out = []
    for rank, i in enumerate(order, start=1):
        old = by_text[texts[i]]
        breakdown = old.formula_breakdown or FormulaBreakdown(s_base=0.5, s_align=0.5, s_belief=0.5)
        breakdown = replace(breakdown, s_obj=float(n_obj[i]), s_retr=float(n_retr[i]), formula_score=float(base[i]))
        out.append(replace(old, rank=rank, rerank_score=float(final[i]), formula_breakdown=breakdown))

    flags = []
    for i, objs in enumerate(mentions):
        for obj in dict.fromkeys(o for o in objs if o not in evidence.objects):
            flags.append(
                HallucinationFlag(
                    candidate_index=i,
                    word=bank.lexicon.surface_word(texts[i], obj),
                    note=f'"{obj}" không có trong bằng chứng E(I) (YOLO và câu truy hồi).',
                )
            )
    return out, flags
