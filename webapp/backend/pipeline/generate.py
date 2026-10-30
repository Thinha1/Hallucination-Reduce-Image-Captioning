from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

from .local_captioner import LocalCheckpointCaptioner
from .schema import CaptionCandidate, DetectedObject
from .yolo_objects import summarize_objects


class CandidateCaptionGenerator:
    """Generate candidate captions from an image and detected objects."""

    def __init__(
        self,
        hf_model_name: str | None = None,
        checkpoint_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        custom_candidates_file: str | Path | None = None,
        max_candidates: int = 5,
        num_beams: int = 10,
        num_beam_groups: int = 5,
        diversity_penalty: float = 0.8,
        max_new_tokens: int = 40,
        safe_beam_size: int = 3,
        clip_source: str = "openai/clip-vit-base-patch16",
        device: str | None = None,
    ) -> None:
        self.hf_model_name = hf_model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.custom_candidates_file = Path(custom_candidates_file) if custom_candidates_file else None
        self.max_candidates = max_candidates
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.max_new_tokens = max_new_tokens
        self.safe_beam_size = safe_beam_size
        self.clip_source = clip_source
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._local_captioner = None

    def generate(
        self,
        image_path: str | Path,
        objects: list[DetectedObject],
        logit_bias: torch.Tensor | None = None,
    ) -> list[CaptionCandidate]:
        """`logit_bias` ([vocab]) is added to the checkpoint decoder's logits at every decoding
        step -- the fixed logit-blocking rule (see pipeline/retrieval.py). None = no blocking."""
        candidates: list[CaptionCandidate] = []
        candidates.extend(self._load_custom_candidates(image_path))
        candidates.extend(self._generate_with_checkpoint(image_path, logit_bias))
        candidates.extend(self._generate_with_hf(image_path))
        candidates.extend(self._generate_object_templates(objects))
        return self._dedupe(candidates)[: self.max_candidates]

    def _load_custom_candidates(self, image_path: str | Path) -> list[CaptionCandidate]:
        if not self.custom_candidates_file:
            return []
        if not self.custom_candidates_file.is_file():
            raise FileNotFoundError(f"Candidate file not found: {self.custom_candidates_file}")

        key = Path(image_path).name
        data = json.loads(self.custom_candidates_file.read_text(encoding="utf-8"))
        raw_items = data.get(key, data if isinstance(data, list) else [])

        candidates: list[CaptionCandidate] = []
        for item in raw_items:
            if isinstance(item, str):
                candidates.append(CaptionCandidate(text=item, source="custom_file"))
            elif isinstance(item, dict) and item.get("caption"):
                candidates.append(
                    CaptionCandidate(
                        text=str(item["caption"]),
                        source=str(item.get("source", "custom_file")),
                        generation_score=item.get("score"),
                        metadata={k: v for k, v in item.items() if k not in {"caption", "score", "source"}},
                    )
                )
        return candidates

    def _generate_with_hf(self, image_path: str | Path) -> list[CaptionCandidate]:
        if not self.hf_model_name:
            return []
        if Path(self.hf_model_name).suffix.lower() in {".pth", ".pt"}:
            raise ValueError(
                "CAPTION_MODEL is for HuggingFace model ids/folders, not .pth files. "
                "Use CAPTION_CHECKPOINT for demo/model.pth and set VOCAB_FILE."
            )

        if self._model is None or self._processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency 'transformers'. "
                    f"Current Python: {sys.executable}. "
                    "Install it into this interpreter with: python -m pip install transformers"
                ) from exc

            try:
                from transformers import AutoModelForVision2Seq as CaptionModel
            except ImportError:
                try:
                    from transformers import BlipForConditionalGeneration as CaptionModel
                except ImportError as exc:
                    raise RuntimeError(
                        "Installed 'transformers' is too old for this caption loader. "
                        f"Current Python: {sys.executable}. "
                        "Upgrade it with: python -m pip install -U transformers"
                    ) from exc

            self._processor = AutoProcessor.from_pretrained(self.hf_model_name)
            self._model = CaptionModel.from_pretrained(
                self.hf_model_name,
                use_safetensors=True,
            ).to(self.device)
            self._model.eval()

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        num_beams = self._valid_num_beams()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=num_beams,
                num_beam_groups=self.num_beam_groups,
                diversity_penalty=self.diversity_penalty,
                num_return_sequences=self.max_candidates,
                early_stopping=True,
            )
        decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)

        candidates: list[CaptionCandidate] = []
        for text in decoded:
            text = text.strip()
            if text:
                candidates.append(
                    CaptionCandidate(
                        text=text,
                        source=f"hf:{self.hf_model_name}",
                        metadata={
                            "decoding": "diverse_beam_search",
                            "num_beams": num_beams,
                            "num_beam_groups": self.num_beam_groups,
                            "diversity_penalty": self.diversity_penalty,
                        },
                    )
                )
        return candidates

    def local_captioner(self) -> LocalCheckpointCaptioner | None:
        """The checkpoint captioner, loaded on first use. Exposed so the logit-blocking rule can
        read the checkpoint's vocabulary before any caption is generated."""
        if not self.checkpoint_path:
            return None
        if not self.vocab_path:
            raise ValueError("CAPTION_CHECKPOINT requires VOCAB_FILE.")
        if self._local_captioner is None:
            self._local_captioner = LocalCheckpointCaptioner(
                checkpoint_path=self.checkpoint_path,
                vocab_path=self.vocab_path,
                device=self.device,
                max_length=self.max_new_tokens,
                clip_source=self.clip_source,
            )
        return self._local_captioner

    def _generate_with_checkpoint(
        self, image_path: str | Path, logit_bias: torch.Tensor | None = None
    ) -> list[CaptionCandidate]:
        if self.local_captioner() is None:
            return []

        candidates: list[CaptionCandidate] = []
        seen: set[str] = set()

        def add(text: str, score: float, token_log_probs: list[float], suffix: str, decoding: str, extra: dict) -> None:
            key = " ".join(text.lower().split())
            if not text or key in seen:
                return
            seen.add(key)
            candidates.append(
                CaptionCandidate(
                    text=text,
                    source=f"checkpoint:{self.checkpoint_path}{suffix}",
                    generation_score=score,
                    metadata={
                        "decoding": decoding,
                        # Per-word log-probs from a clean teacher-forcing pass (see
                        # LocalCheckpointCaptioner._teacher_force_log_probs) -- used by
                        # FormulaReranker for S_base. JSON-serializable, so it round-trips
                        # through --run-pipeline report files for later --rerank-only reuse.
                        "token_log_probs": token_log_probs,
                        **extra,
                    },
                )
            )

        # "Safe" candidate: plain beam search (num_beam_groups=1 disables the diversity penalty
        # entirely -- see LocalCheckpointCaptioner.generate_diverse), matching a standard
        # single-best beam search. Always generated first and included regardless of what the
        # diverse pool below produces, so a poorly-tuned reranker can never end up choosing worse
        # than plain beam search would have on its own.
        if self.safe_beam_size > 0:
            safe_outputs = self._local_captioner.generate_diverse(
                image_path=image_path,
                max_candidates=1,
                num_beams=self.safe_beam_size,
                num_beam_groups=1,
                diversity_penalty=0.0,
                logit_bias=logit_bias,
            )
            for text, score, token_log_probs in safe_outputs:
                add(text, score, token_log_probs, ":safe", "beam_search_plain", {"beam_size": self.safe_beam_size})

        # Diverse pool: alternative phrasings/content for the reranker to weigh against the safe
        # candidate above -- e.g. to catch/fix a hallucination the plain top beam made. Ask for a
        # couple extra: the plain and diverse searches can land on the same text, and deduping
        # that collision (see `add`) shouldn't silently shrink the pool below max_candidates --
        # that would let low-information object-template filler leak in as a substitute below.
        remaining = max(self.max_candidates - len(candidates), 0)
        if remaining:
            outputs = self._local_captioner.generate_diverse(
                image_path=image_path,
                max_candidates=remaining + 2,
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups,
                diversity_penalty=self.diversity_penalty,
                logit_bias=logit_bias,
            )
            extra = {
                "num_beams": self.num_beams,
                "num_beam_groups": self.num_beam_groups,
                "diversity_penalty": self.diversity_penalty,
            }
            for text, score, token_log_probs in outputs:
                if len(candidates) >= self.max_candidates:
                    break
                add(text, score, token_log_probs, "", "diverse_beam_search", extra)

        return candidates

    def _valid_num_beams(self) -> int:
        if self.num_beam_groups < 1:
            raise ValueError("num_beam_groups must be >= 1")
        if self.max_candidates > self.num_beams:
            self.num_beams = self.max_candidates
        remainder = self.num_beams % self.num_beam_groups
        if remainder:
            self.num_beams += self.num_beam_groups - remainder
        return self.num_beams

    def _generate_object_templates(self, objects: list[DetectedObject]) -> list[CaptionCandidate]:
        object_summary = summarize_objects(objects)
        if object_summary == "no detected objects":
            return [CaptionCandidate(text="An image with no clear objects detected.", source="object_template")]

        labels = list(dict.fromkeys(obj.label for obj in objects))
        candidates = [
            f"An image containing {object_summary}.",
            f"A scene with {object_summary}.",
        ]
        if len(labels) >= 2:
            candidates.append(f"A photo showing {labels[0]} and {labels[1]} in the scene.")
        if labels:
            candidates.append(f"A photo focused on {labels[0]}.")

        return [CaptionCandidate(text=text, source="object_template") for text in candidates]

    @staticmethod
    def _dedupe(candidates: list[CaptionCandidate]) -> list[CaptionCandidate]:
        seen: set[str] = set()
        unique: list[CaptionCandidate] = []
        for candidate in candidates:
            normalized = " ".join(candidate.text.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(candidate)
        return unique
