from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .schema import CaptionCandidate


class ClipScorer:
    """Compute image-text similarity for caption candidates."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        # Cache of text -> normalized text embedding, reused by text_similarity() below. Labels
        # (e.g. "person", "cell phone") repeat across the whole run, and each candidate's own
        # text is looked up once per detected object in the same image -- caching avoids
        # re-encoding the same string through the CLIP text tower every time.
        self._text_embed_cache: dict[str, "torch.Tensor"] = {}

    def _load(self):
        if self._model is None or self._processor is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency 'transformers'. Install it with: pip install transformers"
                ) from exc

            self._model = CLIPModel.from_pretrained(
                self.model_name,
                use_safetensors=True,
            ).to(self.device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._model, self._processor

    def score(self, image_path: str | Path, captions: list[CaptionCandidate]) -> list[CaptionCandidate]:
        if not captions:
            return []

        model, processor = self._load()
        image = Image.open(image_path).convert("RGB")
        texts = [candidate.text for candidate in captions]

        inputs = processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (text_features @ image_features.T).squeeze(1)

        scored: list[CaptionCandidate] = []
        for candidate, score in zip(captions, similarities.tolist()):
            scored.append(
                CaptionCandidate(
                    text=candidate.text,
                    source=candidate.source,
                    generation_score=candidate.generation_score,
                    clip_score=float(score),
                    metadata=candidate.metadata,
                )
            )
        return scored

    def _text_embedding(self, text: str):
        cached = self._text_embed_cache.get(text)
        if cached is not None:
            return cached
        model, processor = self._load()
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            # transformers < 5: CLIPModel.get_text_features() returns the projected text
            # tensor directly. transformers >= 5 (@can_return_tuple refactor) returns the
            # BaseModelOutputWithPooling from the text tower instead, with .pooler_output
            # overwritten in place to hold the *projected* features (see
            # CLIPModel.get_text_features in transformers/models/clip/modeling_clip.py) --
            # return_dict=True pins that shape on both versions instead of the version-only
            # default, so this doesn't silently flip again on some future 5.x point release.
            raw = model.get_text_features(**inputs, return_dict=True)
            features = raw if isinstance(raw, torch.Tensor) else getattr(raw, "pooler_output", None)
            if features is None:
                raise RuntimeError(
                    f"CLIPModel.get_text_features() returned {type(raw).__name__} with no "
                    "usable tensor (no .pooler_output). The installed 'transformers' version "
                    "likely changed this API again -- check `pip show transformers`."
                )
            features = features / features.norm(dim=-1, keepdim=True)
        embedding = features.squeeze(0)
        self._text_embed_cache[text] = embedding
        return embedding

    def text_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts' CLIP text-encoder embeddings -- used as the
        sim(w, c) term of FormulaReranker's belief-revision S_belief (see formula_reranker.py).
        Reuses the same CLIP model already loaded for image-text scoring above instead of
        loading a second embedding model (the original paper uses SBERT-STS for this)."""
        a = self._text_embedding(text_a)
        b = self._text_embedding(text_b)
        return float((a @ b).item())
