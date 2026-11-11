from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Every model file the backend loads lives in webapp/backend/weights/ (caption checkpoints +
# vocab, YOLO-World, the two CLIP models, the retrieval bank), so the webapp folder runs on
# its own wherever it is copied. weights/models.json lists the caption checkpoints the UI can
# choose from.
_PIPELINE_DIR = Path(__file__).resolve().parent  # webapp/backend/pipeline
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", str(_PIPELINE_DIR.parent / "weights")))
MODELS_MANIFEST = WEIGHTS_DIR / "models.json"


def _weight(name: str, hf_fallback: str | None = None) -> str:
    """Local copy under weights/ if present, else the HuggingFace id (downloaded on first use)."""
    path = WEIGHTS_DIR / name
    return str(path) if path.exists() or hf_fallback is None else hf_fallback


@dataclass
class CaptionModelSpec:
    id: str
    label: str
    description: str
    checkpoint: Path
    vocab: Path
    default_delta: float


def load_caption_models() -> tuple[list[CaptionModelSpec], str]:
    data = json.loads(MODELS_MANIFEST.read_text(encoding="utf-8"))
    specs = [
        CaptionModelSpec(
            id=m["id"],
            label=m["label"],
            description=m.get("description", ""),
            checkpoint=WEIGHTS_DIR / m["checkpoint"],
            vocab=WEIGHTS_DIR / m["vocab"],
            default_delta=float(m.get("default_delta", 0.0)),
        )
        for m in data["models"]
    ]
    default = data.get("default") or specs[0].id
    return specs, default


@dataclass
class YoloConfig:
    model_name: str = _weight("yolov8l-worldv2.pt")
    confidence_threshold: float = 0.25
    max_objects: int = 20


@dataclass
class ClipScoreConfig:
    model_name: str = _weight("clip-vit-base-patch16", "openai/clip-vit-base-patch16")
    device: str | None = None


@dataclass
class GeneratorConfig:
    max_candidates: int = 5
    num_beams: int = 10
    num_beam_groups: int = 5
    diversity_penalty: float = 0.8
    max_new_tokens: int = 40
    safe_beam_size: int = 3
    # Only supplies the ViT-B/16 architecture; the caption checkpoint overwrites its weights.
    clip_source: str = _weight("clip-vit-base-patch16", "openai/clip-vit-base-patch16")


@dataclass
class RetrievalConfig:
    model_name: str = _weight("clip-vit-base-patch32", "openai/clip-vit-base-patch32")
    captions_path: Path = WEIGHTS_DIR / "retrieval" / "train_captions.json"
    embeddings_path: Path = WEIGHTS_DIR / "retrieval" / "train_caption_emb_b32.npy"
    k: int = 5
    min_votes: int = 3  # r: an object enters E(I) when >= r of the K retrieved captions name it
    w_obj: float = 1.0
    w_retr: float = 0.1
    sobj_min_confidence: float = 0.55  # YOLO-World threshold for S_obj's evidence only (chosen on val), blocking stays at 0.25


@dataclass
class FormulaConfig:
    # Weights chosen on the validation split (S_base / S_align / S_belief).
    alpha: float = 0.2
    beta: float = 0.4
    gamma: float = 0.4


@dataclass
class PipelineConfig:
    device: str
    yolo: YoloConfig = field(default_factory=YoloConfig)
    clipscore: ClipScoreConfig = field(default_factory=ClipScoreConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    formula: FormulaConfig = field(default_factory=FormulaConfig)

    @classmethod
    def defaults(cls) -> "PipelineConfig":
        device = os.getenv("PIPELINE_DEVICE") or ("cuda" if _cuda_available() else "cpu")
        return cls(
            device=device,
            yolo=YoloConfig(model_name=os.getenv("YOLO_MODEL_PATH", YoloConfig.model_name)),
            clipscore=ClipScoreConfig(device=device),
        )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
