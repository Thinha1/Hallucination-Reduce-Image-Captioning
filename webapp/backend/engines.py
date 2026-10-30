"""Holds the singleton, heavy pipeline components. Populated once in main.py's lifespan
startup and read by routes.py -- avoids reloading YOLO/CLIP/the checkpoint decoder on every
request, and avoids a circular import between main.py and routes.py.

Caption checkpoints (weights/models.json) are loaded on first use and then kept, so switching
models in the UI only pays the load once. The retrieval bank is loaded by its own UI node.
"""

from __future__ import annotations

import threading

from .pipeline.clipscore import ClipScorer
from .pipeline.config import CaptionModelSpec, PipelineConfig
from .pipeline.formula_reranker import FormulaReranker
from .pipeline.generate import CandidateCaptionGenerator
from .pipeline.retrieval import ObjectLexicon, RetrievalBank
from .pipeline.yolo_objects import YoloObjectExtractor

config: PipelineConfig | None = None
detector: YoloObjectExtractor | None = None
scorer: ClipScorer | None = None
reranker: FormulaReranker | None = None
lexicon: ObjectLexicon | None = None
retrieval: RetrievalBank | None = None

caption_models: dict[str, CaptionModelSpec] = {}
default_model_id: str | None = None
_generators: dict[str, CandidateCaptionGenerator] = {}
_generator_lock = threading.Lock()


def is_ready() -> bool:
    return None not in (config, detector, scorer, reranker, lexicon, retrieval) and bool(_generators)


def is_model_loaded(model_id: str) -> bool:
    return model_id in _generators


def get_generator(model_id: str) -> CandidateCaptionGenerator:
    if model_id not in caption_models:
        raise KeyError(model_id)
    with _generator_lock:
        if model_id not in _generators:
            spec = caption_models[model_id]
            gen_cfg = config.generator
            generator = CandidateCaptionGenerator(
                checkpoint_path=spec.checkpoint,
                vocab_path=spec.vocab,
                max_candidates=gen_cfg.max_candidates,
                num_beams=gen_cfg.num_beams,
                num_beam_groups=gen_cfg.num_beam_groups,
                diversity_penalty=gen_cfg.diversity_penalty,
                max_new_tokens=gen_cfg.max_new_tokens,
                safe_beam_size=gen_cfg.safe_beam_size,
                clip_source=gen_cfg.clip_source,
                device=config.device,
            )
            generator.local_captioner()  # load now, not in the middle of the first request
            _generators[model_id] = generator
        return _generators[model_id]


def clear() -> None:
    global config, detector, scorer, reranker, lexicon, retrieval
    config = detector = scorer = reranker = lexicon = retrieval = None
    _generators.clear()
