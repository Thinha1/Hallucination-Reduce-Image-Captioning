"""FastAPI app for the captioning pipeline demo.

Run with the Python 3.10 interpreter that already has torch/ultralytics/transformers/fastapi
installed (see webapp/backend/README.md), from the folder that contains webapp/:

    C:\\Users\\thinh\\AppData\\Local\\Programs\\Python\\Python310\\python.exe -m uvicorn webapp.backend.main:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import engines, routes
from .pipeline import vocab
from .pipeline.clipscore import ClipScorer
from .pipeline.config import PipelineConfig, load_caption_models
from .pipeline.formula_reranker import FormulaReranker
from .pipeline.retrieval import ObjectLexicon, RetrievalBank
from .pipeline.yolo_objects import YoloObjectExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PipelineConfig.defaults()
    engines.config = config

    engines.detector = YoloObjectExtractor(
        model_name=config.yolo.model_name,
        confidence_threshold=config.yolo.confidence_threshold,
        max_objects=config.yolo.max_objects,
        device=config.device,
    )
    # Real class names from the loaded checkpoint (not a hand-typed guess) drive the
    # object-noun vocabulary FormulaReranker uses for S_belief/hallucination flagging --
    # see vocab.set_object_noun_vocab. This also eagerly loads the YOLO model at startup instead
    # of on the first request.
    vocab.set_object_noun_vocab(engines.detector.class_names())
    engines.scorer = ClipScorer(model_name=config.clipscore.model_name, device=config.clipscore.device)
    engines.reranker = FormulaReranker(
        alpha=config.formula.alpha,
        beta=config.formula.beta,
        gamma=config.formula.gamma,
        clip_scorer=engines.scorer,
        belief_neg_confidence=0.0,
    )
    engines.lexicon = ObjectLexicon()
    engines.retrieval = RetrievalBank(
        model_name=config.retrieval.model_name,
        captions_path=config.retrieval.captions_path,
        embeddings_path=config.retrieval.embeddings_path,
        lexicon=engines.lexicon,
        k=config.retrieval.k,
        min_votes=config.retrieval.min_votes,
        device=config.device,
    )

    specs, default_id = load_caption_models()
    engines.caption_models = {spec.id: spec for spec in specs}
    engines.default_model_id = default_id
    engines.get_generator(default_id)

    yield

    engines.clear()


app = FastAPI(title="Branching Captioning Pipeline API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)
app.include_router(routes.resources_router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engines_ready": engines.is_ready()}
