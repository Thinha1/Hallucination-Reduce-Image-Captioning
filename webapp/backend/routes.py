from __future__ import annotations

import mimetypes
import time
import base64
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from . import engines
from .jobs import JobNotFoundError, JobState, StepNotReadyError, ensure_ready, create_job, get_job
from .models import (
    BoundingBox,
    CandidateOut,
    CaptionModelOut,
    CaptionStepRequest,
    CaptionStepResponse,
    ClipScoreOut,
    ClipScoreStepResponse,
    DetectedObjectOut,
    DetectStepResponse,
    EvidenceOut,
    FinalStepResponse,
    FormulaBreakdownOut,
    HallucinationOut,
    JobSnapshot,
    ModelsResponse,
    ObjectVoteOut,
    RankingOut,
    RerankStepResponse,
    RetrievalStatusOut,
    RetrievedCaptionOut,
    RetrieveStepResponse,
    StepInfo,
    UploadResponse,
)
from .pipeline.retrieval import apply_retrieval_signals, logit_bias
from .pipeline.schema import CaptionCandidate, DetectedObject, RerankResult

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
resources_router = APIRouter(prefix="/api", tags=["resources"])

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Conversion helpers: pipeline dataclasses -> API (pydantic) shapes
# ---------------------------------------------------------------------------

def _candidates_out(candidates: list[CaptionCandidate]) -> list[CandidateOut]:
    return [
        CandidateOut(
            id=idx,
            text=c.text,
            source=c.source,
            generation_score=c.generation_score,
            clip_score=c.clip_score,
        )
        for idx, c in enumerate(candidates)
    ]


def _objects_out(objects: list[DetectedObject]) -> list[DetectedObjectOut]:
    out = []
    for obj in objects:
        box = None
        if obj.box_xyxy is not None:
            x1, y1, x2, y2 = obj.box_xyxy
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        out.append(DetectedObjectOut(label=obj.label, confidence=obj.confidence, box=box))
    return out


def _clip_scores_out(candidates: list[CaptionCandidate]) -> list[ClipScoreOut]:
    out = []
    for idx, c in enumerate(candidates):
        score = c.clip_score if c.clip_score is not None else 0.0
        percent = max(0.0, min(100.0, score * 100))
        out.append(ClipScoreOut(candidate_id=idx, text=c.text, clip_score=score, percent=round(percent, 1)))
    return out


def _retrieve_out(job: JobState) -> RetrieveStepResponse:
    bank = engines.retrieval
    return RetrieveStepResponse(
        neighbours=[
            RetrievedCaptionOut(index=n.index, caption=n.caption, similarity=round(n.similarity, 4), objects=n.objects)
            for n in job.retrieval.neighbours
        ],
        votes=[
            ObjectVoteOut(object=obj, votes=votes, in_evidence=votes >= bank.min_votes)
            for obj, votes in job.retrieval.votes.items()
        ],
        k=bank.k,
        min_votes=bank.min_votes,
        latency_ms=round(job.latency_ms.get("retrieve", 0.0), 1),
    )


def _rerank_out(job: JobState) -> RerankStepResponse:
    result: RerankResult = job.rerank_result
    scored = job.scored_candidates or []
    ranking: list[RankingOut] = []
    for item in result.ranked:
        try:
            candidate_id = scored.index(item.caption)
        except ValueError:
            candidate_id = -1
        breakdown = None
        if item.formula_breakdown is not None:
            fb = item.formula_breakdown
            breakdown = FormulaBreakdownOut(
                s_base=fb.s_base,
                s_align=fb.s_align,
                s_belief=fb.s_belief,
                s_obj=fb.s_obj,
                s_retr=fb.s_retr,
                formula_score=fb.formula_score,
            )
        ranking.append(
            RankingOut(
                candidate_id=candidate_id,
                rank=item.rank,
                score=item.rerank_score,
                text=item.caption.text,
                formula_breakdown=breakdown,
            )
        )
    hallucinations = [
        HallucinationOut(candidate_id=flag.candidate_index, word=flag.word, note=flag.note)
        for flag in result.hallucinations
    ]
    return RerankStepResponse(
        ranking=ranking,
        selected_caption=result.selected_caption,
        hallucinations=hallucinations,
        model_name=result.model_name,
        retrieval_signals=any(
            r.formula_breakdown is not None and r.formula_breakdown.s_obj is not None for r in result.ranked
        ),
        latency_ms=job.latency_ms.get("rerank", 0.0),
    )


def _job_selected_candidate_index(job: JobState) -> int:
    result = job.rerank_result
    scored = job.scored_candidates or []
    if result and result.ranked:
        try:
            return scored.index(result.ranked[0].caption)
        except ValueError:
            return -1
    return -1


def _final_out(job: JobState) -> FinalStepResponse:
    result: RerankResult = job.rerank_result
    best = result.ranked[0] if result and result.ranked else None
    final_caption = best.caption.text if best else (result.selected_caption if result else "")
    final_clip_score = best.caption.clip_score if best else None
    selected_idx = _job_selected_candidate_index(job)
    violations = [f for f in (result.hallucinations if result else []) if f.candidate_index == selected_idx]
    status_text = (
        "0 Detected Objects Violation (An toan)"
        if not violations
        else f"{len(violations)} Detected Objects Violation ({', '.join(sorted({v.word for v in violations}))})"
    )
    total_latency = sum(
        job.latency_ms.get(step, 0.0) for step in ("detect", "retrieve", "caption", "clipscore", "rerank")
    )
    return FinalStepResponse(
        final_caption=final_caption or "",
        final_clip_score=final_clip_score,
        hallucination_violations=len(violations),
        hallucination_status=status_text,
        total_latency_ms=round(total_latency, 1),
    )


def _job_snapshot(job: JobState) -> JobSnapshot:
    steps = {
        step: StepInfo(status=status, latency_ms=job.latency_ms.get(step), error=job.error.get(step))
        for step, status in job.status.items()
    }
    return JobSnapshot(
        job_id=job.job_id,
        image_url=f"/api/jobs/{job.job_id}/image",
        image_data_url=job.image_meta.get("image_data_url"),
        image_meta=job.image_meta,
        steps=steps,
        candidates=_candidates_out(job.scored_candidates or job.candidates) if (job.candidates or job.scored_candidates) else None,
        objects=_objects_out(job.objects) if job.objects is not None else None,
        retrieval=_retrieve_out(job) if job.retrieval is not None else None,
        scores=_clip_scores_out(job.scored_candidates) if job.scored_candidates else None,
        rerank=_rerank_out(job) if job.rerank_result is not None else None,
        final=_final_out(job) if job.status.get("final") == "completed" else None,
    )


def _get_job_or_404(job_id: str) -> JobState:
    try:
        return get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


def _guard_step(job: JobState, step) -> None:
    try:
        ensure_ready(job, step)
    except StepNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job.status.get(step) == "running":
        raise HTTPException(status_code=409, detail=f"Step '{step}' is already running.")


# ---------------------------------------------------------------------------
# Shared resources: caption models and the retrieval bank
# ---------------------------------------------------------------------------

@resources_router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(
        models=[
            CaptionModelOut(
                id=spec.id,
                label=spec.label,
                description=spec.description,
                default_delta=spec.default_delta,
                loaded=engines.is_model_loaded(spec.id),
            )
            for spec in engines.caption_models.values()
        ],
        default_model_id=engines.default_model_id,
    )


@resources_router.get("/retrieval", response_model=RetrievalStatusOut)
def retrieval_status() -> RetrievalStatusOut:
    if engines.retrieval is None:
        raise HTTPException(status_code=503, detail="Pipeline engines are still starting up.")
    return RetrievalStatusOut(**engines.retrieval.status())


@resources_router.post("/retrieval/load", response_model=RetrievalStatusOut)
def load_retrieval_bank() -> RetrievalStatusOut:
    if engines.retrieval is None:
        raise HTTPException(status_code=503, detail="Pipeline engines are still starting up.")
    try:
        engines.retrieval.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không nạp được kho truy hồi: {exc}") from exc
    return RetrievalStatusOut(**engines.retrieval.status())


# ---------------------------------------------------------------------------
# Step 1: upload
# ---------------------------------------------------------------------------

@router.post("", response_model=UploadResponse)
async def upload_image(image: UploadFile = File(...)) -> UploadResponse:
    if not engines.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline engines are still starting up.")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = mimetypes.guess_extension(image.content_type or "") or ".jpg"

    job = create_job(image_path=Path("__pending__"), image_meta={})
    dest = UPLOAD_DIR / f"{job.job_id}{suffix}"
    dest.write_bytes(raw)

    try:
        with Image.open(dest) as pil_image:
            width, height = pil_image.size
            fmt = pil_image.format or suffix.lstrip(".").upper()
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Not a valid image: {exc}") from exc

    job.image_path = dest
    job.image_meta = {
        "width": width,
        "height": height,
        "format": fmt,
        "size_bytes": len(raw),
        "filename": image.filename or dest.name,
        "image_data_url": _image_data_url(raw, fmt, image.content_type),
    }

    return UploadResponse(
        job_id=job.job_id,
        image_url=f"/api/jobs/{job.job_id}/image",
        image_data_url=job.image_meta["image_data_url"],
        width=width,
        height=height,
        format=fmt,
        size_bytes=len(raw),
        filename=job.image_meta["filename"],
    )


def _image_data_url(raw: bytes, fmt: str, content_type: str | None) -> str:
    mime_type = content_type or mimetypes.types_map.get(f".{fmt.lower()}", "image/jpeg")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@router.get("/{job_id}/image")
def get_image(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    return FileResponse(job.image_path)


@router.get("/{job_id}", response_model=JobSnapshot)
def get_job_snapshot(job_id: str) -> JobSnapshot:
    job = _get_job_or_404(job_id)
    return _job_snapshot(job)


# ---------------------------------------------------------------------------
# Step 2B: caption retrieval (top-K train captions from the bank)
# ---------------------------------------------------------------------------

@router.post("/{job_id}/steps/retrieve", response_model=RetrieveStepResponse)
def run_retrieve_step(job_id: str) -> RetrieveStepResponse:
    job = _get_job_or_404(job_id)
    _guard_step(job, "retrieve")
    if not engines.retrieval.loaded:
        raise HTTPException(status_code=409, detail="Kho truy hồi chưa được nạp. Bấm \"Nạp kho\" ở node Kho truy hồi.")

    job.status["retrieve"] = "running"
    start = time.perf_counter()
    try:
        job.retrieval = engines.retrieval.retrieve(job.image_path)
        job.latency_ms["retrieve"] = (time.perf_counter() - start) * 1000
        job.status["retrieve"] = "completed"
        job.error.pop("retrieve", None)
    except Exception as exc:
        job.status["retrieve"] = "failed"
        job.error["retrieve"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Retrieve step failed: {exc}") from exc

    return _retrieve_out(job)


# ---------------------------------------------------------------------------
# Step 3: caption candidates (plain beam + diverse beam search, with logit blocking)
# ---------------------------------------------------------------------------

@router.post("/{job_id}/steps/caption", response_model=CaptionStepResponse)
def run_caption_step(job_id: str, body: CaptionStepRequest | None = None) -> CaptionStepResponse:
    job = _get_job_or_404(job_id)
    _guard_step(job, "caption")

    model_id = (body.model_id if body else None) or engines.default_model_id
    if model_id not in engines.caption_models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    spec = engines.caption_models[model_id]
    delta = body.delta if body and body.delta is not None else spec.default_delta

    job.status["caption"] = "running"
    # New candidates make every later result stale.
    for step in ("clipscore", "rerank", "final"):
        job.status[step] = "idle"
        job.latency_ms.pop(step, None)
    job.scored_candidates = None
    job.rerank_result = None
    start = time.perf_counter()
    try:
        generator = engines.get_generator(model_id)
        captioner = generator.local_captioner()
        evidence = engines.retrieval.evidence(job.objects or [], job.retrieval, engines.config.yolo.confidence_threshold)
        bias, blocked = logit_bias(engines.lexicon, captioner.vocab, evidence, delta, captioner.device)
        candidates = generator.generate(job.image_path, job.objects or [], logit_bias=bias)
        job.candidates = candidates
        job.evidence, job.model_id, job.delta, job.blocked_objects = evidence, model_id, delta, blocked
        latency = (time.perf_counter() - start) * 1000
        job.latency_ms["caption"] = latency
        job.status["caption"] = "completed"
        job.error.pop("caption", None)
    except Exception as exc:
        job.status["caption"] = "failed"
        job.error["caption"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Caption step failed: {exc}") from exc

    return CaptionStepResponse(
        candidates=_candidates_out(candidates),
        model_id=model_id,
        model_label=spec.label,
        delta=delta,
        evidence=EvidenceOut(
            detected=sorted(evidence.detected), retrieved=sorted(evidence.retrieved), objects=sorted(evidence.objects)
        ),
        blocked_objects=blocked if delta > 0 else [],
        latency_ms=round(latency, 1),
    )


# ---------------------------------------------------------------------------
# Step 2A: object detection (YOLO-World)
# ---------------------------------------------------------------------------

@router.post("/{job_id}/steps/detect", response_model=DetectStepResponse)
def run_detect_step(job_id: str) -> DetectStepResponse:
    job = _get_job_or_404(job_id)
    _guard_step(job, "detect")

    job.status["detect"] = "running"
    start = time.perf_counter()
    try:
        objects = engines.detector.extract(job.image_path)
        job.objects = objects
        latency = (time.perf_counter() - start) * 1000
        job.latency_ms["detect"] = latency
        job.status["detect"] = "completed"
        job.error.pop("detect", None)
    except Exception as exc:
        job.status["detect"] = "failed"
        job.error["detect"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Detect step failed: {exc}") from exc

    return DetectStepResponse(objects=_objects_out(objects), latency_ms=round(latency, 1))


# ---------------------------------------------------------------------------
# Step 4: CLIPScore evaluator
# ---------------------------------------------------------------------------

@router.post("/{job_id}/steps/clipscore", response_model=ClipScoreStepResponse)
def run_clipscore_step(job_id: str) -> ClipScoreStepResponse:
    job = _get_job_or_404(job_id)
    _guard_step(job, "clipscore")

    job.status["clipscore"] = "running"
    start = time.perf_counter()
    try:
        scored = engines.scorer.score(job.image_path, job.candidates or [])
        job.scored_candidates = scored
        latency = (time.perf_counter() - start) * 1000
        job.latency_ms["clipscore"] = latency
        job.status["clipscore"] = "completed"
        job.error.pop("clipscore", None)
    except Exception as exc:
        job.status["clipscore"] = "failed"
        job.error["clipscore"] = str(exc)
        raise HTTPException(status_code=500, detail=f"CLIPScore step failed: {exc}") from exc

    return ClipScoreStepResponse(scores=_clip_scores_out(scored), latency_ms=round(latency, 1))


# ---------------------------------------------------------------------------
# Step 5: five-signal reranking + hallucination check
# ---------------------------------------------------------------------------

@router.post("/{job_id}/steps/rerank", response_model=RerankStepResponse)
def run_rerank_step(job_id: str) -> RerankStepResponse:
    job = _get_job_or_404(job_id)
    _guard_step(job, "rerank")

    job.status["rerank"] = "running"
    job.status["final"] = "idle"
    start = time.perf_counter()
    try:
        scored = job.scored_candidates or []
        result = engines.reranker.rerank(scored, job.objects or [])
        # S_obj and S_retr go on top of the three-signal formula (the proposed method).
        if result.ranked and job.evidence is not None and engines.retrieval.loaded:
            cfg = engines.config.retrieval
            ranked, flags = apply_retrieval_signals(
                result.ranked, scored, engines.retrieval, job.retrieval, job.evidence, cfg.w_obj, cfg.w_retr
            )
            result = replace(
                result,
                ranked=ranked,
                hallucinations=flags,
                selected_caption=ranked[0].caption.text,
                model_name="công thức năm tín hiệu (S_base, S_align, S_belief, S_obj, S_retr)",
            )
        job.rerank_result = result
        latency = (time.perf_counter() - start) * 1000
        job.latency_ms["rerank"] = latency
        job.status["rerank"] = "completed"
        job.error.pop("rerank", None)
    except Exception as exc:
        job.status["rerank"] = "failed"
        job.error["rerank"] = str(exc)
        raise HTTPException(status_code=500, detail=f"Rerank step failed: {exc}") from exc

    return _rerank_out(job)


# ---------------------------------------------------------------------------
# Step 6: final output & evaluation
# ---------------------------------------------------------------------------

@router.get("/{job_id}/steps/final", response_model=FinalStepResponse)
def get_final_step(job_id: str) -> FinalStepResponse:
    job = _get_job_or_404(job_id)
    _guard_step(job, "final")
    job.status["final"] = "completed"
    return _final_out(job)
