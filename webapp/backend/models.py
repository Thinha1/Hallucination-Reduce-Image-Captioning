"""Pydantic response models for the API layer -- separate from the dataclasses in
pipeline/schema.py (those are the engine's internal representation)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    image_url: str
    image_data_url: str
    width: int
    height: int
    format: str
    size_bytes: int
    filename: str
    status: Literal["completed"] = "completed"


class CandidateOut(BaseModel):
    id: int
    text: str
    source: str
    generation_score: float | None = None
    clip_score: float | None = None


class CaptionModelOut(BaseModel):
    id: str
    label: str
    description: str
    default_delta: float
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[CaptionModelOut]
    default_model_id: str


class RetrievalStatusOut(BaseModel):
    loaded: bool
    loading: bool
    num_captions: int | None = None
    embedding_dim: int | None = None
    size_mb: int | None = None
    model_name: str
    k: int
    min_votes: int
    device: str
    load_ms: float | None = None
    error: str | None = None


class RetrievedCaptionOut(BaseModel):
    index: int
    caption: str
    similarity: float
    objects: list[str]


class ObjectVoteOut(BaseModel):
    object: str
    votes: int
    in_evidence: bool  # votes >= min_votes


class RetrieveStepResponse(BaseModel):
    neighbours: list[RetrievedCaptionOut]
    votes: list[ObjectVoteOut]
    k: int
    min_votes: int
    latency_ms: float


class EvidenceOut(BaseModel):
    detected: list[str]
    retrieved: list[str]
    objects: list[str]


class CaptionStepRequest(BaseModel):
    """Which checkpoint to caption with and the logit-blocking strength (None = the model's
    default from weights/models.json, 0 = no blocking)."""

    model_id: str | None = None
    delta: float | None = None


class CaptionStepResponse(BaseModel):
    candidates: list[CandidateOut]
    model_id: str
    model_label: str
    delta: float
    evidence: EvidenceOut
    blocked_objects: list[str]
    latency_ms: float


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObjectOut(BaseModel):
    label: str
    confidence: float
    box: BoundingBox | None = None


class DetectStepResponse(BaseModel):
    objects: list[DetectedObjectOut]
    latency_ms: float


class ClipScoreOut(BaseModel):
    candidate_id: int
    text: str
    clip_score: float
    percent: float


class ClipScoreStepResponse(BaseModel):
    scores: list[ClipScoreOut]
    latency_ms: float


class FormulaBreakdownOut(BaseModel):
    """Per-candidate S_total breakdown -- each value already normalized to [0, 1] across the
    candidate set."""

    s_base: float
    s_align: float
    s_belief: float
    s_obj: float | None = None
    s_retr: float | None = None
    formula_score: float | None = None


class RankingOut(BaseModel):
    candidate_id: int
    rank: int
    score: float
    text: str
    formula_breakdown: FormulaBreakdownOut | None = None


class HallucinationOut(BaseModel):
    candidate_id: int
    word: str
    note: str


class RerankStepResponse(BaseModel):
    ranking: list[RankingOut]
    selected_caption: str | None
    hallucinations: list[HallucinationOut]
    model_name: str
    # True when S_obj and S_retr were added on top of the three-signal formula.
    retrieval_signals: bool = False
    latency_ms: float


class FinalStepResponse(BaseModel):
    final_caption: str
    final_clip_score: float | None
    hallucination_violations: int
    hallucination_status: str
    total_latency_ms: float


class StepInfo(BaseModel):
    status: Literal["idle", "running", "completed", "failed"]
    latency_ms: float | None = None
    error: str | None = None


class JobSnapshot(BaseModel):
    job_id: str
    image_url: str
    image_data_url: str | None = None
    image_meta: dict
    steps: dict[str, StepInfo]
    candidates: list[CandidateOut] | None = None
    objects: list[DetectedObjectOut] | None = None
    retrieval: RetrieveStepResponse | None = None
    scores: list[ClipScoreOut] | None = None
    rerank: RerankStepResponse | None = None
    final: FinalStepResponse | None = None
