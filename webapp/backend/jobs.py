"""In-memory job store.

Each uploaded image becomes one job. Every pipeline step (detect, retrieve, caption, clipscore,
rerank, final) is triggered by its own API call and writes its result onto the same JobState -- this is
what lets the React Flow UI run the graph one node at a time instead of all-at-once. State lives
only in RAM (fine for a single-user thesis demo); it is gone on backend restart.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .pipeline.retrieval import Evidence, RetrievalResult
from .pipeline.schema import CaptionCandidate, DetectedObject, RerankResult

StepKey = Literal["upload", "detect", "retrieve", "caption", "clipscore", "rerank", "final"]
StepStatus = Literal["idle", "running", "completed", "failed"]

STEP_ORDER: tuple[StepKey, ...] = ("upload", "detect", "retrieve", "caption", "clipscore", "rerank", "final")

# Which steps must be "completed" before a given step is allowed to run. Mirrors the branching
# flow: detect (YOLO) and retrieve (caption bank) both only need the uploaded image; caption needs
# both, because the logit-blocking rule uses their evidence E(I); clipscore needs the candidates;
# rerank needs everything merged together; final needs rerank.
STEP_DEPENDENCIES: dict[StepKey, tuple[StepKey, ...]] = {
    "upload": (),
    "detect": ("upload",),
    "retrieve": ("upload",),
    "caption": ("detect", "retrieve"),
    "clipscore": ("caption",),
    "rerank": ("detect", "retrieve", "clipscore"),
    "final": ("rerank",),
}


@dataclass
class JobState:
    job_id: str
    image_path: Path
    image_meta: dict

    objects: list[DetectedObject] | None = None
    retrieval: RetrievalResult | None = None
    evidence: Evidence | None = None
    model_id: str | None = None
    delta: float | None = None
    blocked_objects: list[str] | None = None
    candidates: list[CaptionCandidate] | None = None
    scored_candidates: list[CaptionCandidate] | None = None
    rerank_result: RerankResult | None = None

    status: dict[StepKey, StepStatus] = field(
        default_factory=lambda: {
            "upload": "completed",
            "detect": "idle",
            "retrieve": "idle",
            "caption": "idle",
            "clipscore": "idle",
            "rerank": "idle",
            "final": "idle",
        }
    )
    latency_ms: dict[StepKey, float] = field(default_factory=dict)
    error: dict[StepKey, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobNotFoundError(KeyError):
    pass


class StepNotReadyError(RuntimeError):
    def __init__(self, step: StepKey, missing: list[StepKey]) -> None:
        self.step = step
        self.missing = missing
        super().__init__(f"Step '{step}' requires completed steps: {', '.join(missing)}")


_JOBS: dict[str, JobState] = {}
_STORE_LOCK = threading.Lock()


def create_job(image_path: Path, image_meta: dict) -> JobState:
    job_id = uuid.uuid4().hex[:12]
    job = JobState(job_id=job_id, image_path=image_path, image_meta=image_meta)
    with _STORE_LOCK:
        _JOBS[job_id] = job
    return job


def get_job(job_id: str) -> JobState:
    job = _JOBS.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    return job


def ensure_ready(job: JobState, step: StepKey) -> None:
    missing = [dep for dep in STEP_DEPENDENCIES[step] if job.status.get(dep) != "completed"]
    if missing:
        raise StepNotReadyError(step, missing)
