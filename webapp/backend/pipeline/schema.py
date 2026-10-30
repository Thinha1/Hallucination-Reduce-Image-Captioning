from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class DetectedObject:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CaptionCandidate:
    text: str
    source: str
    generation_score: float | None = None
    clip_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormulaBreakdown:
    """Per-candidate S_total breakdown from FormulaReranker -- each value is already
    min-max normalized to [0, 1] across the candidate set (see formula_reranker.py), so
    they're directly comparable/plottable as-is. Only populated for mode="formula"."""

    s_base: float
    s_align: float
    s_belief: float
    # Retrieval signals of the proposed method (pipeline/retrieval.py), min-max normalized
    # the same way; None when the job ran without the retrieval bank.
    s_obj: float | None = None
    s_retr: float | None = None
    # Three-signal score before S_obj/S_retr were added.
    formula_score: float | None = None


@dataclass(frozen=True)
class RankedCaption:
    caption: CaptionCandidate
    rank: int
    rerank_score: float
    formula_breakdown: FormulaBreakdown | None = None


@dataclass(frozen=True)
class HallucinationFlag:
    """An object word in a candidate caption not supported by the object evidence."""

    candidate_index: int
    word: str
    note: str


@dataclass(frozen=True)
class RerankResult:
    ranked: list[RankedCaption]
    hallucinations: list[HallucinationFlag] = field(default_factory=list)
    model_name: str = ""
    selected_caption: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    image_path: Path
    objects: list[DetectedObject]
    candidates: list[CaptionCandidate]
    ranked: list[RankedCaption]
    hallucinations: list[HallucinationFlag] = field(default_factory=list)
    model_name: str = ""

    @property
    def best_caption(self) -> str:
        if not self.ranked:
            return ""
        return self.ranked[0].caption.text
