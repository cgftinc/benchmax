"""Data model for QA items as they flow through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmax.rag.qa_generation.models import QADataPoint


@dataclass
class FilterVerdict:
    """Result of a filter step applied to a GeneratedQA item.

    This is the boundary type between Filter and Regenerator — it carries
    rich metadata that the Regenerator needs to decide how to refine an item.

    Attributes:
        status: One of "passed", "rejected", or "needs_refinement".
        reason: Machine-readable code (e.g. "too_easy", "incorrect_answer").
        reasoning: Human-readable explanation of why this verdict was given.
        metadata: Pipeline-specific data (for example, refinement hints and
            retrieval diagnostics).
    """

    status: str  # "passed" | "rejected" | "needs_refinement"
    reason: str
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedQA:
    """A QA item flowing through the pipeline, accumulating metadata at each step.

    Wraps an existing ``QADataPoint`` and carries filter verdicts and
    regeneration history so that downstream steps can make informed decisions.

    Attributes:
        qa: The underlying question-answer data point.
        generation_metadata: Arbitrary metadata from the generation step.
        filter_verdict: Set by a Filter; read (and cleared) by a Regenerator.
        regeneration_history: Append-only log of past regeneration attempts.
    """

    qa: QADataPoint
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    filter_verdict: FilterVerdict | None = None
    regeneration_history: list[dict[str, Any]] = field(default_factory=list)
    journey_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_passed(self) -> bool:
        """True when the item has been explicitly marked as passed by a filter."""
        return self.filter_verdict is not None and self.filter_verdict.status == "passed"

    @property
    def is_rejected(self) -> bool:
        """True when the item has been explicitly rejected by a filter."""
        return self.filter_verdict is not None and self.filter_verdict.status == "rejected"

    @property
    def needs_refinement(self) -> bool:
        """True when a filter has flagged this item for regeneration."""
        return self.filter_verdict is not None and self.filter_verdict.status == "needs_refinement"

    def to_qa_data_point(self) -> QADataPoint:
        """Return the underlying QADataPoint."""
        return self.qa

    def resolve_effective_qa_type(self) -> str:
        """Return the effective QA type implied by the item's reference structure.

        Lookup items with ≥2 reference chunks are treated as multi_hop;
        multi_hop items with ≤1 reference chunk are treated as lookup.
        """
        qa_type = str(self.qa.get("qa_type", "")).strip().lower()
        if not qa_type:
            qa_type = (
                str(self.generation_metadata.get("qa_type_target", "")).strip().lower() or "lookup"
            )
        ref_chunks = list(self.qa.get("reference_chunks", []) or [])
        if qa_type == "lookup" and len(ref_chunks) >= 2:
            return "multi_hop"
        if qa_type == "multi_hop" and len(ref_chunks) <= 1:
            return "lookup"
        return qa_type or "lookup"

    def append_journey_event(
        self,
        *,
        stage: str,
        event_type: str,
        task_id: str = "",
        refinement_count: int | None = None,
        qa_type_before: str = "",
        qa_type_after: str = "",
        reason_code: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append a normalized item-lifecycle event for later analysis."""
        resolved_task_id = (
            str(task_id).strip() or str(self.generation_metadata.get("task_id", "")).strip()
        )
        if refinement_count is None:
            try:
                resolved_refinement_count = int(self.generation_metadata.get("refinement_count", 0))
            except (TypeError, ValueError):
                resolved_refinement_count = 0
        else:
            resolved_refinement_count = int(refinement_count)

        self.journey_events.append(
            {
                "stage": str(stage).strip(),
                "event_type": str(event_type).strip(),
                "task_id": resolved_task_id,
                "refinement_count": resolved_refinement_count,
                "qa_type_before": str(qa_type_before).strip(),
                "qa_type_after": str(qa_type_after).strip(),
                "reason_code": str(reason_code).strip(),
                "details": dict(details) if isinstance(details, dict) else {},
            }
        )
