"""Pipeline metrics collection for CgftPipeline."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageMetrics:
    """Metrics for one pipeline stage execution."""

    stage_name: str
    wall_time_seconds: float = 0.0
    items_in: int = 0
    items_out_passed: int = 0
    items_out_rejected: int = 0
    items_out_needs_refinement: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    errors: int = 0


@dataclass
class BatchMetrics:
    """Metrics for one micro-batch iteration."""

    batch_index: int
    generated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    regeneration_count: int = 0
    acceptance_rate: float = 0.0
    cumulative_acceptance_rate: float = 0.0
    cumulative_fill_rate: float = 0.0
    stage_metrics: list[StageMetrics] = field(default_factory=list)


@dataclass
class PipelineMetrics:
    """Top-level metrics aggregator."""

    stages: dict[str, StageMetrics] = field(default_factory=dict)
    batch_history: list[BatchMetrics] = field(default_factory=list)
    rejection_by_stage: dict[str, int] = field(default_factory=dict)
    rejection_by_reason_code: dict[str, int] = field(default_factory=dict)
    wasted_llm_calls: int = 0
    wasted_cost_usd: float = 0.0
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_regenerations: int = 0
    overall_acceptance_rate: float = 0.0
    target_samples: int = 0
    fill_rate: float = 0.0
    wall_time_seconds: float = 0.0

    def merge_stage_metrics(self, other: PipelineMetrics) -> None:
        """Merge stage metrics from another PipelineMetrics (e.g. a batch)."""
        for name, stage in other.stages.items():
            existing = self.stages.get(name)
            if existing is None:
                self.stages[name] = dataclasses.replace(stage)
                continue
            existing.wall_time_seconds += stage.wall_time_seconds
            existing.items_in += stage.items_in
            existing.items_out_passed += stage.items_out_passed
            existing.items_out_rejected += stage.items_out_rejected
            existing.items_out_needs_refinement += stage.items_out_needs_refinement
            existing.llm_calls += stage.llm_calls
            existing.llm_input_tokens += stage.llm_input_tokens
            existing.llm_output_tokens += stage.llm_output_tokens
            existing.estimated_cost_usd += stage.estimated_cost_usd
            existing.errors += stage.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return dataclasses.asdict(self)


@contextmanager
def stage_timer(
    metrics: PipelineMetrics, stage_name: str, items_in_count: int
) -> Generator[StageMetrics, None, None]:
    stage = metrics.stages.setdefault(stage_name, StageMetrics(stage_name=stage_name))
    stage.items_in += items_in_count
    t0 = time.monotonic()
    yield stage
    stage.wall_time_seconds += time.monotonic() - t0
