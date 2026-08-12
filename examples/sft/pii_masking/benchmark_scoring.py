"""Align masked model output to source spans, and score it.

The model emits masked *text*; the metric consumes source *character spans*.
Bridging those is the whole job of this module, and it is done conservatively on
purpose.

Alignment accepts exactly one unambiguous monotonic mapping. Zero mappings and
several mappings are both invalid — there is deliberately no leftmost or fuzzy
tie-break, because either rule manufactures spans through arbitrary
post-processing and would quietly inflate the score. An invalid output
contributes no predicted spans at all: it lowers recall and shows up separately
in an invalid-output rate, so the conservatism is visible rather than hidden.

Metrics are label-agnostic character micro-averages. Per task, each document's
gold and predicted intervals are merged, integer counters accumulate across the
whole task, and division happens once at the end — micro, not macro, so a task's
score is not dominated by its shortest documents. The headline figure is the
unweighted mean of the two task scores, never a pooled corpus metric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Frozen grammar. Index zero is valid: `[NAME_0]` appears in real fixtures.
PLACEHOLDER_PATTERN = re.compile(r"\[(?:[A-Z][A-Z0-9]*)(?:_[0-9]+)?\]")

ALIGNMENT_VERSION = "pii-mask-alignment-v1"
SCORING_VERSION = "pii-mask-scoring-v1"

VALID = "valid"
INVALID_NO_ALIGNMENT = "invalid_no_alignment"
INVALID_AMBIGUOUS_ALIGNMENT = "invalid_ambiguous_alignment"

# Enumeration stops as soon as a second solution exists: the outcome is
# "ambiguous" either way, and some outputs admit combinatorially many.
_SOLUTION_LIMIT = 2

LITERAL = "literal"
PLACEHOLDER = "placeholder"


class ScoringError(ValueError):
    """A span or interval violates the scoring contract."""


# ── segmentation ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Segment:
    """One run of the model output: literal text, or a placeholder run."""

    kind: str
    text: str


def segment_output(output: str) -> list[Segment]:
    """Split output into alternating literal and placeholder-run segments.

    Adjacent placeholders collapse into a single run. Their individual
    boundaries are not recoverable from the text — `[A][B]` could split the
    covered characters anywhere — so treating them as one label-agnostic run is
    the only claim the output actually supports.
    """
    segments: list[Segment] = []
    position = 0
    for match in PLACEHOLDER_PATTERN.finditer(output):
        literal = output[position : match.start()]
        if literal:
            segments.append(Segment(LITERAL, literal))
        if segments and segments[-1].kind == PLACEHOLDER:
            pass  # adjacent placeholder: already covered by the open run
        else:
            segments.append(Segment(PLACEHOLDER, ""))
        position = match.end()

    tail = output[position:]
    if tail:
        segments.append(Segment(LITERAL, tail))
    return segments


# ── alignment ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Alignment:
    """The outcome of aligning one output against its source."""

    status: str
    intervals: tuple[tuple[int, int], ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.status == VALID


def _enumerate(
    source: str,
    segments: Sequence[Segment],
    index: int,
    position: int,
    spans: list[tuple[int, int]],
    found: list[list[tuple[int, int]]],
) -> None:
    if len(found) >= _SOLUTION_LIMIT:
        return
    if index == len(segments):
        if position == len(source):
            found.append(list(spans))
        return

    segment = segments[index]
    last = index == len(segments) - 1

    if segment.kind == LITERAL:
        if index == 0:
            # A leading literal is anchored at offset zero; it cannot float.
            if source.startswith(segment.text):
                _enumerate(source, segments, 1, len(segment.text), spans, found)
            return

        # Preceded by a placeholder run that must cover at least one character.
        search_from = position + 1
        if last:
            # A trailing literal is anchored at the end of the source.
            start = len(source) - len(segment.text)
            if start >= search_from and source.startswith(segment.text, start):
                spans.append((position, start))
                _enumerate(source, segments, index + 1, len(source), spans, found)
                spans.pop()
            return

        found_at = source.find(segment.text, search_from)
        while found_at != -1 and len(found) < _SOLUTION_LIMIT:
            spans.append((position, found_at))
            _enumerate(source, segments, index + 1, found_at + len(segment.text), spans, found)
            spans.pop()
            found_at = source.find(segment.text, found_at + 1)
        return

    # Placeholder run.
    if last:
        if len(source) > position:
            spans.append((position, len(source)))
            _enumerate(source, segments, index + 1, len(source), spans, found)
            spans.pop()
        return
    # Otherwise the following literal decides where this run ends.
    _enumerate(source, segments, index + 1, position, spans, found)


def align(source: str, output: str) -> Alignment:
    """Map one masked output back onto source character intervals.

    Returns exactly one alignment or an invalid status. Never guesses.
    """
    segments = segment_output(output)

    if not any(segment.kind == PLACEHOLDER for segment in segments):
        # A no-placeholder output claims the source contained nothing to mask,
        # which is only credible if it reproduced the source exactly.
        return Alignment(VALID, ()) if output == source else Alignment(INVALID_NO_ALIGNMENT)

    found: list[list[tuple[int, int]]] = []
    _enumerate(source, segments, 0, 0, [], found)

    if not found:
        return Alignment(INVALID_NO_ALIGNMENT)
    if len(found) > 1:
        return Alignment(INVALID_AMBIGUOUS_ALIGNMENT)
    return Alignment(VALID, tuple(found[0]))


# ── intervals ─────────────────────────────────────────────────────────────────
def check_intervals(intervals: Iterable[tuple[int, int]], length: int) -> None:
    """Require half-open intervals that stay inside a text of ``length``."""
    for start, end in intervals:
        if start < 0 or end > length:
            raise ScoringError(f"interval [{start}, {end}) falls outside a text of length {length}")
        if start >= end:
            raise ScoringError(f"interval [{start}, {end}) is empty or inverted")


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping *and* touching intervals.

    Touching intervals merge because the metric counts characters, not spans:
    [0,3) and [3,5) cover exactly the same characters as [0,5), and counting
    them separately would double-count nothing but complicate every later step.
    """
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def overlap_length(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> int:
    """Total characters covered by both merged interval lists."""
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


# ── counters and metrics ──────────────────────────────────────────────────────
@dataclass
class TaskCounters:
    """Integer accumulators for one task. Division happens once, at the end."""

    true_chars: int = 0
    predicted_chars: int = 0
    overlap_chars: int = 0
    text_chars: int = 0
    documents: int = 0
    invalid_documents: int = 0
    invalid_by_class: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        *,
        source_length: int,
        gold: Sequence[tuple[int, int]],
        predicted: Sequence[tuple[int, int]],
        status: str = VALID,
    ) -> None:
        """Accumulate one document."""
        check_intervals(gold, source_length)
        check_intervals(predicted, source_length)
        merged_gold = merge_intervals(gold)
        merged_predicted = merge_intervals(predicted)

        self.documents += 1
        self.text_chars += source_length
        self.true_chars += total_length(merged_gold)
        self.predicted_chars += total_length(merged_predicted)
        self.overlap_chars += overlap_length(merged_gold, merged_predicted)
        if status != VALID:
            self.invalid_documents += 1
            self.invalid_by_class[status] = self.invalid_by_class.get(status, 0) + 1

    @property
    def false_positive_chars(self) -> int:
        return self.predicted_chars - self.overlap_chars

    @property
    def non_pii_chars(self) -> int:
        return self.text_chars - self.true_chars


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta, defined as 0 when precision and recall are both 0."""
    if precision + recall == 0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def task_metrics(counters: TaskCounters) -> dict[str, Any]:
    """Micro-averaged metrics for one task, with explicit zero-denominators."""
    precision = (
        counters.overlap_chars / counters.predicted_chars if counters.predicted_chars else 0.0
    )
    recall = counters.overlap_chars / counters.true_chars if counters.true_chars else 0.0
    fpr = counters.false_positive_chars / counters.non_pii_chars if counters.non_pii_chars else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f2": f_beta(precision, recall, 2.0),
        "fpr": fpr,
        "support_chars": counters.true_chars,
        "predicted_chars": counters.predicted_chars,
        "overlap_chars": counters.overlap_chars,
        "text_chars": counters.text_chars,
        "documents": counters.documents,
        "invalid_documents": counters.invalid_documents,
        "invalid_rate": (
            counters.invalid_documents / counters.documents if counters.documents else 0.0
        ),
        "invalid_by_class": dict(sorted(counters.invalid_by_class.items())),
    }


_AVERAGED_KEYS = ("precision", "recall", "f1", "f2", "fpr", "invalid_rate")


def task_average(per_task: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Unweighted mean of already-micro-aggregated task metrics.

    Deliberately not a pooled corpus metric: pooling would let the larger task
    silently dominate the headline number.
    """
    if not per_task:
        return {key: 0.0 for key in _AVERAGED_KEYS}
    count = len(per_task)
    return {
        key: sum(float(metrics[key]) for metrics in per_task.values()) / count
        for key in _AVERAGED_KEYS
    }


def build_report(
    per_model: Mapping[str, Mapping[str, TaskCounters]],
    *,
    base_model_key: str = "base",
    sft_model_key: str = "sft",
) -> dict[str, Any]:
    """Assemble the source-free report, including base-to-SFT deltas."""
    report: dict[str, Any] = {
        "alignment_version": ALIGNMENT_VERSION,
        "scoring_version": SCORING_VERSION,
        "primary_metric": "f2",
        "models": {},
    }
    for model_key, tasks in sorted(per_model.items()):
        per_task = {task: task_metrics(counters) for task, counters in sorted(tasks.items())}
        report["models"][model_key] = {
            "tasks": per_task,
            "task_average": task_average(per_task),
        }

    models = report["models"]
    if base_model_key in models and sft_model_key in models:
        base = models[base_model_key]["task_average"]
        sft = models[sft_model_key]["task_average"]
        report["delta"] = {key: sft[key] - base[key] for key in _AVERAGED_KEYS}
    return report


def score(protocol: Any, protocol_dir: Any, *, final: bool = False) -> str:
    """Recompute the report from the prediction journal. Offline.

    Reads only what is already on disk: no network, no model calls, no source
    rows. A ``--final`` report additionally requires complete coverage, which the
    journal owner enforces.
    """
    from pathlib import Path

    journal = Path(protocol_dir) / "predictions.jsonl"
    if not journal.is_file():
        raise SystemExit(f"no prediction journal at {journal}; run `evaluate` before scoring")

    # Imported only once there is something to score, so the missing-journal
    # path stays a clean message rather than an import error.
    from .benchmark_inference import load_journal, score_journal

    return score_journal(protocol, load_journal(journal), final=final)
