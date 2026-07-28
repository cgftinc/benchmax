"""Generic trace processing pipeline — provider-agnostic.

Operates on ``NormalizedTrace`` / ``TraceMessage`` objects produced by any
``TraceAdapter`` implementation.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from castform.traces.adapter import (
    DetectedSystemPrompt,
    DetectedTool,
    DetectedTools,
    NormalizedTrace,
    TraceMessage,
)

logger = logging.getLogger(__name__)

MIN_TRAIN_SAMPLES = 16

VALID_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_system_prompt(
    traces: list[NormalizedTrace],
) -> DetectedSystemPrompt | None:
    """Scan *all* traces for ``role=="system"`` messages.

    Returns the most common system prompt.  If multiple distinct prompts are
    found, ``variants`` contains the alternatives so callers can surface a
    warning.
    """
    prompts: list[str] = []
    for trace in traces:
        for msg in trace.messages:
            if msg.role == "system":
                prompts.append(msg.content)
                break  # only first system msg per trace

    if not prompts:
        return None

    counts = Counter(prompts)
    most_common, count = counts.most_common(1)[0]
    variants = [p for p, _ in counts.most_common() if p != most_common]

    return DetectedSystemPrompt(
        prompt=most_common,
        count=count,
        total_traces=len(traces),
        variants=variants,
    )


def detect_tools(traces: list[NormalizedTrace]) -> DetectedTools:
    """Scan all traces for ``tool_calls`` in assistant messages.

    Tool names that are not valid Python identifiers are silently dropped
    (they cannot be used in env codegen and could be injection vectors).
    """
    tool_data: dict[str, dict[str, Any]] = {}  # name → {count, args, keys}

    for trace in traces:
        for msg in trace.messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if not VALID_IDENTIFIER_RE.match(tc.name):
                    continue
                entry = tool_data.setdefault(
                    tc.name,
                    {"count": 0, "sample_args": [], "param_keys": set()},
                )
                entry["count"] += 1
                args = tc.arguments_dict()
                # validate argument keys
                safe_args = {k: v for k, v in args.items() if VALID_IDENTIFIER_RE.match(k)}
                if safe_args and len(entry["sample_args"]) < 3:
                    entry["sample_args"].append(safe_args)
                entry["param_keys"].update(safe_args.keys())

    tools = [
        DetectedTool(
            name=name,
            call_count=data["count"],
            sample_args=data["sample_args"],
            param_keys=data["param_keys"],
        )
        for name, data in sorted(tool_data.items())
    ]
    return DetectedTools(tools=tools)


# ---------------------------------------------------------------------------
# Windowing — traces → training examples
# ---------------------------------------------------------------------------


@dataclass
class TrainingExample:
    """One training example derived from a trace turn.

    Dual representation:
    - ``prompt_messages`` / ``completion_messages`` — structured
      ``TraceMessage`` lists.  These are the authoritative training data,
      used by ``to_jsonl_dict()`` and consumed by a BaseEnv dataset parser.
    - ``prompt`` / ``ground_truth`` — human-readable summaries for display.
      These are lossy (tool call arguments are flattened to text) and
      should NOT be used for training.
    """

    prompt_messages: list[TraceMessage]
    completion_messages: list[TraceMessage]
    prompt: str
    ground_truth: str
    trace_id: str
    turn_index: int
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Lossless serialisation for intermediate pipeline storage (blob).

        Unlike ``to_jsonl_dict()`` (which is lossy and shaped for the trainer),
        this preserves all fields for roundtripping through ``from_dict()``.
        """
        return {
            "prompt_messages": [m.to_dict() for m in self.prompt_messages],
            "completion_messages": [m.to_dict() for m in self.completion_messages],
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "trace_id": self.trace_id,
            "turn_index": self.turn_index,
            "scores": self.scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingExample:
        """Reconstruct from a dict produced by ``to_dict()``."""
        from castform.traces.adapter import normalize_message

        return cls(
            prompt_messages=[normalize_message(m) for m in data["prompt_messages"]],
            completion_messages=[normalize_message(m) for m in data["completion_messages"]],
            prompt=data["prompt"],
            ground_truth=data["ground_truth"],
            trace_id=data["trace_id"],
            turn_index=data["turn_index"],
            scores=data.get("scores", {}),
            metadata=data.get("metadata", {}),
        )

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialise to the JSONL schema consumed by a BaseEnv dataset parser.

        Both ``prompt_messages`` and ``ground_truth`` use ``to_dict()``
        with structured ``tool_calls`` (arguments as JSON strings). The
        trainer handles Arrow compatibility at template time by parsing
        argument strings to dicts before ``apply_chat_template``.

        Schema::

            {
                "prompt_messages": list[{role, content, tool_calls, ...}],
                "ground_truth": {role, content, tool_calls, ...},
                "rollout_args": {
                    "trace_id": str,
                    "turn_index": int,
                    "total_messages": int,
                    "scores": dict[str, float],
                    "raw_prompt": str,
                },
            }
        """
        gt = self.completion_messages[0].to_dict() if self.completion_messages else {}
        return {
            "prompt_messages": [m.to_dict() for m in self.prompt_messages],
            "ground_truth": gt,
            "rollout_args": {
                "trace_id": self.trace_id,
                "turn_index": self.turn_index,
                "total_messages": len(self.prompt_messages) + len(self.completion_messages),
                "scores": self.scores,
                "raw_prompt": self.prompt,
            },
        }


def _serialize_messages(messages: list[TraceMessage]) -> str:
    """Render a list of messages as a human-readable string."""
    parts: list[str] = []
    for m in messages:
        prefix = m.role.upper()
        if m.name:
            prefix += f" ({m.name})"
        parts.append(f"[{prefix}] {m.content}")
        if m.tool_calls:
            for tc in m.tool_calls:
                parts.append(f"  → tool_call: {tc.name}({tc.arguments})")
    return "\n".join(parts)


def build_training_examples(
    traces: list[NormalizedTrace],
    *,
    include_system_prompt: bool = False,
) -> list[TrainingExample]:
    """Window each trace's messages into prompt/completion pairs.

    For each assistant turn in the conversation:
    - prompt = all messages before this assistant turn
    - completion = this assistant turn + any immediately following tool
      response messages (up to the next user or assistant message)

    Traces with errors are skipped.
    """
    examples: list[TrainingExample] = []

    for trace in traces:
        if trace.errors:
            continue

        msgs = trace.messages
        if not msgs:
            continue

        # Find indices of assistant messages
        turn_index = 0
        i = 0
        while i < len(msgs):
            if msgs[i].role != "assistant":
                i += 1
                continue

            # prompt = everything before this assistant turn
            prompt_start = 0
            if not include_system_prompt and msgs[0].role == "system":
                prompt_start = 1
            prompt_msgs = list(msgs[prompt_start:i])

            # completion = this assistant msg + following tool responses
            completion_msgs: list[TraceMessage] = [msgs[i]]
            j = i + 1
            while j < len(msgs) and msgs[j].role == "tool":
                completion_msgs.append(msgs[j])
                j += 1

            if prompt_msgs:
                examples.append(
                    TrainingExample(
                        prompt_messages=prompt_msgs,
                        completion_messages=completion_msgs,
                        prompt=_serialize_messages(prompt_msgs),
                        ground_truth=_serialize_messages(completion_msgs),
                        trace_id=trace.id,
                        turn_index=turn_index,
                        scores=dict(trace.scores),
                        metadata=dict(trace.metadata),
                    )
                )

            turn_index += 1
            i = j

    return examples


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DropReason:
    """Structured drop reason for filtered examples."""

    filter: str  # stage: "heuristic", "tool_relay", "dedup", "tool_calls"
    reason: str  # specific: "too_short", "relay", "duplicate", "excluded"
    detail: str | None = None  # optional: cluster id, tool name, etc.

    def __str__(self) -> str:
        if self.detail:
            return f"{self.filter}:{self.reason}:{self.detail}"
        return f"{self.filter}:{self.reason}"


@dataclass
class FilterResult:
    """Result of applying filters."""

    kept: list[TrainingExample]
    dropped: list[tuple[TrainingExample, DropReason]]

    @property
    def summary(self) -> dict[str, int]:
        """Counts per filter stage."""
        counts: dict[str, int] = {}
        for _, reason in self.dropped:
            counts[reason.filter] = counts.get(reason.filter, 0) + 1
        return counts

    @property
    def summary_detail(self) -> dict[str, int]:
        """Counts per filter:reason pair."""
        counts: dict[str, int] = {}
        for _, reason in self.dropped:
            key = f"{reason.filter}:{reason.reason}"
            counts[key] = counts.get(key, 0) + 1
        return counts


def apply_heuristic_filters(
    examples: list[TrainingExample],
    *,
    min_completion_chars: int = 50,
) -> FilterResult:
    """Filter examples using deterministic heuristics.

    Returns ``FilterResult`` with both kept and dropped examples (with
    reasons) so callers can show what was filtered and why.
    """
    kept: list[TrainingExample] = []
    dropped: list[tuple[TrainingExample, DropReason]] = []

    for ex in examples:
        actual_content = ex.completion_messages[0].content if ex.completion_messages else ""
        if len(actual_content) < min_completion_chars:
            dropped.append((ex, DropReason("heuristic", "too_short")))
            continue
        kept.append(ex)

    return FilterResult(kept=kept, dropped=dropped)


def filter_by_tool_calls(
    examples: list[TrainingExample],
    exclude_tools: list[str],
) -> FilterResult:
    """Filter out examples whose completion consists solely of excluded tool calls.

    An example is dropped when every tool call in the completion matches
    ``exclude_tools`` and the assistant's text content is minimal (< 50
    chars).  Examples with substantive text or non-excluded tool calls are
    kept.

    Use ``detect_tools()`` to discover available tool names first, then
    pass the ones you want to exclude.
    """
    exclude_set = set(exclude_tools)
    kept: list[TrainingExample] = []
    dropped: list[tuple[TrainingExample, DropReason]] = []

    for ex in examples:
        tool_names: list[str] = []
        for msg in ex.completion_messages:
            if msg.tool_calls:
                tool_names.extend(tc.name for tc in msg.tool_calls)

        if not tool_names:
            kept.append(ex)
            continue

        all_excluded = all(name in exclude_set for name in tool_names)
        text_content = ex.completion_messages[0].content if ex.completion_messages else ""
        if all_excluded and len(text_content) < 50:
            dropped.append((ex, DropReason("tool_calls", "excluded", tool_names[0])))
            continue
        kept.append(ex)

    return FilterResult(kept=kept, dropped=dropped)


# ---------------------------------------------------------------------------
# Tool result relay filter
# ---------------------------------------------------------------------------


def _extract_text_from_value(value: Any) -> list[str]:
    """Recursively extract leaf string/number values from a JSON structure."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text_from_value(item))
        return parts
    if isinstance(value, dict):
        parts = []
        for v in value.values():
            parts.extend(_extract_text_from_value(v))
        return parts
    return []


def _tokenize_for_overlap(text: str) -> set[str]:
    """Lowercase, split on whitespace, drop single-char tokens."""
    return {w for w in text.lower().split() if len(w) > 1}


def _extract_tool_result_tokens(content: str) -> set[str]:
    """Extract tokens from a tool result, handling JSON payloads."""
    try:
        parsed = json.loads(content)
        text_parts = _extract_text_from_value(parsed)
        return _tokenize_for_overlap(" ".join(text_parts))
    except (json.JSONDecodeError, TypeError):
        return _tokenize_for_overlap(content)


def filter_tool_result_relay(
    examples: list[TrainingExample],
    *,
    overlap_threshold: float = 0.6,
) -> FilterResult:
    """Filter out turns that mostly relay a preceding tool result.

    An example is dropped when the assistant's text completion has high
    token overlap with any preceding tool result and the completion
    contains no tool calls of its own.
    """
    kept: list[TrainingExample] = []
    dropped: list[tuple[TrainingExample, DropReason]] = []

    for ex in examples:
        # Skip if completion has tool calls (it's making a new decision)
        has_tool_calls = any(msg.tool_calls for msg in ex.completion_messages)
        if has_tool_calls:
            kept.append(ex)
            continue

        # Collect all tool result messages from prompt
        tool_token_sets: list[set[str]] = []
        for msg in ex.prompt_messages:
            if msg.role == "tool" and msg.content:
                tool_token_sets.append(_extract_tool_result_tokens(msg.content))

        if not tool_token_sets:
            kept.append(ex)
            continue

        # Tokenize completion
        completion_text = ex.completion_messages[0].content if ex.completion_messages else ""
        completion_tokens = _tokenize_for_overlap(completion_text)

        if not completion_tokens:
            kept.append(ex)
            continue

        # Max overlap against any single tool result
        max_overlap = (
            max(
                len(completion_tokens & tool_tokens) / len(completion_tokens)
                for tool_tokens in tool_token_sets
                if tool_tokens
            )
            if any(tool_token_sets)
            else 0.0
        )

        if max_overlap > overlap_threshold:
            dropped.append((ex, DropReason("tool_relay", "relay")))
        else:
            kept.append(ex)

    return FilterResult(kept=kept, dropped=dropped)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS = [
    (re.compile(r"\S+@\S+"), "<EMAIL>"),
    (re.compile(r"\b\d+\b"), "<N>"),
]


_DEDUP_MAX_CHARS = 500


def _normalize_for_dedup(text: str) -> str:
    """Normalize and truncate completion text for dedup comparison."""
    text = text[:_DEDUP_MAX_CHARS].lower()
    for pattern, replacement in _ENTITY_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trigram_set(text: str) -> frozenset[str]:
    """Build a character trigram set from already-truncated text."""
    if len(text) < 3:
        return frozenset()
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between pre-computed trigram sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def deduplicate_completions(
    examples: list[TrainingExample],
    *,
    similarity_threshold: float = 0.85,
    max_per_cluster: int = 3,
) -> FilterResult:
    """Remove near-duplicate completions via trigram Jaccard clustering.

    Canonical sort by ``(trace_id, turn_index)`` ensures deterministic
    results regardless of input order.
    """
    # Canonical sort for determinism
    sorted_examples = sorted(examples, key=lambda ex: (ex.trace_id, ex.turn_index))

    # Pre-compute trigram sets once (capped at 500 chars each)
    trigrams: list[tuple[int, frozenset[str]]] = []
    for i, ex in enumerate(sorted_examples):
        text = ex.completion_messages[0].content if ex.completion_messages else ""
        trigrams.append((i, _trigram_set(_normalize_for_dedup(text))))

    # Greedy clustering against pre-computed sets.
    # Cap at 500 cluster reps to keep worst case O(n × 500) ≈ linear.
    max_cluster_reps = 500
    clusters: list[list[int]] = []
    cluster_reps: list[frozenset[str]] = []
    remaining_start: int | None = None

    for pos, (idx, tri) in enumerate(trigrams):
        placed = False
        for ci, rep in enumerate(cluster_reps):
            if _jaccard(tri, rep) > similarity_threshold:
                clusters[ci].append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
            cluster_reps.append(tri)
            if len(cluster_reps) >= max_cluster_reps:
                remaining_start = pos + 1
                break

    # Past the cap: keep remaining examples as-is (assumed unique)
    if remaining_start is not None:
        for idx, _ in trigrams[remaining_start:]:
            clusters.append([idx])

    # Keep max_per_cluster from each cluster
    keep_indices: set[int] = set()
    cluster_map: dict[int, int] = {}  # example index → cluster id
    for ci, cluster in enumerate(clusters):
        for i, idx in enumerate(cluster):
            if i < max_per_cluster:
                keep_indices.add(idx)
            else:
                cluster_map[idx] = ci

    kept: list[TrainingExample] = []
    dropped: list[tuple[TrainingExample, DropReason]] = []
    for i, ex in enumerate(sorted_examples):
        if i in keep_indices:
            kept.append(ex)
        else:
            dropped.append((ex, DropReason("dedup", "duplicate", f"cluster_{cluster_map[i]}")))

    return FilterResult(kept=kept, dropped=dropped)


# ---------------------------------------------------------------------------
# Outcome balance diagnostic
# ---------------------------------------------------------------------------


@dataclass
class OutcomeBalance:
    """Dataset outcome balance diagnostic."""

    total: int
    success_count: int
    failure_count: int
    unknown_count: int
    failure_fraction: float
    is_balanced: bool
    message: str | None


def check_outcome_balance(
    examples: list[TrainingExample],
    *,
    score_name: str = "task_success",
    success_threshold: float = 0.5,
    min_failure_fraction: float = 0.15,
) -> OutcomeBalance:
    """Check success/failure balance for GRPO training.

    An example is a success if ``scores[score_name] >= success_threshold``,
    failure if below.  Examples missing the score are ``unknown``.

    This is advisory — it does NOT filter.
    """
    success = 0
    failure = 0
    unknown = 0

    for ex in examples:
        score = ex.scores.get(score_name)
        if score is None:
            unknown += 1
        elif score >= success_threshold:
            success += 1
        else:
            failure += 1

    scored = success + failure
    fraction = failure / scored if scored else 0.0

    message = None
    if scored == 0:
        is_balanced = True
        message = f"No examples have '{score_name}' scores — balance check not applicable."
    elif fraction < min_failure_fraction:
        is_balanced = False
        message = (
            f"Only {fraction:.1%} failure traces (need >= {min_failure_fraction:.0%}). "
            f"GRPO needs reward variance — consider collecting more failure cases "
            f"or downsampling successes."
        )
    else:
        is_balanced = True

    return OutcomeBalance(
        total=len(examples),
        success_count=success,
        failure_count=failure,
        unknown_count=unknown,
        failure_fraction=fraction,
        is_balanced=is_balanced,
        message=message,
    )


# ---------------------------------------------------------------------------
# Filter composition
# ---------------------------------------------------------------------------

_DATASET_LEVEL_FILTERS = {"dedup"}


def apply_filters(
    examples: list[TrainingExample],
    steps: list[tuple[str, dict[str, Any]]],
) -> FilterResult:
    """Run named filters in sequence, accumulate all drops.

    Each step is ``(filter_name, kwargs_dict)``.  Dataset-level filters
    (e.g. ``"dedup"``) must come after all per-example filters.

    Supported filters: ``"heuristic"``, ``"tool_relay"``, ``"tool_calls"``,
    ``"dedup"``.

    Raises ``ValueError`` if a per-example filter is listed after a
    dataset-level filter or if a filter name is unknown.
    """
    registry: dict[str, Callable[..., FilterResult]] = {
        "heuristic": apply_heuristic_filters,
        "tool_relay": filter_tool_result_relay,
        "tool_calls": filter_by_tool_calls,
        "dedup": deduplicate_completions,
    }

    # Validate ordering: per-example filters must precede dataset-level
    seen_dataset = False
    for name, _ in steps:
        if name not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown filter {name!r}. Available: {available}")
        if name in _DATASET_LEVEL_FILTERS:
            seen_dataset = True
        elif seen_dataset:
            raise ValueError(
                f"Per-example filter {name!r} listed after dataset-level filter. "
                f"Reorder so per-example filters run first."
            )

    all_dropped: list[tuple[TrainingExample, DropReason]] = []
    current = list(examples)

    for name, kwargs in steps:
        result = registry[name](current, **kwargs)
        all_dropped.extend(result.dropped)
        current = result.kept

    return FilterResult(kept=current, dropped=all_dropped)


def apply_score_filter(
    examples: list[TrainingExample],
    score_name: str,
    *,
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[TrainingExample]:
    """Filter examples by provider-reported scores."""
    result: list[TrainingExample] = []
    for ex in examples:
        score = ex.scores.get(score_name)
        if score is None:
            continue
        if min_score is not None and score < min_score:
            continue
        if max_score is not None and score > max_score:
            continue
        result.append(ex)
    return result


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    examples: list[TrainingExample],
    train_count: int,
    eval_count: int,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into train/eval and serialise to JSONL-ready dicts.

    Shuffles with a fixed seed before splitting to avoid temporal data
    leakage (traces are fetched chronologically).

    Raises ``ValueError`` if ``train_count < MIN_TRAIN_SAMPLES`` or if there
    are not enough examples.
    """
    import random

    if train_count < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAIN_SAMPLES} training examples, got train_count={train_count}"
        )
    total = train_count + eval_count
    if len(examples) < total:
        raise ValueError(
            f"Not enough examples: need {total} (train={train_count} + "
            f"eval={eval_count}), have {len(examples)}"
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train = [ex.to_jsonl_dict() for ex in shuffled[:train_count]]
    eval_ = [ex.to_jsonl_dict() for ex in shuffled[train_count : train_count + eval_count]]
    return train, eval_
