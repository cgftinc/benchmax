"""Deterministic reward primitives that require no model judge."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

Completion = str | Sequence[Mapping[str, Any]]

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SOURCE_CITATION = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)


def extract_completion_text(completion: Completion) -> str:
    """Return the final assistant answer from a completion.

    Message transcripts represent an answer only when their final message is
    an assistant message. Earlier assistant turns are tool-use history, not the
    final answer being rewarded.
    """

    if isinstance(completion, str):
        return completion.strip()
    if not completion:
        return ""
    final = completion[-1]
    if not isinstance(final, Mapping) or final.get("role") != "assistant":
        return ""
    content = final.get("content")
    return content.strip() if isinstance(content, str) else ""


def extract_answer_block(text: str) -> str:
    """Return the first ``<answer>`` block, or the full stripped text."""

    match = _ANSWER_TAG.search(text or "")
    return (match.group(1) if match else text).strip()


def clip01(value: Any) -> float:
    """Convert ``value`` to a float and clamp it to ``[0, 1]``."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def percent_of_text_a_in_text_b(text_a: str, text_b: str) -> float:
    """Return the fraction of characters in ``text_a`` matched by ``text_b``."""

    if not text_a:
        return 0.0
    matcher = SequenceMatcher(None, text_a, text_b)
    matched = sum(size for _, _, size in matcher.get_matching_blocks())
    return matched / len(text_a)


def overlap_reward(
    completion: Completion,
    ground_truth: Any,
    *,
    reference_chunks: Sequence[Mapping[str, Any]] = (),
    minimum_overlap: float = 0.25,
) -> float:
    """Reward textual overlap with reference chunks or a ground truth string."""

    if not 0 <= minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be within [0, 1]")
    reference = (
        " ".join(str(chunk.get("content", "")) for chunk in reference_chunks)
        if reference_chunks
        else str(ground_truth or "")
    )
    if not reference:
        return 0.0
    score = percent_of_text_a_in_text_b(reference, extract_completion_text(completion))
    return score if score >= minimum_overlap else 0.0


def count_search_calls(completion: Completion) -> int:
    """Count serialized ``<tool_call>`` markers in assistant messages."""

    if isinstance(completion, str):
        return completion.count("<tool_call>")
    count = 0
    for message in completion:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            count += content.count("<tool_call>")
    return count


def search_within_budget(calls: int, max_calls: int) -> bool:
    """Return whether a non-negative call count is within the budget."""

    if calls < 0 or max_calls < 0:
        raise ValueError("calls and max_calls must be non-negative")
    return calls <= max_calls


def citation_score(
    completion: Completion,
    reference_chunks: Sequence[Mapping[str, Any]],
    *,
    source_field: str | Sequence[str] = "source_id",
    canonicalize: Callable[[str], str] | None = None,
) -> dict[str, float]:
    """Score ``[Source: id]`` citation precision and recall."""

    fields = (source_field,) if isinstance(source_field, str) else tuple(source_field)
    if not fields:
        raise ValueError("source_field must contain at least one field")
    normalize = canonicalize or str.strip
    cited = {normalize(value) for value in _SOURCE_CITATION.findall(
        extract_completion_text(completion)
    )}
    cited.discard("")

    references: set[str] = set()
    for chunk in reference_chunks:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        for field in fields:
            value = metadata.get(field)
            if value is not None and str(value).strip():
                normalized = normalize(str(value))
                if normalized:
                    references.add(normalized)
                break
    if not cited or not references:
        return {"precision": 0.0, "recall": 0.0}
    matches = cited & references
    return {
        "precision": len(matches) / len(cited),
        "recall": len(matches) / len(references),
    }


def tool_call_efficiency(
    completion: Completion,
    *,
    correctness: float = 1.0,
    reference_chunk_count: int = 0,
    max_calls: int = 10,
    decay_rate: float = 0.2,
    ranges: Sequence[tuple[int, int | None, float]] | None = None,
) -> float:
    """Score tool usage with explicit ranges or correctness-scaled decay."""

    if reference_chunk_count < 0 or max_calls < 0 or decay_rate < 0:
        raise ValueError("counts and decay_rate must be non-negative")
    calls = count_search_calls(completion)
    if ranges is not None:
        for minimum, maximum, score in ranges:
            if minimum < 0 or (maximum is not None and maximum < minimum):
                raise ValueError("tool-call ranges must be ordered and non-negative")
            if minimum <= calls and (maximum is None or calls <= maximum):
                return clip01(score)
        return 0.0

    correctness = clip01(correctness)
    if correctness == 0 or calls > max_calls:
        return 0.0
    excess = max(0, calls - (reference_chunk_count + 2))
    return correctness * math.exp(-decay_rate * excess)
