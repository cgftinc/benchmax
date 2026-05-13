"""Shared reward helpers for search environments.

No Chunk or Pydantic dependency — uses only plain strings and dicts.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def percent_of_text_a_in_text_b(text_a: str, text_b: str) -> float:
    """Calculate percentage of text_a that appears in text_b."""
    if not text_a:
        return 0.0
    matcher = SequenceMatcher(None, text_a, text_b)
    matched_chars = sum(size for _, _, size in matcher.get_matching_blocks())
    return matched_chars / len(text_a)


def overlap_reward(
    completion: str | list[dict[str, Any]],
    ground_truth: Any,
    **kwargs: Any,
) -> float:
    """Compute overlap reward based on chunk text overlap.

    Uses reference_chunks from kwargs when available, falls back to
    ground_truth string.  Penalizes >= 4 tool calls with 0 reward.
    Returns 0 if overlap < 25%.
    """
    reference_chunks = kwargs.get("reference_chunks", [])
    if reference_chunks:
        reference_string = " ".join(
            chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in reference_chunks
        )
    else:
        reference_string = str(ground_truth or "")

    completion_str = completion if isinstance(completion, str) else ""
    if isinstance(completion, list):
        completion_str = " ".join(
            c.get("content", "")
            for c in completion
            if isinstance(c, dict) and c.get("role", "") != "assistant"
        )
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            if msg.get("role", "") != "assistant":
                continue
            msg_content = msg.get("content", "")
            if msg_content.count("<tool_call>") >= 4:
                return 0.0

    if reference_string:
        score = percent_of_text_a_in_text_b(reference_string, completion_str)
        if score >= 0.25:
            return score
    return 0.0


def extract_completion_text(completion: str | list[dict[str, Any]]) -> str:
    """Extract text from a completion (string or message list)."""
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    parts: list[str] = []
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content)
    return "\n".join(parts)


def extract_answer_block(text: str) -> str:
    """Extract content from <answer> tags, or return full text."""
    match = _ANSWER_TAG_RE.search(text or "")
    return (match.group(1) if match else text).strip()


def clip01(value: Any) -> float:
    """Clip a value to [0, 1]."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def count_search_calls(completion: str | list[dict[str, Any]]) -> int:
    """Count search tool calls in a completion."""
    if isinstance(completion, str):
        return completion.count("<tool_call>")
    if not isinstance(completion, list):
        return 0
    count = 0
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                count += content.count("<tool_call>")
    return count


def search_within_budget(calls: int, max_calls: int) -> bool:
    """Check if the number of search calls is within budget."""
    return calls <= max_calls
