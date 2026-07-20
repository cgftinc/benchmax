"""Legacy SFT row shapes -> the canonical ``{"messages": [...], "tools"?: [...]}`` shape.

Self-contained (no shared normalizer with ``traces/adapter.py``, which is
lossy for multimodal content and is rewritten by harbor-proper's merge).
Invoked only by :func:`benchmax.sft.dataset.load_sft_dataset` — callers
never see a raw row. Never raises: a row that matches no recognized legacy
shape passes through unchanged for :func:`benchmax.sft.schema.validate_row`
to reject with a clear message.
"""

from __future__ import annotations

import json
from typing import Any


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one parsed JSON row into the canonical SFT row shape.

    Recognizes an already-canonical ``messages`` row, the
    ``prompt_messages``/``completion_messages`` split, a bare ``prompt``
    and/or ``completion`` string, and flat (un-nested) tool-call entries —
    combinable in any mix. ``tools`` and per-message ``weight``/content
    parts are preserved verbatim; only legacy shapes are rewritten.
    """
    canonical: dict[str, Any] = {}
    messages = _extract_messages(row)
    if messages is not None:
        canonical["messages"] = (
            [_normalize_message(m) for m in messages] if isinstance(messages, list) else messages
        )
    if "tools" in row:
        canonical["tools"] = row["tools"]
    return canonical


def _extract_messages(row: dict[str, Any]) -> Any:
    if "messages" in row:
        return row["messages"]

    prompt_part = _extract_prompt_part(row)
    completion_part = _extract_completion_part(row)
    if prompt_part is None and completion_part is None:
        return None
    return (prompt_part or []) + (completion_part or [])


def _extract_prompt_part(row: dict[str, Any]) -> list[Any] | None:
    if "prompt_messages" in row:
        value = row["prompt_messages"]
        return value if isinstance(value, list) else [value]
    if "prompt" in row and isinstance(row["prompt"], str):
        return [{"role": "user", "content": row["prompt"]}]
    return None


def _extract_completion_part(row: dict[str, Any]) -> list[Any] | None:
    if "completion_messages" in row:
        value = row["completion_messages"]
        return value if isinstance(value, list) else [value]
    if "completion" in row and isinstance(row["completion"], str):
        return [{"role": "assistant", "content": row["completion"]}]
    return None


def _normalize_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    normalized = dict(message)
    tool_calls = normalized.get("tool_calls")
    if isinstance(tool_calls, list):
        normalized["tool_calls"] = [_normalize_tool_call(tc) for tc in tool_calls]
    return normalized


def _normalize_tool_call(tool_call: Any) -> Any:
    if not isinstance(tool_call, dict):
        return tool_call
    function = tool_call.get("function")
    if isinstance(function, dict):
        return {
            "id": tool_call.get("id", ""),
            "type": tool_call.get("type", "function"),
            "function": {
                "name": function.get("name", ""),
                "arguments": _stringify_arguments(function.get("arguments", "{}")),
            },
        }
    if "name" in tool_call:
        return {
            "id": tool_call.get("id", ""),
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": _stringify_arguments(tool_call.get("arguments", "{}")),
            },
        }
    return tool_call


def _stringify_arguments(arguments: Any) -> Any:
    """OpenAI tool-call ``arguments`` is a JSON-encoded string; coerce a dict, pass a string through."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except TypeError:
        return arguments
