"""Legacy SFT row shapes -> the canonical ``{"messages": [...], "tools"?: [...]}`` shape.

Deliberately self-contained rather than sharing a normalizer with a trace
importer: those are lossy for multimodal content, and this one must preserve
every part verbatim. Invoked only by
:func:`benchmax.sft.dataset.load_sft_dataset` — callers never see a raw row.
Never raises: a row that matches no recognized legacy shape passes through
unchanged for :func:`benchmax.sft.schema.validate_row` to reject with a clear
message.

Copy-on-write: only the specific legacy keys actually consumed into
``messages`` (``prompt_messages``/``prompt``/``completion_messages``/
``completion``) are removed. Every other top-level field — recognized or
not — survives to the canonical row, and every field on a tool-call/
function dict beyond the ones being reshaped survives too, so an
unsupported field becomes an explicit schema issue rather than silent
data loss.
"""

from __future__ import annotations

import json
from typing import Any


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one parsed JSON row into the canonical SFT row shape.

    Recognizes an already-canonical ``messages`` row, the
    ``prompt_messages``/``completion_messages`` split, a bare ``prompt``
    and/or ``completion`` string, and flat (un-nested) tool-call entries —
    combinable in any mix. ``tools``, per-message ``weight``/content parts,
    and any other field are preserved verbatim; only legacy shapes are
    rewritten.
    """
    if "messages" in row:
        canonical = dict(row)
        messages = canonical["messages"]
        if isinstance(messages, list):
            canonical["messages"] = [_normalize_message(m) for m in messages]
        return canonical

    consumed_keys: set[str] = set()
    prompt_part = _extract_prompt_part(row, consumed_keys)
    completion_part = _extract_completion_part(row, consumed_keys)
    if prompt_part is None and completion_part is None:
        return dict(row)

    canonical = {k: v for k, v in row.items() if k not in consumed_keys}
    messages = (prompt_part or []) + (completion_part or [])
    canonical["messages"] = [_normalize_message(m) for m in messages]
    return canonical


def _extract_prompt_part(row: dict[str, Any], consumed_keys: set[str]) -> list[Any] | None:
    if "prompt_messages" in row:
        consumed_keys.add("prompt_messages")
        value = row["prompt_messages"]
        return value if isinstance(value, list) else [value]
    if "prompt" in row and isinstance(row["prompt"], str):
        consumed_keys.add("prompt")
        return [{"role": "user", "content": row["prompt"]}]
    return None


def _extract_completion_part(row: dict[str, Any], consumed_keys: set[str]) -> list[Any] | None:
    if "completion_messages" in row:
        consumed_keys.add("completion_messages")
        value = row["completion_messages"]
        return value if isinstance(value, list) else [value]
    if "completion" in row and isinstance(row["completion"], str):
        consumed_keys.add("completion")
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
        # Already nested — copy every field on both dicts, only touching
        # `arguments` so id/type/extra keys on either level survive intact.
        normalized = dict(tool_call)
        normalized_function = dict(function)
        normalized_function["arguments"] = _stringify_arguments(function.get("arguments", "{}"))
        normalized["function"] = normalized_function
        return normalized

    if "name" in tool_call:
        # Flat shape: name/arguments live directly on the tool-call dict —
        # lift them into `function`, keep every other field as-is.
        normalized = {k: v for k, v in tool_call.items() if k not in {"name", "arguments"}}
        normalized.setdefault("type", "function")
        normalized["function"] = {
            "name": tool_call["name"],
            "arguments": _stringify_arguments(tool_call.get("arguments", "{}")),
        }
        return normalized

    return tool_call


def _stringify_arguments(arguments: Any) -> Any:
    """OpenAI tool-call ``arguments`` is a JSON-encoded string; coerce a dict, pass a
    string through."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except TypeError:
        return arguments
