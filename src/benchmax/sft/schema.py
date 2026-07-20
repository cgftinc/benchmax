"""The SFT row contract: ``{"messages": [...]}`` plus optional ``tools`` and
per-assistant-message ``weight``.

:func:`validate_row` is pure structural validation — role shape, tool-call
well-formedness, content-part well-formedness, at least one trained
assistant turn, and JSON-serializability. It has no notion of on-disk
provenance (that's :mod:`benchmax.sft.dataset`'s job) or severity tiers
(that's :mod:`benchmax.sft.validate`'s) — it just returns a flat list of
human-readable error messages, empty when the row is well-formed.
"""

from __future__ import annotations

import json
from typing import Any

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
VALID_CONTENT_PART_TYPES = frozenset({"text", "image_url"})
VALID_WEIGHTS = frozenset({0, 1})


def validate_row(row: Any) -> list[str]:
    """Structural errors in one canonical SFT row. Empty list means well-formed."""
    if not isinstance(row, dict):
        return [f"row must be a JSON object, got {type(row).__name__}"]

    messages = row.get("messages")
    if messages is None:
        return ["'messages' is required"]
    if not isinstance(messages, list) or not messages:
        return ["'messages' must be a non-empty list"]

    errors: list[str] = []
    trained_assistant_turns = 0
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{i}] must be an object")
            continue

        role = message.get("role")
        if role not in VALID_ROLES:
            errors.append(
                f"messages[{i}].role must be one of {sorted(VALID_ROLES)}, got {role!r}"
            )
        if role == "tool" and not isinstance(message.get("tool_call_id"), str):
            errors.append(f"messages[{i}] (tool) must have a string 'tool_call_id'")

        errors.extend(_validate_content(message.get("content"), i))

        if role == "assistant":
            weight = message.get("weight")
            if "weight" in message and weight not in VALID_WEIGHTS:
                errors.append(f"messages[{i}].weight must be 0 or 1, got {weight!r}")
            elif weight != 0:
                trained_assistant_turns += 1

            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                errors.extend(_validate_tool_calls(tool_calls, i))

    if trained_assistant_turns == 0:
        errors.append(
            "row has no trained assistant turn (all assistant messages have "
            "weight=0, or none are present)"
        )

    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        errors.append("'tools' must be a list")

    if not errors:
        try:
            json.dumps(row)
        except (TypeError, ValueError) as exc:
            errors.append(f"row is not JSON-serializable: {exc}")

    return errors


def _validate_content(content: Any, index: int) -> list[str]:
    if content is None or isinstance(content, str):
        return []
    if not isinstance(content, list):
        return [f"messages[{index}].content must be a string or a list of content parts"]

    errors: list[str] = []
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            errors.append(f"messages[{index}].content[{j}] must be an object")
            continue
        part_type = part.get("type")
        if part_type not in VALID_CONTENT_PART_TYPES:
            errors.append(
                f"messages[{index}].content[{j}].type must be one of "
                f"{sorted(VALID_CONTENT_PART_TYPES)}, got {part_type!r}"
            )
            continue
        if part_type == "text" and not isinstance(part.get("text"), str):
            errors.append(f"messages[{index}].content[{j}] (text) must have a string 'text'")
        elif part_type == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not (url.startswith("data:") or url.startswith("https:")):
                errors.append(
                    f"messages[{index}].content[{j}] (image_url) must have image_url.url "
                    "starting with 'data:' or 'https:'"
                )
    return errors


def _validate_tool_calls(tool_calls: Any, index: int) -> list[str]:
    if not isinstance(tool_calls, list):
        return [f"messages[{index}].tool_calls must be a list"]

    errors: list[str] = []
    for j, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            errors.append(f"messages[{index}].tool_calls[{j}] must be an object")
            continue
        if tc.get("type") != "function":
            errors.append(f"messages[{index}].tool_calls[{j}].type must be 'function'")
        function = tc.get("function")
        if not isinstance(function, dict):
            errors.append(f"messages[{index}].tool_calls[{j}].function must be an object")
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            errors.append(
                f"messages[{index}].tool_calls[{j}].function.name must be a non-empty string"
            )
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            errors.append(
                f"messages[{index}].tool_calls[{j}].function.arguments must be a "
                "JSON-encoded string"
            )
        else:
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                errors.append(
                    f"messages[{index}].tool_calls[{j}].function.arguments must be valid JSON"
                )
    return errors
