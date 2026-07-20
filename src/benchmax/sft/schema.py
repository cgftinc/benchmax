"""The SFT row contract: ``{"messages": [...]}`` plus optional ``tools`` and
per-assistant-message ``weight``.

:func:`validate_row` is pure structural validation — role shape, tool-call
well-formedness, content-part well-formedness, at least one trained
assistant turn, and JSON-serializability. It has no notion of on-disk
provenance (that's :mod:`benchmax.sft.dataset`'s job) or severity tiers
(that's :mod:`benchmax.sft.validate`'s) — it just returns a flat list of
human-readable error messages, empty when the row is well-formed.

Every check that compares a user-supplied value against a fixed set
type-checks first: raw JSONL can put anything JSON-shaped (a list, a
dict, a bool) where a string or int is expected, and an unhashable value
in a set-membership test raises ``TypeError`` instead of producing a
clean issue.
"""

from __future__ import annotations

import json
from typing import Any

ALLOWED_TOP_LEVEL_KEYS = frozenset({"messages", "tools"})
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
VALID_CONTENT_PART_TYPES = frozenset({"text", "image_url"})
VALID_WEIGHTS = frozenset({0, 1})


def validate_row(row: Any) -> list[str]:
    """Structural errors in one canonical SFT row. Empty list means well-formed.

    Every check below runs unconditionally — including JSON-serializability
    — rather than short-circuiting on the first error, so a row with
    several independent problems reports all of them.
    """
    if not isinstance(row, dict):
        return [f"row must be a JSON object, got {type(row).__name__}"]

    errors: list[str] = []

    unexpected = set(row) - ALLOWED_TOP_LEVEL_KEYS
    if unexpected:
        errors.append(
            f"unexpected field(s) {sorted(unexpected)}; only "
            f"{sorted(ALLOWED_TOP_LEVEL_KEYS)} are accepted top-level keys"
        )

    messages = row.get("messages")
    if messages is None:
        errors.append("'messages' is required")
    elif not isinstance(messages, list) or not messages:
        errors.append("'messages' must be a non-empty list")
    else:
        errors.extend(_validate_messages(messages))

    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        errors.append("'tools' must be a list")

    try:
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"row is not JSON-serializable: {exc}")

    return errors


def _validate_messages(messages: list[Any]) -> list[str]:
    errors: list[str] = []
    trained_assistant_turns = 0

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{i}] must be an object")
            continue

        role = message.get("role")
        if not isinstance(role, str) or role not in VALID_ROLES:
            errors.append(
                f"messages[{i}].role must be one of {sorted(VALID_ROLES)}, got {role!r}"
            )

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                errors.append(f"messages[{i}] (tool) must have a non-empty string 'tool_call_id'")

        errors.extend(_validate_content(message.get("content"), i))

        has_weight = "weight" in message
        weight = message.get("weight")
        if has_weight and role != "assistant":
            errors.append(f"messages[{i}].weight is only valid on assistant messages")

        if role == "assistant":
            weight_valid = True
            if has_weight:
                # bool is a subclass of int (isinstance(True, int) is True) and
                # 1.0 == 1, so an exact `type(...) is int` check is required to
                # reject True/1.0 rather than silently accepting them as 0/1.
                if type(weight) is not int or weight not in VALID_WEIGHTS:
                    errors.append(f"messages[{i}].weight must be 0 or 1, got {weight!r}")
                    weight_valid = False

            tool_calls = message.get("tool_calls")
            has_valid_tool_calls = False
            if tool_calls is not None:
                tool_call_errors = _validate_tool_calls(tool_calls, i)
                errors.extend(tool_call_errors)
                has_valid_tool_calls = (
                    isinstance(tool_calls, list) and len(tool_calls) > 0 and not tool_call_errors
                )

            is_masked = has_weight and weight_valid and weight == 0
            has_content = _has_meaningful_content(message.get("content"))
            if not is_masked and (has_content or has_valid_tool_calls):
                trained_assistant_turns += 1

    if trained_assistant_turns == 0:
        errors.append(
            "row has no trained assistant turn (need an assistant message with content "
            "or tool calls, and weight != 0)"
        )

    return errors


def _has_meaningful_content(content: Any) -> bool:
    if isinstance(content, str):
        return content != ""
    if isinstance(content, list):
        return len(content) > 0
    return False


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
        if not isinstance(part_type, str) or part_type not in VALID_CONTENT_PART_TYPES:
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

        tool_call_id = tc.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            errors.append(f"messages[{index}].tool_calls[{j}].id must be a non-empty string")

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
