"""Braintrust-specific message extraction from trace trees.

Converts raw Braintrust span trees into flat ``TraceMessage`` lists using a
3-strategy heuristic approach derived from the prototype at
``expt-platform-bt-proto/app/braintrust/page.tsx``.
"""

from __future__ import annotations

import json
from typing import Any

from benchmax.traces.adapter import ToolCall, TraceMessage, normalize_message, _is_structured_content


def extract_messages(trace: dict[str, Any]) -> list[TraceMessage]:
    """Convert a Braintrust trace tree into a flat message list.

    Strategies tried in order:

    1. Root ``input`` has full conversation (contains assistant messages).
    2. Reconstruct from child spans — find the LLM child with the most
       ``input.messages``, then append outputs from subsequent children.
    3. Fallback — root input + all child outputs in timestamp order.

    Returns an empty list if no messages can be extracted.
    """
    root_input = trace.get("input", {})
    children = _sorted_children(trace.get("children", []))
    root_messages = _parse_messages_field(root_input)

    # Strategy 1: root input already has the full conversation
    if _has_assistant_messages(root_messages):
        return root_messages

    # Strategy 2: reconstruct from child spans
    result = _reconstruct_from_children(root_messages, children)
    if result:
        return result

    # Strategy 3: fallback — root input + all child outputs in order
    return _fallback_extraction(root_messages, children)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def _sorted_children(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort children by ``created`` timestamp (they arrive in arbitrary order)."""
    return sorted(children, key=lambda c: c.get("created", ""))


def _has_assistant_messages(messages: list[TraceMessage]) -> bool:
    return any(m.role == "assistant" for m in messages)


def _reconstruct_from_children(
    root_messages: list[TraceMessage],
    children: list[dict[str, Any]],
) -> list[TraceMessage]:
    """Strategy 2: find the LLM child with the most input messages, then
    reconstruct the full conversation.

    Returns an empty list if no LLM children are found.
    """
    llm_children = [c for c in children if _get_span_type(c) == "llm"]

    if not llm_children:
        return []

    # Pick the LLM child with the most input.messages
    best_child = max(
        llm_children,
        key=lambda c: len(_parse_messages_field(c.get("input", {}))),
    )
    best_idx = children.index(best_child) if best_child in children else -1

    # Start with the best child's input messages (the fullest context)
    messages = list(_parse_messages_field(best_child.get("input", {})))

    # Track content already in the best child's input so we can skip
    # subsequent children whose output was already consumed.
    seen_content = {m.content for m in messages if m.content}

    # Append the best child's output
    output_msgs = _parse_output(best_child.get("output", {}))
    messages.extend(output_msgs)

    # Append subsequent children's outputs (skip duplicates already in input)
    if best_idx >= 0:
        for child in children[best_idx + 1 :]:
            span_type = _get_span_type(child)
            if span_type == "tool":
                tool_output = _extract_tool_response(child)
                if tool_output and tool_output.content not in seen_content:
                    messages.append(tool_output)
                    seen_content.add(tool_output.content)
            else:
                output_msgs = _parse_output(child.get("output", {}))
                for msg in output_msgs:
                    if msg.content not in seen_content:
                        messages.append(msg)
                        seen_content.add(msg.content)

    return messages if _has_assistant_messages(messages) else []


def _fallback_extraction(
    root_messages: list[TraceMessage],
    children: list[dict[str, Any]],
) -> list[TraceMessage]:
    """Strategy 3: root input + all child outputs in timestamp order."""
    messages = list(root_messages)
    for child in children:
        output_msgs = _parse_output(child.get("output", {}))
        messages.extend(output_msgs)
    return messages


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _get_span_type(span: dict[str, Any]) -> str:
    """Extract ``span_attributes.type`` from a span dict."""
    attrs = span.get("span_attributes", {})
    if isinstance(attrs, dict):
        return str(attrs.get("type", ""))
    return ""


def _parse_messages_field(input_data: Any) -> list[TraceMessage]:
    """Parse an ``input`` or ``output`` dict's ``messages`` field.

    Handles both ``{"messages": [...]}`` and bare list formats.
    """
    if not input_data:
        return []

    if isinstance(input_data, list):
        return [_parse_msg(m) for m in input_data if isinstance(m, dict)]

    if isinstance(input_data, dict):
        messages = input_data.get("messages", [])
        if isinstance(messages, list):
            return [_parse_msg(m) for m in messages if isinstance(m, dict)]

    return []


def _parse_output(output_data: Any) -> list[TraceMessage]:
    """Parse an ``output`` field into messages.

    Braintrust outputs can be:
    - ``{"messages": [...]}`` — list of messages
    - A single message dict with ``role`` and ``content``
    - A plain string (treated as assistant content)
    """
    if not output_data:
        return []

    if isinstance(output_data, str):
        return [TraceMessage(role="assistant", content=output_data)]

    if isinstance(output_data, dict):
        # Check if it's a single message
        if "role" in output_data:
            return [_parse_msg(output_data)]
        # Check for messages list
        messages = output_data.get("messages", [])
        if isinstance(messages, list) and messages:
            return [_parse_msg(m) for m in messages if isinstance(m, dict)]
        # Check for content field
        content = output_data.get("content", "")
        if content:
            return [TraceMessage(role="assistant", content=str(content))]

    if isinstance(output_data, list):
        return [_parse_msg(m) for m in output_data if isinstance(m, dict)]

    return []


def _parse_msg(msg: dict[str, Any]) -> TraceMessage:
    """Parse a single message dict, handling both OpenAI and structured-content formats.

    Delegates to ``normalize_message`` for structured-content messages
    (openclaw / Anthropic style where ``content`` is a list of typed blocks).
    Otherwise normalises tool_calls format variations directly.
    """
    if _is_structured_content(msg):
        return normalize_message(msg)

    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    if content is None:
        content = ""
    content = str(content)

    tool_calls = _normalize_tool_calls(msg)
    tool_call_id = msg.get("tool_call_id")
    name = msg.get("name")

    return TraceMessage(
        role=role,
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        tool_call_id=tool_call_id,
        name=name,
    )


def _normalize_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from a message, handling format variations."""
    raw_calls = msg.get("tool_calls")
    if not raw_calls:
        # Try older "function" format
        func = msg.get("function")
        if func and isinstance(func, dict):
            return [
                ToolCall(
                    name=func.get("name", ""),
                    arguments=_ensure_json_string(func.get("arguments", "{}")),
                    id=msg.get("id"),
                )
            ]
        return []

    if not isinstance(raw_calls, list):
        return []

    result: list[ToolCall] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        # Handle nested function format: {"function": {"name": ..., "arguments": ...}}
        func = tc.get("function", {})
        if isinstance(func, dict) and "name" in func:
            result.append(
                ToolCall(
                    name=func["name"],
                    arguments=_ensure_json_string(func.get("arguments", "{}")),
                    id=tc.get("id"),
                )
            )
        elif "name" in tc:
            # Direct format: {"name": ..., "arguments": ...}
            result.append(
                ToolCall(
                    name=tc["name"],
                    arguments=_ensure_json_string(tc.get("arguments", "{}")),
                    id=tc.get("id"),
                )
            )

    return result


def _ensure_json_string(value: Any) -> str:
    """Ensure *value* is a JSON string.  If it's a dict, serialise it."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _extract_tool_response(child: dict[str, Any]) -> TraceMessage | None:
    """Extract a tool response message from a tool-type child span."""
    output = child.get("output")
    if output is None:
        return None

    content = ""
    if isinstance(output, str):
        content = output
    elif isinstance(output, dict):
        content = str(output.get("content", output.get("result", str(output))))
    else:
        content = str(output)

    name = child.get("name", "")
    tool_call_id = child.get("span_id", child.get("id", ""))

    return TraceMessage(
        role="tool",
        content=str(content),
        tool_call_id=str(tool_call_id) if tool_call_id else None,
        name=name if name else None,
    )


def extract_scores(trace: dict[str, Any]) -> dict[str, float]:
    """Extract scores from a Braintrust trace root span.

    Scores can be:
    - ``trace["scores"]`` — dict of ``{name: score}``
    - ``trace["scores"]`` — list of ``{name, score}`` dicts
    """
    raw_scores = trace.get("scores")
    if not raw_scores:
        return {}

    if isinstance(raw_scores, dict):
        return {k: float(v) for k, v in raw_scores.items() if isinstance(v, (int, float))}

    if isinstance(raw_scores, list):
        result: dict[str, float] = {}
        for entry in raw_scores:
            if isinstance(entry, dict) and "name" in entry and "score" in entry:
                try:
                    result[entry["name"]] = float(entry["score"])
                except (ValueError, TypeError):
                    pass
        return result

    return {}
