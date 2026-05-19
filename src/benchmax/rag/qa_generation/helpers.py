"""Helper functions for QA generation and corpus analysis."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..chunkers.models import Chunk

# ============================================================================
# Template Rendering and Parsing
# ============================================================================

_SIMPLE_TEMPLATE_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_CONDITIONAL_TOKEN_RE = re.compile(r"\[\[if\s+([a-zA-Z_][a-zA-Z0-9_]*)\]\]|\[\[endif\]\]")
_ESCAPED_OPEN_BRACE = "__CGFT_ESCAPED_OPEN_BRACE__"
_ESCAPED_CLOSE_BRACE = "__CGFT_ESCAPED_CLOSE_BRACE__"


def _template_value_present(value: Any) -> bool:
    """Return whether a template variable value is considered present."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _render_conditionals(template: str, variables: dict[str, Any]) -> str:
    """Render [[if field]]...[[endif]] blocks before placeholder substitution."""
    parts_stack: list[list[str]] = [[]]
    active_stack: list[bool] = [True]
    pos = 0

    for match in _CONDITIONAL_TOKEN_RE.finditer(template):
        parts_stack[-1].append(template[pos : match.start()])

        if match.group(1):
            field = match.group(1)
            parent_active = active_stack[-1]
            value_present = _template_value_present(variables.get(field))
            block_active = parent_active and value_present
            parts_stack.append([])
            active_stack.append(block_active)
        else:
            if len(parts_stack) == 1:
                raise ValueError("Unbalanced conditional template markers: unexpected [[endif]].")
            block_content = "".join(parts_stack.pop())
            block_active = active_stack.pop()
            if block_active:
                parts_stack[-1].append(block_content)

        pos = match.end()

    parts_stack[-1].append(template[pos:])
    if len(parts_stack) != 1:
        raise ValueError("Unbalanced conditional template markers: missing [[endif]].")
    return "".join(parts_stack[0])


def render_template(template: str, variables: dict[str, Any]) -> str:
    """
    Render a template string with variables.

    Args:
        template: Template string with {variable} placeholders.
            Supports optional conditional blocks:
            [[if variable_name]] ... [[endif]]
        variables: Dictionary of variable values

    Returns:
        Rendered template string
    """
    protected = template.replace("{{", _ESCAPED_OPEN_BRACE).replace("}}", _ESCAPED_CLOSE_BRACE)
    conditioned = _render_conditionals(protected, variables)
    required_fields = set(_SIMPLE_TEMPLATE_FIELD_RE.findall(conditioned))
    missing = sorted(field for field in required_fields if field not in variables)
    if missing:
        raise KeyError(f"Missing template variable(s): {', '.join(missing)}")

    rendered = _SIMPLE_TEMPLATE_FIELD_RE.sub(
        lambda match: str(variables[match.group(1)]),
        conditioned,
    )
    return rendered.replace(_ESCAPED_OPEN_BRACE, "{").replace(_ESCAPED_CLOSE_BRACE, "}")


# ============================================================================
# Chunk Filtering and Sampling
# ============================================================================


def filter_chunks_by_length(
    chunk_list: list[Chunk],
    min_chars: int = 0,
    max_chars: int | None = None,
) -> list[Chunk]:
    """Filter chunks by character length.

    This is a simple filter that works for both single-hop and multi-hop
    chunk selection. For more complex sampling (with max_chunks and
    sampling strategies), use select_eligible_chunks().

    Args:
        chunk_list: List of chunks to filter
        min_chars: Minimum character length (inclusive, default 0)
        max_chars: Maximum character length (inclusive, None for no limit)

    Returns:
        List of chunks meeting the length criteria

    Example:
        >>> # Filter for chunks with meaningful content (>500 chars)
        >>> eligible = filter_chunks_by_length(collection, min_chars=500)
        >>>
        >>> # Filter for medium-sized chunks
        >>> medium = filter_chunks_by_length(collection, min_chars=500, max_chars=2000)
    """
    filtered = []
    for chunk in chunk_list:
        length = len(chunk)
        if length >= min_chars:
            if max_chars is None or length <= max_chars:
                filtered.append(chunk)

    return filtered
