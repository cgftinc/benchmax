"""Small, dependency-free output helpers shared by CLI command groups."""

from __future__ import annotations

import json as _json
from typing import Any


def print_json(obj: Any) -> None:
    """Emit ``obj`` as pretty JSON (``default=str`` so stray types don't crash)."""
    print(_json.dumps(obj, indent=2, default=str))


def render_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Print a left-aligned fixed-width table. No-op styling — pipe-friendly."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def fmt_value(value: Any) -> str:
    """Compact numeric formatting for scalar values; pass through non-numbers."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
