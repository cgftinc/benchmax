"""Small, dependency-free output helpers shared by CLI command groups."""

from __future__ import annotations

import json as _json
import os
import re
import shutil
import sys
from typing import Any


# Castform brand palette (web-app diagram tokens: blue #3b76f6, orange #f97316),
# softened for the terminal and tuned per background. We can't query the terminal
# background synchronously, but many emulators export COLORFGBG ("fg;bg"); when bg
# reads light we pick deeper, higher-contrast tones, otherwise softer pastels that
# don't glare on black. Falls back to the dark (pastel) set. Rendered as truecolor.
def _is_light_terminal() -> bool:
    parts = os.environ.get("COLORFGBG", "").split(";")
    if parts and parts[-1].strip().isdigit():
        return int(parts[-1].strip()) >= 7  # 7/15 = light background
    return False


if _is_light_terminal():
    BLUE = (37, 99, 235)  # #2563eb — deeper, reads on white
    ORANGE = (194, 87, 28)  # #c2571c — terracotta
else:
    BLUE = (125, 166, 232)  # #7da6e8 — soft sky
    ORANGE = (230, 166, 99)  # #e6a663 — soft amber

_GREY = (140, 140, 140)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def color_enabled() -> bool:
    """True when we should emit ANSI — a real TTY and ``NO_COLOR`` unset."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def paint(
    text: str,
    rgb: tuple[int, int, int] | None = None,
    *,
    bold: bool = False,
    italic: bool = False,
    dim: bool = False,
) -> str:
    """Wrap ``text`` in ANSI styling, or return it unchanged when color is off."""
    if not color_enabled():
        return text
    codes: list[str] = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if italic:
        codes.append("3")
    if rgb is not None:
        codes.append(f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}")
    if not codes:
        return text
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"


def _visible_len(s: str) -> int:
    """Length of ``s`` ignoring ANSI escapes — for padding pre-colored text."""
    return len(_ANSI_RE.sub("", s))


def term_width(default: int = 80) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def rule_label(text: str, color: tuple[int, int, int], width: int) -> str:
    """The standard section divider: a centered title flanked by rules, fully
    colored — ``──────── title ────────`` — spanning ``width`` columns."""
    total = max(width - len(text) - 2, 0)
    left = total // 2
    line = "─" * left + f" {text} " + "─" * (total - left)
    return paint(line, color, bold=True)


def boxed(
    lines: list[str],
    *,
    color: tuple[int, int, int],
    width: int,
    title: str = "",
) -> list[str]:
    """Render a rounded box ``width`` columns wide (content area) around ``lines``.

    ``lines`` may already contain ANSI codes or be nested boxes — widths are
    measured ignoring escapes, so padding stays aligned. ``title`` is centered in
    the top border. Returns the box as a list of rendered rows.
    """
    inner = width + 2  # one space of padding on each side
    seg = f" {title} " if title else ""
    fill = max(inner - _visible_len(seg), 0)
    left = fill // 2
    top = "╭" + "─" * left + seg + "─" * (fill - left) + "╮"
    bottom = "╰" + "─" * inner + "╯"
    bar = paint("│", color)
    out = [paint(top, color, bold=True)]
    for ln in lines:
        pad = max(width - _visible_len(ln), 0)
        out.append(f"{bar} {ln}{' ' * pad} {bar}")
    out.append(paint(bottom, color))
    return out


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


def truncate(text: object, n: int) -> str:
    """Collapse whitespace and clip to ``n`` chars (…-suffixed) for one-line display.

    ``None`` → ``""``, but a falsy-yet-real value is preserved (a gold answer of
    ``0`` must render as ``0``, not blank)."""
    s = " ".join(("" if text is None else str(text)).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def final_answer(messages: list | None) -> str | None:
    """The last assistant-turn content in a transcript — the model's committed answer."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return None
