"""castform guide — the getting-started walkthrough, rendered for the terminal.

`castform setup` prints a compact get-started block and points here for the full
walkthrough. This renders the packaged ``scaffold/STARTER.md`` with a tiny,
purpose-built markdown formatter (not a general one) under the same convention
``setup`` uses: section headers become fully-colored ``──── title ────`` rules,
*prompts* become colored titled boxes, *commands* render as unboxed indented
text, blockquotes get a colored bar, and tables align into columns. Command/code
text is left uncolored so it reads on any terminal theme — only dividers, prompt
boxes, and accents carry brand color.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from importlib import resources

from benchmax.cli._output import (
    BLUE,
    ORANGE,
    boxed,
    paint,
    rule_label,
    term_width,
)

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_EM_RE = re.compile(r"\*([^*]+)\*")

# A fenced block is a *command* example (rendered unboxed) when its first line
# looks like a shell command; otherwise it's a *prompt* (a colored, titled box).
_CMD_PREFIXES = ("castform", "uv ", "uv\t", "pip ", "cd ", "python", "source ", "$")


def _inline(text: str) -> str:
    """Theme-safe inline markdown: ``**bold**``/``*em*`` via ANSI (no color),
    inline ``code`` left in backticks so it reads without relying on color."""
    text = _BOLD_RE.sub(lambda m: paint(m.group(1), bold=True), text)
    return _EM_RE.sub(lambda m: paint(m.group(1), italic=True), text)


def _collapse_blanks(rows: list[str]) -> list[str]:
    """Drop leading blanks and collapse runs of blank lines to a single one."""
    out: list[str] = []
    for r in rows:
        if not r.strip():
            if not out or not out[-1].strip():
                continue
            out.append("")
        else:
            out.append(r)
    while out and not out[-1].strip():
        out.pop()
    return out


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(c) <= set("-: ") and c for c in cells)


def _render_table(rows: list[str]) -> list[str]:
    """Render a GitHub-style pipe table as aligned columns; header row bold."""
    parsed = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    parsed = [c for c in parsed if not _is_separator_row(c)]
    if not parsed:
        return []
    ncol = max(len(c) for c in parsed)
    widths = [0] * ncol
    for row in parsed:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))
    out: list[str] = []
    for k, row in enumerate(parsed):
        line = "   ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip()
        out.append(paint(line, bold=True) if k == 0 else _inline(line))
    return out


def _render_blockquote(qlines: list[str], width: int) -> list[str]:
    """A blockquote with a colored bar; prose re-flows, ``-`` items stay separate."""
    bar = paint("▌", BLUE)
    out: list[str] = []
    buf: list[str] = []

    def emit(text: str) -> None:
        for w in textwrap.wrap(text, width - 2) or [""]:
            out.append(f"{bar} {_inline(w)}")

    def flush() -> None:
        if buf:
            emit(" ".join(buf))
            buf.clear()

    for q in qlines:
        if not q:
            flush()
            out.append(bar)
        elif q.startswith(("- ", "* ")):
            flush()
            emit(q)
        else:
            buf.append(q)
    flush()
    return out


def _gather_fence(lines: list[str], i: int) -> tuple[list[str], int]:
    """``lines[i]`` is an opening ```` ``` ````; return (code, index-after-close)."""
    j = i + 1
    code: list[str] = []
    while j < len(lines) and not lines[j].startswith("```"):
        code.append(lines[j])
        j += 1
    return code, j + 1


def _is_command_block(code: list[str]) -> bool:
    first = next((c for c in code if c.strip()), "")
    return first.lstrip().startswith(_CMD_PREFIXES)


def _prompt_box(code: list[str], title: str, width: int) -> list[str]:
    """A prompt to paste → a colored, titled box (the shared prompt convention)."""
    text = " ".join(c.strip() for c in code if c.strip())
    inner = width - 4
    body = [paint(w, italic=True) for w in textwrap.wrap(text, inner) or [""]]
    return boxed(body, color=ORANGE, width=inner, title=title)


def render_doc(md: str, width: int) -> list[str]:
    """Render our getting-started markdown under the shared convention.

    Section dividers → fully-colored ``──── title ────`` rules; *prompts* (natural
    -language fenced blocks) → colored, titled boxes; *commands* (shell fenced
    blocks) → unboxed indented text; blockquotes → a colored bar. Consecutive
    prose lines reflow into one wrapped paragraph (source hard-wraps are dropped).
    A bold label immediately preceding a prompt becomes that box's title.
    """
    out: list[str] = []
    para: list[str] = []
    pending_label: str | None = None  # a bold label awaiting its prompt box

    def flush() -> None:
        if para:
            for w in textwrap.wrap(" ".join(para), width):
                out.append(_inline(w))
            para.clear()

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):  # fenced block → prompt box or command text
            flush()
            code, i = _gather_fence(lines, i)
            out.append("")
            if _is_command_block(code):
                out += [f"    {c}" for c in code]  # unboxed, uncolored
            else:
                out += _prompt_box(code, pending_label or "prompt", width)
            pending_label = None
            continue

        if line.startswith("## "):  # section divider
            flush()
            pending_label = None
            out += ["", rule_label(line[3:].strip(), BLUE, width)]
            i += 1
            continue
        if line.startswith("# "):  # title divider
            flush()
            pending_label = None
            out += ["", rule_label(line[2:].strip(), ORANGE, width), ""]
            i += 1
            continue

        if line.lstrip().startswith("|"):  # table block
            flush()
            tbl: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out += _render_table(tbl)
            continue

        if line.startswith(">"):  # blockquote → colored bar
            flush()
            qlines: list[str] = []
            while i < len(lines) and lines[i].startswith(">"):
                qlines.append(lines[i][1:].strip())
                i += 1
            out += _render_blockquote(qlines, width)
            continue

        if not line.strip():  # blank → paragraph break
            flush()
            out.append("")
            i += 1
            continue

        # A bold label whose next block is a *prompt* becomes that box's title —
        # so suppress it as prose. (Labels before command blocks stay as prose.)
        stripped = line.strip()
        if stripped.startswith("**"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("```"):
                peek, _ = _gather_fence(lines, j)
                if not _is_command_block(peek):
                    m = _BOLD_RE.search(stripped)
                    pending_label = (m.group(1) if m else stripped).rstrip(":").strip()
                    i += 1
                    continue

        para.append(stripped)  # prose — accumulate, wrap on flush
        i += 1

    flush()
    return _collapse_blanks(out)


def _cmd_help(args: argparse.Namespace) -> int:
    width = min(term_width() - 2, 76)
    md = (resources.files("benchmax.cli.scaffold") / "STARTER.md").read_text(
        encoding="utf-8"
    )
    print("\n".join(render_doc(md, width)))
    print()  # trailing blank line (shared convention)
    return 0
