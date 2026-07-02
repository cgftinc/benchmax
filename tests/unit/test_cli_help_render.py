"""Unit tests for `castform guide`'s markdown-lite renderer (render_doc).

Color is auto-disabled when stdout isn't a TTY (pytest capture), so these assert
on the structural transforms — headers, code boxes, blockquotes, paragraph
reflow, tables — without ANSI noise.
"""

from __future__ import annotations

from benchmax.cli import help as help_cmd
from benchmax.cli.help import render_doc


def _joined(md: str, width: int = 76) -> str:
    return "\n".join(render_doc(md, width))


def test_header_becomes_centered_rule():
    rows = render_doc("# Get started", 40)
    line = next(r for r in rows if "Get started" in r)
    assert line.startswith("─") and line.rstrip().endswith("─")  # rule both sides


def test_command_block_is_unboxed_indented_text():
    out = _joined("```\ncastform validate\n```")
    assert "╭" not in out  # commands are never boxed
    assert "    castform validate" in out  # indented, plain


def test_prompt_block_becomes_titled_box():
    md = "**Your own task** (today):\n\n```\ni want to train a model on stuff.\n```"
    out = _joined(md, width=60)
    assert "╭" in out and "╰" in out  # prompts ARE boxed
    assert "Your own task" in out  # bold label becomes the box title
    assert "i want to train a model on stuff." in out
    # the label is consumed as the title, not also emitted as prose
    assert "Your own task (today):" not in out


def test_prose_lines_join_into_one_paragraph():
    # Two hard-wrapped source lines must reflow, not wrap each fragment alone.
    out = _joined("the quick brown fox\njumps over the lazy dog", width=80)
    assert "the quick brown fox jumps over the lazy dog" in out


def test_blockquote_gets_bar_and_keeps_list_items_separate():
    out = _joined("> intro line\n> - first item\n> - second item")
    assert "▌ intro line" in out
    assert "▌ - first item" in out
    assert "▌ - second item" in out


def test_table_aligns_and_drops_separator_row():
    out = _joined("| Do | Command |\n|----|---------|\n| Sign in | `castform login` |")
    assert "---" not in out  # separator row dropped
    assert "Do" in out and "Sign in" in out and "castform login" in out


def test_cmd_help_prints_guide(capsys):
    assert help_cmd._cmd_help(object()) == 0
    out = capsys.readouterr().out
    assert "Get started" in out and "Quick commands" in out
