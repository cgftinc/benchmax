"""Dedicated tests for the shared CLI output helpers (castform.cli._output).

These are extracted, cross-module utilities (used by `runs rollout` and
`validate --reward-audit`), so per the repo CLAUDE.md they get their own
edge-case tests rather than only transitive coverage through callers.
"""

from __future__ import annotations

from castform.cli._output import final_answer, fmt_value, truncate


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("hi there", 40) == "hi there"

    def test_collapses_whitespace(self):
        assert truncate("a\n  b\tc", 40) == "a b c"

    def test_clips_with_ellipsis_at_n(self):
        out = truncate("abcdefghij", 5)
        assert out == "abcd…" and len(out) == 5

    def test_none_becomes_empty(self):
        assert truncate(None, 10) == ""

    def test_falsy_but_real_value_preserved(self):
        # A gold answer of 0 / False must render as-is, not blank (regression:
        # `str(text or "")` collapsed these to "").
        assert truncate(0, 10) == "0"
        assert truncate(0.0, 10) == "0.0"
        assert truncate(False, 10) == "False"

    def test_empty_string_stays_empty(self):
        assert truncate("", 10) == ""


class TestFinalAnswer:
    def test_returns_last_assistant_content(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "final"},
        ]
        assert final_answer(msgs) == "final"

    def test_none_and_empty(self):
        assert final_answer(None) is None
        assert final_answer([]) is None

    def test_no_assistant_turn(self):
        assert final_answer([{"role": "user", "content": "q"}]) is None

    def test_skips_empty_assistant_content(self):
        msgs = [
            {"role": "assistant", "content": "real"},
            {"role": "assistant", "content": ""},
        ]
        assert final_answer(msgs) == "real"


def test_fmt_value_compact_floats():
    assert fmt_value(0.12345) == "0.1235"
    assert fmt_value(3) == "3"
    assert fmt_value("x") == "x"
