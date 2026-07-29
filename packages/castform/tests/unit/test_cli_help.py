"""Snapshot of `castform --help` — guards the command-group structure (slice 1.1).

The golden is committed next to this file. When commands are added, regenerate
with ``CASTFORM_UPDATE_GOLDEN=1 uv run pytest tests/unit/test_cli_help.py`` and
review the diff — an unintended change to the top-level surface fails this test.
"""

from __future__ import annotations

import os
from pathlib import Path

from castform.cli import build_parser, cli_version

_GOLDEN = Path(__file__).parent / "cli_help.golden"


def test_help_snapshot(monkeypatch):
    # Pin width so the wrapped help is identical across terminals / CI.
    monkeypatch.setenv("COLUMNS", "80")
    # Mask the live version so bumps don't churn the golden.
    got = build_parser().format_help().replace(f"v{cli_version()}", "v<version>")
    if os.environ.get("CASTFORM_UPDATE_GOLDEN"):
        _GOLDEN.write_text(got, encoding="utf-8")
    assert got == _GOLDEN.read_text(encoding="utf-8"), (
        "castform --help drifted from the committed snapshot; if intentional, "
        "regen with CASTFORM_UPDATE_GOLDEN=1 and review the diff."
    )


def test_retired_orchestration_commands_are_not_registered():
    top_level = next(action for action in build_parser()._actions if action.dest == "command")
    assert not ({"launch", "data", "corpus"} & set(top_level.choices))
