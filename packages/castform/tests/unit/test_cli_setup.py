"""Offline tests for the coding-agent guidance written by ``castform setup``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from castform import cli
from castform.cli import setup

_SKILLS = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)

_ENV_FILES = (
    "pyproject.toml",
    "main.py",
    "train.jsonl",
    "eval.jsonl",
)


def _ns(tmp: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dir": str(tmp),
        "agent": "both",
        "force": False,
        "skip_login": True,
        "verbose": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_setup_writes_guidance_for_both_agents(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "GETTING_STARTED.md").exists()

    for name in _SKILLS:
        assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").exists()
        assert (tmp_path / ".agents" / "skills" / name / "SKILL.md").exists()

    assert ".claude/skills" in (tmp_path / "CLAUDE.md").read_text()
    assert ".agents/skills" in (tmp_path / "AGENTS.md").read_text()

    output = capsys.readouterr().out
    assert "ask your agent" in output
    assert "Benchmax" in output and "examples" in output
    assert "helpful commands" in output


@pytest.mark.parametrize(
    ("agent", "guide", "skills_dir", "absent"),
    [
        ("claude", "CLAUDE.md", ".claude", "AGENTS.md"),
        ("codex", "AGENTS.md", ".agents", "CLAUDE.md"),
    ],
)
def test_setup_can_target_one_agent(tmp_path, agent, guide, skills_dir, absent):
    assert setup._cmd_setup(_ns(tmp_path, agent=agent)) == 0
    assert (tmp_path / guide).exists()
    assert not (tmp_path / absent).exists()
    for name in _SKILLS:
        assert (tmp_path / skills_dir / "skills" / name / "SKILL.md").exists()


def test_setup_writes_no_environment_files(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    for name in _ENV_FILES:
        assert not (tmp_path / name).exists(), name
    assert not (tmp_path / "tests").exists()


def test_setup_never_changes_existing_environment_files(tmp_path):
    for name in _ENV_FILES:
        (tmp_path / name).write_text(f"existing {name}")

    assert setup._cmd_setup(_ns(tmp_path, force=True)) == 0
    for name in _ENV_FILES:
        assert (tmp_path / name).read_text() == f"existing {name}"


def test_setup_rerun_reports_guidance_already_present(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    capsys.readouterr()
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert "already present" in capsys.readouterr().out


def test_setup_verbose_lists_every_guidance_file(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path, verbose=True)) == 0
    output = capsys.readouterr().out
    assert "wrote " in output and "SKILL.md" in output
    assert "agent skills" not in output


def test_setup_preserves_or_forces_existing_guidance(tmp_path):
    guide = tmp_path / "CLAUDE.md"
    guide.write_text("MINE")

    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    assert guide.read_text() == "MINE"

    assert setup._cmd_setup(_ns(tmp_path, agent="claude", force=True)) == 0
    assert guide.read_text() != "MINE"


def test_setup_guidance_uses_benchmax_examples_as_source_of_truth(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    guide = (tmp_path / "CLAUDE.md").read_text()
    starter = (tmp_path / "GETTING_STARTED.md").read_text()
    design = (
        tmp_path / ".claude" / "skills" / "design-environment" / "SKILL.md"
    ).read_text()
    combined = "\n".join((guide, starter, design))

    assert "https://github.com/castform-ai/benchmax/tree/main/examples" in combined
    assert "source of truth" in combined
    assert "does not generate environment code" in combined
    assert "choose the closest" in combined
    assert "uv run python main.py validate" in combined
    assert "uv run python main.py launch" in combined


def test_setup_parser_has_no_environment_template_flags():
    setup_parser = next(
        action
        for action in cli.build_parser()._actions
        if action.dest == "command"
    ).choices["setup"]
    option_strings = {
        option
        for action in setup_parser._actions
        for option in action.option_strings
    }
    assert "--template" not in option_strings
    assert "--no-template" not in option_strings
