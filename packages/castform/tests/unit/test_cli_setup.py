"""Slice 1.8 offline: `castform setup` scaffolds the right files (no login)."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from castform.cli import setup

from ._scaffold import discover_env_class, load_module

_SKILLS = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)

# Every template now ships these — a runnable seed + tiny day-one datasets.
_SEED_FILES = (
    "pyproject.toml",
    "main.py",
    "train.jsonl",
    "eval.jsonl",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ns(tmp, **kw):
    base = dict(
        dir=str(tmp),
        agent="both",
        force=False,
        skip_login=True,
        no_template=False,
        template="generic",
        verbose=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_setup_writes_both_agents(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(setup, "version", lambda distribution: "0.1.2")
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "GETTING_STARTED.md").exists()
    # both agents get the full skill set, each under its own skills dir
    for name in _SKILLS:
        assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").exists()
        assert (tmp_path / ".agents" / "skills" / name / "SKILL.md").exists()
    # same prose, but each body points at its own agent's skills dir
    assert ".claude/skills" in (tmp_path / "CLAUDE.md").read_text()
    assert ".agents/skills" in (tmp_path / "AGENTS.md").read_text()
    out = capsys.readouterr().out  # the get-started block is emitted
    assert "ask your agent" in out  # the prompt box
    assert "helpful commands" in out  # the commands divider
    assert "python main.py validate" in out
    assert "python main.py launch" in out
    assert "castform launch" not in out
    assert "castform guide" in out  # the guide is surfaced in the commands


def test_setup_claude_only(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    assert (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / ".claude" / "skills" / "launch-run" / "SKILL.md").exists()


def test_setup_codex_writes_agents_skills(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path, agent="codex")) == 0
    assert (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / ".claude").exists()  # no claude dir for codex-only
    # codex gets the same skills under .agents/skills, and AGENTS.md points there
    for name in _SKILLS:
        assert (tmp_path / ".agents" / "skills" / name / "SKILL.md").exists()
    guide = (tmp_path / "AGENTS.md").read_text()
    assert ".agents/skills" in guide
    assert ".claude/skills" not in guide
    assert "Keep correctness dominant" in guide
    view_progress = (tmp_path / ".agents" / "skills" / "view-progress" / "SKILL.md").read_text()
    assert "castform runs rollout" in view_progress
    assert "--view" not in view_progress


def test_setup_default_shows_grouped_summary(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "agent guides" in out and "agent skills" in out
    assert "env template" in out  # generic now ships a seed main.py + datasets
    # no per-file log by default
    assert "wrote " not in out
    assert "SKILL.md" not in out


def test_setup_no_template_rerun_reports_already_present(tmp_path, capsys):
    """The docs+skills scaffold is idempotent: re-running --no-template reports
    everything already present. (A seed template instead hits the main.py overwrite
    guard on re-run — see test_setup_refuses_existing_main_py.)"""
    assert setup._cmd_setup(_ns(tmp_path, no_template=True)) == 0
    capsys.readouterr()  # discard first run
    assert setup._cmd_setup(_ns(tmp_path, no_template=True)) == 0
    assert "already present" in capsys.readouterr().out


def test_setup_verbose_lists_every_file(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path, verbose=True)) == 0
    out = capsys.readouterr().out
    assert "wrote " in out and "SKILL.md" in out  # full per-file log
    assert "agent skills" not in out  # grouped summary suppressed


def test_setup_skips_existing_without_force(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    assert (tmp_path / "CLAUDE.md").read_text() == "MINE"


def test_setup_force_overwrites(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, agent="claude", force=True)) == 0
    assert (tmp_path / "CLAUDE.md").read_text() != "MINE"


def test_setup_content_cites_real_verbs(tmp_path):
    setup._cmd_setup(_ns(tmp_path, agent="claude"))
    guide = (tmp_path / "CLAUDE.md").read_text()
    launch_skill = (tmp_path / ".claude" / "skills" / "launch-run" / "SKILL.md").read_text()
    assert "python main.py validate" in guide
    assert "python main.py launch" in guide
    assert "castform launch" not in guide + launch_skill
    assert "max_context_tokens" in launch_skill
    assert "upload_assets(bundle=bundle" in launch_skill


def test_setup_generic_ships_seed_env_and_data(tmp_path, monkeypatch):
    """The generic flow ships a runnable seed main.py (a minimal single-turn env)
    plus tiny day-one datasets, so `python main.py validate` runs with zero edits."""
    monkeypatch.setattr(setup, "version", lambda distribution: "0.1.2")
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "CLAUDE.md").exists()  # docs written
    for name in _SEED_FILES:
        assert (tmp_path / name).exists(), name
    # the seed loads: a no-tool CustomEnv + non-empty datasets
    module = load_module(tmp_path / "main.py")
    assert discover_env_class(module).__name__ == "CustomEnv"
    assert _read_jsonl(tmp_path / "train.jsonl")
    assert _read_jsonl(tmp_path / "eval.jsonl")
    project = tomllib.loads((tmp_path / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == [
        "benchmax==0.1.2",
        "castform==0.1.2",
    ]


def test_setup_no_template_ships_docs_and_skills_only(tmp_path):
    """--no-template scaffolds docs + skills only — no seed main.py / datasets."""
    assert setup._cmd_setup(_ns(tmp_path, no_template=True)) == 0
    assert (tmp_path / "CLAUDE.md").exists()  # docs written
    for name in _SEED_FILES:
        assert not (tmp_path / name).exists(), name


def test_setup_refuses_existing_main_py(tmp_path, capsys):
    """The overwrite guard now applies to every seed-writing template: an existing
    main.py + no --force must fail loudly and leave the file untouched."""
    (tmp_path / "main.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path)) == 1
    assert (tmp_path / "main.py").read_text() == "MINE"  # untouched
    assert "already exists" in capsys.readouterr().err


def test_setup_template_rag_uses_generic_seed_and_provider_guidance(
    tmp_path, monkeypatch
):
    """The RAG template links to maintained examples instead of copying an adapter."""
    monkeypatch.setattr(setup, "version", lambda distribution: "0.1.2")
    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0

    module = load_module(tmp_path / "main.py")
    assert discover_env_class(module).__name__ == "CustomEnv"
    assert set(_read_jsonl(tmp_path / "train.jsonl")[0]) == {
        "prompt",
        "ground_truth",
    }

    project = tomllib.loads((tmp_path / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == [
        "benchmax==0.1.2",
        "castform[rag]==0.1.2",
    ]
    guidance = "\n".join(
        [
            (tmp_path / "GETTING_STARTED.md").read_text(),
            (
                tmp_path / ".agents" / "skills" / "generate-data" / "SKILL.md"
            ).read_text(),
        ]
    )
    for example in ("neon_rag", "turbopuffer_rag", "chroma_rag", "pinecone_rag"):
        assert f"benchmax/tree/main/examples/{example}" in guidance
    assert "HostedCorpusSearch" not in (tmp_path / "main.py").read_text()


def test_setup_template_rag_refuses_existing_main_py(tmp_path, capsys):
    """The hollow-pass guard: existing main.py + --template rag (no --force) must
    fail loudly and leave the file untouched — never silently skip."""
    (tmp_path / "main.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 1
    assert (tmp_path / "main.py").read_text() == "MINE"  # untouched
    assert "already exists" in capsys.readouterr().err


def test_setup_template_rag_force_overwrites(tmp_path):
    (tmp_path / "main.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, template="rag", force=True)) == 0
    main_py = (tmp_path / "main.py").read_text()
    assert "class CustomEnv(BaseEnv)" in main_py and main_py != "MINE"


def test_setup_force_replaces_main_py_but_keeps_datasets(tmp_path):
    """--force clears the main.py overwrite guard but must NOT clobber datasets —
    real project-generated output is never overwritten by the placeholder seed."""
    (tmp_path / "main.py").write_text("MINE")
    (tmp_path / "train.jsonl").write_text("REAL TRAIN DATA")
    (tmp_path / "eval.jsonl").write_text("REAL EVAL DATA")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'mine'\n")
    assert setup._cmd_setup(_ns(tmp_path, template="rag", force=True)) == 0
    assert (tmp_path / "main.py").read_text() != "MINE"  # guard cleared, seed written
    assert (tmp_path / "train.jsonl").read_text() == "REAL TRAIN DATA"  # kept
    assert (tmp_path / "eval.jsonl").read_text() == "REAL EVAL DATA"  # kept
    assert "name = 'mine'" in (tmp_path / "pyproject.toml").read_text()


def test_getting_started_teaches_script_workflow(tmp_path):
    """GETTING_STARTED teaches the script workflow and baseline handoff."""
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    gs = (tmp_path / "GETTING_STARTED.md").read_text()
    assert "Build a Castform environment for <task>" in gs
    assert "uv run python main.py validate" in gs
    assert "uv run python main.py launch" in gs
    assert "green baseline" in gs and "iterate" in gs.lower()
    assert "castform launch" not in gs
    assert "castform data" not in gs
    assert "Useful CLI commands" in gs


def test_setup_env_conditional_surfacing(tmp_path):
    """Generic scaffold carries NO RAG-specific guidance; --template rag surfaces it.
    Single-source docs use `<!-- rag:start/end -->` markers that setup strips for
    non-rag templates. Also: the scaffold guide names main.py and drops the old
    'setup does not write main.py' convention."""

    def _emit(template):
        d = tmp_path / template
        assert setup._cmd_setup(_ns(d, template=template)) == 0
        texts = [(d / "CLAUDE.md").read_text()]
        for name in _SKILLS:
            texts.append((d / ".claude" / "skills" / name / "SKILL.md").read_text())
        return "\n".join(texts)

    generic = _emit("generic")
    rag = _emit("rag")

    # the delimiter comments never leak into either emitted scaffold
    for blob in (generic, rag):
        assert "rag:start" not in blob and "rag:end" not in blob

    # RAG-only guidance is present in the RAG scaffold and absent from generic.
    for sentinel in (
        "reference_chunks",
        "source-ID canonicalization",
        "castform.rag",
        "benchmax/tree/main/examples",
    ):
        assert sentinel in rag, f"rag scaffold missing {sentinel!r}"
        assert sentinel not in generic, f"generic scaffold leaked {sentinel!r}"

    # reversed convention + main.py naming in both scaffolds' guide/skills
    for blob in (generic, rag):
        assert "main.py" in blob
        assert "does not write" not in blob and "does **not** write" not in blob
