"""Slice 1.8 offline: `castform setup` scaffolds the right files (no login)."""

from __future__ import annotations

import argparse
import json

from benchmax.cli import setup

_SKILLS = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)


_TEMPLATES = ("run.py", "train_dataset.jsonl", "eval_dataset.jsonl")


def _ns(tmp, **kw):
    base = dict(
        dir=str(tmp),
        agent="both",
        force=False,
        skip_login=True,
        no_template=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_setup_writes_both_agents(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "GETTING_STARTED.md").exists()
    for name in _SKILLS:
        assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").exists()
    # CLAUDE.md and AGENTS.md share one source (same body)
    assert (tmp_path / "CLAUDE.md").read_text() == (tmp_path / "AGENTS.md").read_text()
    assert "Quick commands" in capsys.readouterr().out  # starter emitted


def test_setup_claude_only(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    assert (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / ".claude" / "skills" / "launch-run" / "SKILL.md").exists()


def test_setup_codex_only_has_no_skills(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path, agent="codex")) == 0
    assert (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / ".claude").exists()


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
    assert "castform validate" in guide and "castform launch" in guide
    assert "max_rollout_len" in guide  # the real launch knob is documented


def test_setup_writes_starter_template(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    for name in _TEMPLATES:
        assert (tmp_path / name).exists(), name
    run_py = (tmp_path / "run.py").read_text()
    assert "class StarterEnv(BaseEnv)" in run_py
    # No importable last_answer helper exists — the template must inline it, else
    # it would ImportError before validate even runs.
    assert "def _last_answer" in run_py
    assert not any(
        line.strip().startswith(("from", "import")) and "last_answer" in line
        for line in run_py.splitlines()
    )


def test_setup_no_template_omits_starter(tmp_path):
    assert setup._cmd_setup(_ns(tmp_path, no_template=True)) == 0
    assert (tmp_path / "CLAUDE.md").exists()  # docs still written
    for name in _TEMPLATES:
        assert not (tmp_path / name).exists(), name


def test_setup_template_skips_existing_without_force(tmp_path):
    (tmp_path / "run.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "run.py").read_text() == "MINE"


def test_setup_template_force_overwrites(tmp_path):
    (tmp_path / "run.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, force=True)) == 0
    assert (tmp_path / "run.py").read_text() != "MINE"


def test_getting_started_uses_one_prompt_model(tmp_path):
    """GETTING_STARTED mirrors the UI's one-prompt model (3 AGENT_PROMPTS variants
    + baseline handoff), not the old rigid 1-4 checklist."""
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    gs = (tmp_path / "GETTING_STARTED.md").read_text()
    # the 3 paste-able variants (generic / rag / traces) matching the UI backbone
    assert "improve a model on <your task>" in gs
    assert "retrieval-augmented" in gs
    assert "production traces" in gs
    # explicit baseline -> iterate-or-launch handoff
    assert "green baseline" in gs and "iterate or launch" in gs.lower()
    # the rigid checklist + reward-Q&A-first framing are gone
    assert "work through them in order" not in gs.lower()
    assert "how to reward it" not in gs
    assert "Quick commands" in gs


def test_starter_reward_discriminates(tmp_path):
    """Deterministic proof the shipped reward is discriminating: 1.0 on a correct
    answer, 0.0 on a wrong one. (A strong cheap model may still score uniformly on
    a tiny live `validate` sample — that's expected, not a broken reward.)"""
    import asyncio

    from benchmax.cli._project import load_project

    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    proj = load_project(directory=str(tmp_path))
    env = proj.env_class()
    row = proj.train_dataset[0]  # an easy row with a short ground_truth

    def reward(answer):
        msgs = [{"role": "assistant", "content": answer}]
        return asyncio.run(env.compute_reward("r", msgs, row))

    assert reward(row["ground_truth"]) == {"correct": 1.0}
    assert reward("definitely the wrong answer") == {"correct": 0.0}


def test_starter_dataset_has_easy_hard_mix(tmp_path):
    """The dataset interleaves easy + verified-hard rows so the default 2-example
    validate sample spans the reward variance (first two rows differ in kind)."""
    assert setup._cmd_setup(_ns(tmp_path, agent="claude")) == 0
    lines = (tmp_path / "train_dataset.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 6
    gt_lens = [len(json.loads(line)["ground_truth"]) for line in lines[:2]]
    assert min(gt_lens) <= 2 and max(gt_lens) >= 20  # row0 easy, row1 hard
