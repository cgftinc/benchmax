"""Slice 1.8 offline: `castform setup` scaffolds the right files (no login)."""

from __future__ import annotations

import argparse

from benchmax.cli import setup

_SKILLS = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)

# castform setup ships these — never an env or datasets for the generic flow.
_NEVER_SHIPPED = ("run.py", "train_dataset.jsonl", "eval_dataset.jsonl")


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


def test_setup_writes_both_agents(tmp_path, capsys):
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
    assert "castform validate" in out
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
    assert ".agents/skills" in (tmp_path / "AGENTS.md").read_text()
    assert ".claude/skills" not in (tmp_path / "AGENTS.md").read_text()
    assert "Gate secondary bonuses" in (tmp_path / "AGENTS.md").read_text()
    view_progress = (
        tmp_path / ".agents" / "skills" / "view-progress" / "SKILL.md"
    ).read_text()
    assert "external-eval" in view_progress


def test_setup_default_shows_grouped_summary(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "agent guides" in out and "agent skills" in out
    assert "env template" not in out  # generic ships no run.py
    # no per-file log by default
    assert "wrote " not in out
    assert "SKILL.md" not in out


def test_setup_default_reports_already_present_on_rerun(tmp_path, capsys):
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    capsys.readouterr()  # discard first run
    assert setup._cmd_setup(_ns(tmp_path)) == 0
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
    assert "castform validate" in guide and "castform launch" in guide
    assert "max_rollout_len" in guide  # the real launch knob is documented


def test_setup_generic_ships_no_env_or_data(tmp_path):
    """The generic flow scaffolds docs + skills only — the agent writes run.py
    from the design-environment skill, and generates its own datasets."""
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "CLAUDE.md").exists()  # docs written
    for name in _NEVER_SHIPPED:
        assert not (tmp_path / name).exists(), name


def test_setup_generic_leaves_existing_run_py_untouched(tmp_path):
    """Generic never touches run.py, so a user's own run.py survives setup."""
    (tmp_path / "run.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "run.py").read_text() == "MINE"


def test_setup_template_rag_writes_searchenv(tmp_path):
    """--template rag writes a run.py the loader discovers as CustomSearchEnv (a
    SearchEnv subclass), and it constructs with no args, offline."""
    from benchmax.cli._project import _load_module_from_file, discover_env_class
    from benchmax.envs.postgres_search.search_env import SearchEnv

    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0
    run_py = (tmp_path / "run.py").read_text()
    assert "class CustomSearchEnv(SearchEnv)" in run_py
    assert "MAX_SEARCH_CALLS = 6" in run_py
    # Self-contained: the reward arithmetic + weights are visible/editable in the
    # file, and the run's budgets are baked in so it reproduces without CLI flags.
    assert "async def compute_reward" in run_py
    assert "W_CORRECTNESS" in run_py
    assert "VALIDATE_CONFIG = {" in run_py
    assert "LAUNCH_CONFIG = {" in run_py
    mod = _load_module_from_file(tmp_path / "run.py")
    env_cls = discover_env_class(mod)  # imported SearchEnv is ignored (other module)
    assert env_cls.__name__ == "CustomSearchEnv"
    assert issubclass(env_cls, SearchEnv)
    assert isinstance(env_cls(), SearchEnv)  # no-arg construct, no network
    # The config blocks are what validate/launch read (LoadedProject surfaces them).
    from benchmax.cli._project import _read_config

    assert _read_config(mod, "VALIDATE_CONFIG")["max_turns"] == 7
    assert _read_config(mod, "LAUNCH_CONFIG")["max_rollout_len"] == 16384


def test_setup_template_rag_writes_run_py_no_datasets(tmp_path):
    """--template rag ships the SearchEnv run.py only; datasets come from
    `castform data qa-gen`, so none are written at setup time."""
    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0
    assert (tmp_path / "run.py").exists()
    assert not (tmp_path / "train_dataset.jsonl").exists()
    assert not (tmp_path / "eval_dataset.jsonl").exists()


def test_setup_template_rag_refuses_existing_run_py(tmp_path, capsys):
    """The hollow-pass guard: existing run.py + --template rag (no --force) must
    fail loudly and leave the file untouched — never silently skip."""
    (tmp_path / "run.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 1
    assert (tmp_path / "run.py").read_text() == "MINE"  # untouched
    assert "already exists" in capsys.readouterr().err


def test_setup_template_rag_force_overwrites(tmp_path):
    (tmp_path / "run.py").write_text("MINE")
    assert setup._cmd_setup(_ns(tmp_path, template="rag", force=True)) == 0
    run_py = (tmp_path / "run.py").read_text()
    assert "CustomSearchEnv" in run_py and run_py != "MINE"


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


def test_rag_scaffold_reward_threads_canonicalize_and_timeout(tmp_path, monkeypatch):
    """The scaffold's inline compute_reward must honor a _canonicalize_id override
    (citations) and the env's judge_timeout — behavior the inherited SearchEnv had,
    which a naive inline reward silently drops (review findings #0, #6)."""
    import asyncio

    from benchmax.cli._project import _load_module_from_file, discover_env_class

    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0
    mod = _load_module_from_file(tmp_path / "run.py")
    env_cls = discover_env_class(mod)

    env = env_cls.__new__(env_cls)  # skip network __init__
    env._judge_model = "m"
    env._judge_base_url = "u"
    env._judge_timeout = 99.0
    env._judge_token_provider = lambda: "k"
    env._max_search_calls = 6
    env._w_search_efficiency = 0.1
    # A corpus-specific matcher (case-insensitive) — proves _canonicalize_id is threaded.
    env._canonicalize_id = lambda s: str(s or "").strip().lower()

    captured: dict = {}

    async def _fake_judge(**kw):
        captured.update(kw)
        return (1.0, 1.0)

    monkeypatch.setattr(mod, "judge_answer_quality", _fake_judge)

    msgs = [{"role": "assistant", "content": "<answer>x [Source: DOCA]</answer>"}]
    task = {
        "question": "Q",
        "ground_truth": "x",
        "reference_chunks": [{"metadata": {"file": "doca"}}],
    }
    reward = asyncio.run(env.compute_reward("r", msgs, task))

    assert captured["timeout"] == 99.0  # #6: env judge_timeout threaded
    # #0: "DOCA" cite matched gold "doca" via the injected lowercasing canonicalizer
    assert reward["citation_recall"] > 0
