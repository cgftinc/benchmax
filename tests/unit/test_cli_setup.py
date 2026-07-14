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

# Every template now ships these — a runnable seed + tiny day-one datasets.
_SEED_FILES = ("main.py", "train_dataset.jsonl", "eval_dataset.jsonl")


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
    assert "castform validate" in guide and "castform launch" in guide
    assert "max_rollout_len" in guide  # the real launch knob is documented


def test_setup_generic_ships_seed_env_and_data(tmp_path):
    """The generic flow ships a runnable seed main.py (a minimal single-turn env)
    plus tiny day-one datasets, so `python main.py validate` runs with zero edits."""
    from benchmax.cli._project import load_project

    assert setup._cmd_setup(_ns(tmp_path)) == 0
    assert (tmp_path / "CLAUDE.md").exists()  # docs written
    for name in _SEED_FILES:
        assert (tmp_path / name).exists(), name
    # the seed loads: a no-tool CustomEnv + non-empty datasets
    project = load_project(directory=str(tmp_path))
    assert project.env_class.__name__ == "CustomEnv"
    assert project.train_dataset and project.eval_dataset


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


def test_setup_template_rag_writes_searchenv(tmp_path):
    """--template rag writes a main.py the loader discovers as CustomSearchEnv (a
    SearchEnv subclass), and it constructs with no args, offline."""
    from benchmax.cli._project import _load_module_from_file, discover_env_class
    from benchmax.envs.postgres_search.search_env import SearchEnv

    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0
    main_py = (tmp_path / "main.py").read_text()
    assert "class CustomSearchEnv(SearchEnv)" in main_py
    assert "MAX_SEARCH_CALLS = 6" in main_py
    # Self-contained: the reward arithmetic + weights are visible/editable in the
    # file, and the run's budgets are baked in so it reproduces without CLI flags.
    assert "async def compute_reward" in main_py
    assert "W_CORRECTNESS" in main_py
    assert "VALIDATE_CONFIG = {" in main_py
    assert "LAUNCH_CONFIG = {" in main_py
    mod = _load_module_from_file(tmp_path / "main.py")
    env_cls = discover_env_class(mod)  # imported SearchEnv is ignored (other module)
    assert env_cls.__name__ == "CustomSearchEnv"
    assert issubclass(env_cls, SearchEnv)
    assert isinstance(env_cls(), SearchEnv)  # no-arg construct, no network
    # The config blocks are what validate/launch read (LoadedProject surfaces them).
    from benchmax.cli._project import _read_config

    assert _read_config(mod, "VALIDATE_CONFIG")["max_turns"] == 7
    assert _read_config(mod, "LAUNCH_CONFIG")["max_rollout_len"] == 16384


def test_setup_template_rag_writes_seed_and_datasets(tmp_path):
    """--template rag ships the SearchEnv main.py + tiny seed datasets (question/
    answer/reference_chunks shape). Real data replaces them via `castform data
    qa-gen`; the datasets skip-if-exists, so real data is never clobbered."""
    from benchmax.cli._project import load_project

    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0
    assert (tmp_path / "main.py").exists()
    assert (tmp_path / "train_dataset.jsonl").exists()
    assert (tmp_path / "eval_dataset.jsonl").exists()
    project = load_project(directory=str(tmp_path))
    assert project.env_class.__name__ == "CustomSearchEnv"
    assert set(project.train_dataset[0]) >= {"question", "answer", "reference_chunks"}


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
    assert "CustomSearchEnv" in main_py and main_py != "MINE"


def test_setup_force_replaces_main_py_but_keeps_datasets(tmp_path):
    """--force clears the main.py overwrite guard but must NOT clobber datasets —
    real `castform data qa-gen` output is never overwritten by the placeholder seed."""
    (tmp_path / "main.py").write_text("MINE")
    (tmp_path / "train_dataset.jsonl").write_text("REAL TRAIN DATA")
    (tmp_path / "eval_dataset.jsonl").write_text("REAL EVAL DATA")
    assert setup._cmd_setup(_ns(tmp_path, template="rag", force=True)) == 0
    assert (tmp_path / "main.py").read_text() != "MINE"  # guard cleared, seed written
    assert (tmp_path / "train_dataset.jsonl").read_text() == "REAL TRAIN DATA"  # kept
    assert (tmp_path / "eval_dataset.jsonl").read_text() == "REAL EVAL DATA"  # kept


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

    # RAG-only sentinels: present in the rag scaffold, absent in the generic one
    for sentinel in (
        "gold-hit",
        "retrieval_hit",
        "MAX_TOOL_OUTPUT_CHARS",
        "MAX_SEARCH_CALLS",
        "reference_chunks",
        "SearchEnv",
    ):
        assert sentinel in rag, f"rag scaffold missing {sentinel!r}"
        assert sentinel not in generic, f"generic scaffold leaked {sentinel!r}"

    # reversed convention + main.py naming in both scaffolds' guide/skills
    for blob in (generic, rag):
        assert "main.py" in blob
        assert "does not write" not in blob and "does **not** write" not in blob


def test_rag_scaffold_reward_threads_canonicalize_and_timeout(tmp_path, monkeypatch):
    """The scaffold's inline compute_reward must honor a _canonicalize_id override
    (citations) and the env's judge_timeout — behavior the inherited SearchEnv had,
    which a naive inline reward silently drops (review findings #0, #6)."""
    import asyncio

    from benchmax.cli._project import _load_module_from_file, discover_env_class

    assert setup._cmd_setup(_ns(tmp_path, template="rag")) == 0
    mod = _load_module_from_file(tmp_path / "main.py")
    env_cls = discover_env_class(mod)

    env = env_cls.__new__(env_cls)  # skip network __init__
    env._judge_model = "m"
    env._judge_base_url = "u"
    env._judge_timeout = 99.0
    env._judge_token_provider = lambda: "k"
    # A corpus-specific matcher (case-insensitive) — proves _canonicalize_id is threaded.
    env._canonicalize_id = lambda s: str(s or "").strip().lower()

    captured: dict = {}

    async def _fake_judge(**kw):
        captured.update(kw)
        return {"score": 1.0}

    monkeypatch.setattr(mod, "evaluate_single_rubric", _fake_judge)

    msgs = [{"role": "assistant", "content": "<answer>x [Source: DOCA]</answer>"}]
    task = {
        "question": "Q",
        "ground_truth": "x",
        "reference_chunks": [{"metadata": {"file": "doca"}}],
    }
    reward = asyncio.run(
        env.compute_reward("r", msgs, task, termination_reason="finished")
    )

    assert captured["timeout"] == 99.0  # #6: env judge_timeout threaded
    # #0: "DOCA" cite matched gold "doca" via the injected lowercasing canonicalizer
    # (retrieval_hit is the ungated recall component)
    assert reward["retrieval_hit"] > 0
