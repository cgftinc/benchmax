"""Slice 1.4 CLI: project loading + validate report rendering. Offline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from benchmax.cli import build_parser, validate
from benchmax.cli._project import (
    ProjectError,
    _load_jsonl,
    _load_module_from_file,
    discover_env_class,
)
from benchmax.platform.client import ExampleValidation, ValidationResult
from benchmax.platform.validation import ValidationReport

# --- project loader ------------------------------------------------------

_ENV_SRC = (
    "from benchmax.envs.base_env import BaseEnv\n\nclass {name}(BaseEnv):\n    pass\n"
)


def _write_run(tmp_path, *names):
    body = "from benchmax.envs.base_env import BaseEnv\n\n"
    for n in names:
        body += f"class {n}(BaseEnv):\n    pass\n\n"
    p = tmp_path / "main.py"
    p.write_text(body)
    return p


def test_discover_single_env(tmp_path):
    mod = _load_module_from_file(_write_run(tmp_path, "MyEnv"))
    assert discover_env_class(mod).__name__ == "MyEnv"


def test_discover_no_env_raises(tmp_path):
    p = tmp_path / "main.py"
    p.write_text("x = 1\n")
    mod = _load_module_from_file(p)
    with pytest.raises(ProjectError, match="No BaseEnv"):
        discover_env_class(mod)


def test_discover_ambiguous_raises(tmp_path):
    mod = _load_module_from_file(_write_run(tmp_path, "EnvA", "EnvB"))
    with pytest.raises(ProjectError, match="Multiple env classes"):
        discover_env_class(mod)


def test_discover_explicit_name(tmp_path):
    mod = _load_module_from_file(_write_run(tmp_path, "EnvA", "EnvB"))
    assert discover_env_class(mod, "EnvB").__name__ == "EnvB"


def test_load_jsonl_ok(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert _load_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_bad_line(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ProjectError, match="invalid JSON"):
        _load_jsonl(p)


def test_load_jsonl_missing(tmp_path):
    with pytest.raises(ProjectError, match="not found"):
        _load_jsonl(tmp_path / "nope.jsonl")


def test_load_project_bad_module_is_project_error():
    # A missing dep / bad module path must surface as ProjectError, not a raw
    # ModuleNotFoundError traceback (regression: the --module branch was unwrapped).
    from benchmax.cli._project import load_project

    with pytest.raises(ProjectError, match="Could not import module"):
        load_project(module_path="benchmax.totally.not.a.module")


# --- validate report rendering ------------------------------------------


class _FakeProject:
    training_mode = "rl"
    env_class = type("E", (), {})
    train_dataset = [{"prompt": "x"}]
    eval_dataset = []
    module = None
    from_file = True
    launch_config: dict = {}
    validate_config: dict = {}


def _validate_ns(**over) -> argparse.Namespace:
    base = dict(
        dir=".",
        run_file="main.py",
        module=None,
        env_class=None,
        train="train_dataset.jsonl",
        eval="eval_dataset.jsonl",
        env_arg=None,
        pip=None,
        provider=None,
        model=None,
        examples=2,
        group_samples=2,
        max_turns=4,
        max_tool_calls=8,
        local_only=False,
        verbose=False,
        full_messages=False,
        reward_audit=False,
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _report(examples, group, local_ran=False):
    remote = ValidationResult(examples=examples, group_reward=group)
    return ValidationReport(
        local_passed=0,
        local_failed=0,
        remote=remote,
        local_ran=local_ran,
        remote_ran=True,
    )


def _patch(monkeypatch, report):
    monkeypatch.setattr(validate, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: report)


def test_validate_shows_rewards_and_group(monkeypatch, capsys):
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"acc": 1.0}),
            ExampleValidation(index=1, ok=True, rewards={"acc": 0.0}),
        ],
        group=ExampleValidation(index=-1, ok=True, rewards={"rank": 0.5}),
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0
    out = capsys.readouterr().out
    assert "acc" in out and "total reward" in out
    assert "group reward" in out and "rank=0.5" in out
    assert "GREEN baseline" in out


def test_validate_surfaces_error(monkeypatch, capsys):
    report = _report(
        examples=[ExampleValidation(index=0, ok=False, error="bad judge api key")],
        group=ExampleValidation(index=-1, ok=False, error="group: bad judge api key"),
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 1  # report not ok
    out = capsys.readouterr().out
    assert "bad judge api key" in out
    assert "reward errors" in out and "FAILED" in out
    assert "validate failed" in out


def test_validate_group_not_run(monkeypatch, capsys):
    report = _report(
        examples=[ExampleValidation(index=0, ok=True, rewards={"acc": 1.0})],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0
    out = capsys.readouterr().out
    assert "group reward" in out and "not run" in out


def test_validate_json(monkeypatch, capsys):
    report = _report(
        examples=[ExampleValidation(index=0, ok=True, rewards={"acc": 1.0})],
        group=ExampleValidation(index=-1, ok=True, rewards={"rank": 0.5}),
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns(json=True)) == 0
    out = capsys.readouterr().out
    assert '"rewards"' in out and '"acc": 1.0' in out


def test_validate_json_includes_messages(monkeypatch, capsys):
    # --json carries the captured transcript when full_messages surfaced it, so a
    # reward audit can read real completions, not just scores.
    transcript = [{"role": "assistant", "content": "answer text"}]
    report = _report(
        examples=[
            ExampleValidation(
                index=0, ok=True, rewards={"acc": 1.0}, messages=transcript
            ),
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns(json=True)) == 0
    out = capsys.readouterr().out
    assert '"messages"' in out and "answer text" in out


def test_validate_probe_renders_at_command_layer(monkeypatch, capsys):
    """An env overriding validate_probe gets its result rendered as a scorecard row
    (human) and a `probe` key (--json). This runs with validate_env MOCKED, so a
    row here proves the probe fires at the COMMAND layer — not inside validate_env
    (the original _run_local_checks defect: local checks don't run on default
    remote validate, so a probe there would be inert)."""
    from benchmax.envs.base_env import BaseEnv

    class _ProbeEnv(BaseEnv):
        async def list_tools(self):
            return []

        async def run_tool(self, rollout_id, tool_name, **k):
            return ""

        async def compute_reward(self, rollout_id, messages, task, **k):
            return {}

        async def validate_probe(self, eval_dataset):
            return {"ok": True, "summary": "gold-hit@10 = 0.60", "value": 0.6}

    class _P:
        training_mode = "rl"
        env_class = _ProbeEnv
        train_dataset = [{"prompt": "x"}]
        eval_dataset = [{"question": "q", "answer": "a"}]
        module = None
        from_file = True
        launch_config: dict = {}
        validate_config: dict = {}

    report = _report(
        examples=[ExampleValidation(index=0, ok=True, rewards={"acc": 1.0})], group=None
    )
    monkeypatch.setattr(validate, "load_project", lambda **k: _P())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: report)

    # default args: local_only=False → local=False (the remote path)
    assert validate._cmd_validate(_validate_ns()) == 0
    out = capsys.readouterr().out
    assert "validate probe" in out and "gold-hit@10 = 0.60" in out

    assert validate._cmd_validate(_validate_ns(json=True)) == 0
    out = capsys.readouterr().out
    assert '"probe"' in out and "gold-hit@10 = 0.60" in out


def test_validate_no_probe_row_when_env_does_not_override(monkeypatch, capsys):
    """An env that doesn't override validate_probe → no probe row (no regression)."""
    report = _report(
        examples=[ExampleValidation(index=0, ok=True, rewards={"acc": 1.0})], group=None
    )
    _patch(monkeypatch, report)  # _FakeProject.env_class is a bare type (no probe)
    assert validate._cmd_validate(_validate_ns()) == 0
    assert "validate probe" not in capsys.readouterr().out


def test_env_arg_parsing():
    assert validate._parse_env_args(["a=1", "b=hi", "c=true"]) == {
        "a": 1,
        "b": "hi",
        "c": True,
    }


def test_validate_pip_forwards_to_sandbox(monkeypatch):
    # --pip must reach validate_env as pip_dependencies (provider RAG envs whose
    # search client imports a provider SDK hollow-green without it in the sandbox).
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={})], group=None
        )

    monkeypatch.setattr(validate, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(_validate_ns(pip=["turbopuffer", "pinecone>=5"]))
    assert captured["pip_dependencies"] == ["turbopuffer", "pinecone>=5"]


def test_validate_pip_repeatable_in_parser():
    args = build_parser().parse_args(
        ["validate", "--pip", "turbopuffer", "--pip", "chromadb>=1.0.0"]
    )
    assert args.pip == ["turbopuffer", "chromadb>=1.0.0"]


def test_validate_turn_budget_forwards_to_rollout(monkeypatch):
    # --max-turns / --max-tool-calls must reach validate_env so a deep-search env
    # (e.g. SearchEnv MAX_SEARCH_CALLS=6) can be validated at the budget the prompt
    # advertises, instead of the truncating 4/8 default.
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={})], group=None
        )

    monkeypatch.setattr(validate, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(_validate_ns(max_turns=11, max_tool_calls=11))
    assert captured["max_turns"] == 11
    assert captured["max_tool_calls"] == 11


def test_validate_turn_budget_defaults_in_parser():
    # Parser default is None (unset) so main.py's VALIDATE_CONFIG can supply it;
    # _cmd_validate resolves None → config → the 4/8 fallback.
    args = build_parser().parse_args(["validate"])
    assert args.max_turns is None and args.max_tool_calls is None
    args = build_parser().parse_args(
        ["validate", "--max-turns", "11", "--max-tool-calls", "12"]
    )
    assert args.max_turns == 11 and args.max_tool_calls == 12


def _validate_config_project(**config):
    class _P(_FakeProject):
        validate_config = config

    return _P()


def test_validate_config_supplies_budget_when_flag_omitted(monkeypatch):
    # main.py VALIDATE_CONFIG fills max_turns/examples when the CLI omits them.
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={})], group=None
        )

    monkeypatch.setattr(
        validate,
        "load_project",
        lambda **k: _validate_config_project(max_turns=9, max_tool_calls=7, examples=5),
    )
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(
        _validate_ns(max_turns=None, max_tool_calls=None, examples=None)
    )
    assert captured["max_turns"] == 9
    assert captured["max_tool_calls"] == 7
    assert captured["remote_examples"] == 5


def test_validate_cli_flag_overrides_config(monkeypatch):
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={})], group=None
        )

    monkeypatch.setattr(
        validate, "load_project", lambda **k: _validate_config_project(max_turns=9)
    )
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(
        _validate_ns(max_turns=3)
    )  # explicit flag wins over config 9
    assert captured["max_turns"] == 3


def test_validate_falls_back_to_default_without_config_or_flag(monkeypatch):
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={})], group=None
        )

    monkeypatch.setattr(validate, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(
        _validate_ns(max_turns=None, max_tool_calls=None, examples=None)
    )
    assert captured["max_turns"] == 4 and captured["max_tool_calls"] == 8
    assert captured["remote_examples"] == 2


def test_validate_provider_injects_sdk(monkeypatch):
    # --provider chroma must reach validate_env as the provider's SDK (incl. the
    # un-guessable snowballstemmer) merged with any --pip — without the agent
    # naming the package.
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={})], group=None
        )

    monkeypatch.setattr(validate, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(_validate_ns(pip=["mydep"], provider="chroma"))
    assert captured["pip_dependencies"] == [
        "mydep",
        "chromadb>=1.0.0",
        "snowballstemmer>=2.2.0",
    ]


# --- constant / all-zero reward warning ---------------------------------


def test_validate_warns_on_constant_component(monkeypatch, capsys):
    # 'fmt' is uniformly 0 across both rollouts (the "can't learn" footgun);
    # 'acc' varies and must NOT be flagged. Soft warning — exit code unchanged.
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"acc": 1.0, "fmt": 0.0}),
            ExampleValidation(index=1, ok=True, rewards={"acc": 0.0, "fmt": 0.0}),
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0  # still green
    out = capsys.readouterr().out
    assert "some components constant" in out and "'fmt' never vary" in out
    assert "'acc' never vary" not in out


# --- hollow-green recommendation carries an actionable hint ----------------

_GREEN_REPORT = type("R", (), {"ok": True})()


def test_recommendation_hollow_green_includes_remediation():
    # A constant total → the hint must name the concrete next moves so the agent
    # doesn't re-validate blindly: --full-messages to read a swallowed Error:, and
    # --provider/--pip for a provider RAG env.
    rec = validate._recommendation(_GREEN_REPORT, [{"acc": 0.5}, {"acc": 0.5}])
    assert "NO training signal" in rec
    assert "--full-messages" in rec and "Error:" in rec
    assert "--provider" in rec and "--pip" in rec


def test_recommendation_varying_reward_has_no_hint():
    # Reward total varies → the plain GREEN verdict, no remediation noise.
    rec = validate._recommendation(_GREEN_REPORT, [{"acc": 1.0}, {"acc": 0.0}])
    assert "GREEN baseline" in rec
    assert "--full-messages" not in rec and "--provider" not in rec


def _skill_text(name: str) -> str:
    skill = Path(validate.__file__).parent / "scaffold/skills" / name / "SKILL.md"
    return skill.read_text("utf-8")


def test_skill_green_sample_line_byte_equal_to_code():
    # verify-environment renders the GREEN verdict as a sample scorecard line; it
    # must stay byte-equal to what _recommendation prints (2-space indent) so the
    # doc never drifts from the tool.
    green = validate._recommendation(_GREEN_REPORT, [{"acc": 1.0}, {"acc": 0.0}])
    assert f"  {green}" in _skill_text("verify-environment")


def test_skill_hollow_green_prose_reflects_new_remediation():
    # The paired edit (N6): the doc's hollow-green guidance names the same new flags
    # the code now recommends.
    text = _skill_text("verify-environment")
    assert "--full-messages" in text
    assert "--provider <name>" in text and "--pip <sdk>" in text


def test_validate_constant_needs_two_rollouts(monkeypatch, capsys):
    # A single rollout can't establish variance — don't warn.
    report = _report(
        examples=[ExampleValidation(index=0, ok=True, rewards={"acc": 0.0})],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0
    assert "never vary" not in capsys.readouterr().out


def test_validate_all_constant_is_hollow_green(monkeypatch, capsys):
    # validate "passes" (report.ok) but every reward is 0 — no training signal.
    # The headline check + recommendation must call this out, not green-light it.
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"acc": 0.0, "fmt": 0.0}),
            ExampleValidation(index=1, ok=True, rewards={"acc": 0.0, "fmt": 0.0}),
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0  # exit code unchanged
    out = capsys.readouterr().out
    assert "rewards DON'T vary" in out
    assert "NO training signal" in out
    assert "GREEN baseline" not in out


def test_validate_ragged_zero_is_hollow_green(monkeypatch, capsys):
    # Regression: a component present in <2 rollouts can't be "constant", so the
    # old per-component gate green-lit an all-zero run with ragged keys. The
    # total-reward gate catches it.
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"score": 0.0}),
            ExampleValidation(index=1, ok=True, rewards={"score": 0.0}),
            ExampleValidation(index=2, ok=True, rewards={"bonus": 0.0}),
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0
    out = capsys.readouterr().out
    assert "NO training signal" in out and "GREEN baseline" not in out


def test_validate_json_includes_constant_warning(monkeypatch, capsys):
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"acc": 0.0}),
            ExampleValidation(index=1, ok=True, rewards={"acc": 0.0}),
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns(json=True)) == 0
    out = capsys.readouterr().out
    assert '"warnings"' in out and '"component": "acc"' in out


def test_constant_components_helper():
    # direct unit: ignores varying + single-value sets, flags the constant one.
    assert validate._constant_components(
        [{"a": 1.0, "b": 0.0}, {"a": 2.0, "b": 0.0}]
    ) == [("b", 0.0)]
    assert validate._constant_components([{"a": 0.0}]) == []  # <2 rollouts
    assert validate._constant_components([]) == []


def test_constant_total_helper():
    # constant total (incl. ragged keys + bool exclusion) -> the value; else None.
    assert validate._constant_total([{"a": 0.0}, {"a": 0.0}]) == 0.0
    assert validate._constant_total([{"a": 0.0}, {"b": 0.0}]) == 0.0  # ragged, both 0
    assert validate._constant_total([{"a": 1.0}, {"a": 2.0}]) is None  # varies
    assert validate._constant_total([{"a": 0.0}]) is None  # <2 rollouts
    assert (
        validate._constant_total([{"p": True}, {"p": False}]) == 0.0
    )  # bools excluded


# --- reward audit (--reward-audit) --------------------------------------


class _RagProject:
    """A project whose dataset carries gold, indexable by rollout index."""

    training_mode = "rl"
    env_class = type("E", (), {})
    train_dataset = [
        {"prompt": "where do I add the exception?", "ground_truth": "edit /etc/docker"},
        {"prompt": "second question", "ground_truth": "the second gold answer"},
    ]
    eval_dataset = []
    module = None
    from_file = True
    launch_config: dict = {}
    validate_config: dict = {}


def test_reward_audit_shows_components_gold_and_answers(monkeypatch, capsys):
    report = _report(
        examples=[
            ExampleValidation(
                index=0,
                ok=True,
                rewards={"answer_correctness": 1.0, "citation_recall": 0.3},
                messages=[{"role": "assistant", "content": "edit /etc/docker now"}],
            ),
            ExampleValidation(
                index=1,
                ok=True,
                rewards={"answer_correctness": 0.0, "citation_recall": 0.0},
                messages=[{"role": "assistant", "content": "no idea, sorry"}],
            ),
        ],
        group=None,
    )
    monkeypatch.setattr(validate, "load_project", lambda **k: _RagProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: report)
    assert validate._cmd_validate(_validate_ns(reward_audit=True)) == 0
    out = capsys.readouterr().out
    assert "reward audit" in out
    # per-component stats + the real question/gold/answer
    assert "answer_correctness" in out and "citation_recall" in out
    assert "gold: edit /etc/docker" in out
    assert "edit /etc/docker now" in out  # the model's captured answer


def test_reward_audit_implies_full_messages_capture(monkeypatch):
    # --reward-audit alone must ask validate_env to capture transcripts, else the
    # audit has no answers to show.
    captured: dict = {}

    def _capture(**k):
        captured.update(k)
        return _report(
            examples=[ExampleValidation(index=0, ok=True, rewards={"a": 1.0})],
            group=None,
        )

    monkeypatch.setattr(validate, "load_project", lambda **k: _RagProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _capture)
    validate._cmd_validate(_validate_ns(reward_audit=True))
    assert captured["full_messages"] is True


def test_reward_audit_json_carries_audit_and_gold(monkeypatch, capsys):
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"answer_correctness": 1.0}),
        ],
        group=None,
    )
    monkeypatch.setattr(validate, "load_project", lambda **k: _RagProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: report)
    assert validate._cmd_validate(_validate_ns(reward_audit=True, json=True)) == 0
    out = capsys.readouterr().out
    assert '"audit"' in out
    assert '"primary_component": "answer_correctness"' in out
    assert '"gold": "edit /etc/docker"' in out


def test_mirrors_correctness_flags_redundant_not_independent():
    # A component that is constant within each correctness stratum mirrors it;
    # one that varies within a stratum (recall) does not.
    corr = [1.0, 1.0, 0.0, 0.0]
    dup = [0.5, 0.5, 0.0, 0.0]  # 0.5 * correctness → redundant
    recall = [0.3, 0.1, 0.0, 0.0]  # varies within the correct stratum → real signal
    assert validate._mirrors_correctness(dup, corr) is True
    assert validate._mirrors_correctness(recall, corr) is False
    # correctness that never varies can't be a basis for the check
    assert validate._mirrors_correctness([0.5, 0.5], [1.0, 1.0]) is False


def test_audit_components_notes():
    ok_rewards = [
        {"answer_correctness": 1.0, "dup": 0.5, "recall": 0.3, "fmt": 0.1},
        {"answer_correctness": 1.0, "dup": 0.5, "recall": 0.1, "fmt": 0.1},
        {"answer_correctness": 0.0, "dup": 0.0, "recall": 0.0, "fmt": 0.1},
        {"answer_correctness": 0.0, "dup": 0.0, "recall": 0.0, "fmt": 0.1},
    ]
    rows, corr_key = validate._audit_components(ok_rewards, set())
    assert corr_key == "answer_correctness"
    notes = {r["component"]: r["note"] for r in rows}
    assert notes["answer_correctness"] == "primary (gate)"
    assert "mirrors the primary reward" in notes["dup"]
    assert "constant" in notes["fmt"]
    assert notes["recall"] == ""  # discriminates independently


def test_primary_reward_key_env_declared_then_heuristic():
    """The gate key is env-supplied first (any env-type), with the RAG name
    heuristic as fallback — so the audit isn't anchored on a literal 'correct' key."""
    pk = validate._primary_reward_key
    # RAG anchors via the heuristic with no declaration
    assert pk({"answer_correctness": 0, "citation_recall": 0}) == "answer_correctness"
    # judge dict, no *correct* key, no declaration → None (skip, not misfire)
    assert pk({"helpfulness": 0, "conciseness": 0}) is None
    # judge dict + env-declared gate → anchors on the declared component
    assert pk({"helpfulness": 0, "conciseness": 0}, "helpfulness") == "helpfulness"
    # a declaration that isn't a component falls back to the heuristic (→ None here)
    assert pk({"a": 0, "b": 0}, "missing") is None


def test_reward_audit_non_rag_dict_no_misfire_and_declared_anchors():
    """A non-RAG (judge-shaped) reward dict: no gate → no spurious 'mirrors' note;
    with an env-declared gate the redundancy check anchors correctly."""
    ok_rewards = [
        {"quality": 1.0, "dup": 0.5},
        {"quality": 1.0, "dup": 0.5},
        {"quality": 0.0, "dup": 0.0},
        {"quality": 0.0, "dup": 0.0},
    ]
    # no declaration + no *correct* key → no anchor, so nothing is flagged redundant
    rows, corr = validate._audit_components(ok_rewards, set())
    assert corr is None
    assert all("mirrors" not in r["note"] for r in rows)
    # declaring the gate lights up the redundancy check ('dup' mirrors 'quality')
    rows, corr = validate._audit_components(ok_rewards, set(), "quality")
    assert corr == "quality"
    notes = {r["component"]: r["note"] for r in rows}
    assert "mirrors the primary reward" in notes["dup"]


def test_reward_audit_json_uses_env_declared_primary(monkeypatch, capsys):
    """--reward-audit --json anchors on the env's PRIMARY_REWARD_KEY end-to-end."""
    import json as _json

    class _JudgeEnv:
        PRIMARY_REWARD_KEY = "quality"

    class _P:
        training_mode = "rl"
        env_class = _JudgeEnv
        train_dataset = [{"prompt": "x"}, {"prompt": "y"}]
        eval_dataset: list = []
        module = None
        from_file = True
        launch_config: dict = {}
        validate_config: dict = {}

    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"quality": 1.0, "dup": 0.2}),
            ExampleValidation(index=1, ok=True, rewards={"quality": 0.0, "dup": 0.2}),
        ],
        group=None,
    )
    monkeypatch.setattr(validate, "load_project", lambda **k: _P())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: report)
    assert validate._cmd_validate(_validate_ns(json=True, reward_audit=True)) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["audit"]["primary_component"] == "quality"


def test_audit_marks_group_components_na():
    ok_rewards = [{"acc": 1.0, "rank": 0.0}, {"acc": 0.0, "rank": 0.0}]
    rows, _ = validate._audit_components(ok_rewards, {"rank"})
    notes = {r["component"]: r["note"] for r in rows}
    assert "group-scored" in notes["rank"]  # not flagged "constant"


# --- review fixes: dataset-shape + config validation --------------------


def test_example_gold_reads_question_key():
    # RAG rows key the question under 'question' (+ 'answer' for gold); generic
    # rows use 'prompt'/'ground_truth'. Both must yield a question + gold.
    assert validate._example_gold({"question": "Q?", "answer": "A"}) == ("Q?", "A")
    assert validate._example_gold({"prompt": "P", "ground_truth": "G"}) == ("P", "G")


def test_row_question_and_gold_shared_helper():
    # The shared dataset-shape parser (used by runs + validate) — its own edge tests.
    from benchmax.cli._project import row_question_and_gold

    assert row_question_and_gold({"prompt": "P", "ground_truth": "G"}) == ("P", "G")
    assert row_question_and_gold({"question": "Q", "answer": "A"}) == ("Q", "A")
    # chat-list prompt → last user turn
    assert row_question_and_gold(
        {
            "prompt": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            "answer": "A",
        }
    ) == ("u", "A")
    # ground_truth wins over answer; a falsy-but-real gold (0) is preserved
    assert row_question_and_gold(
        {"question": "Q", "ground_truth": 0, "answer": "x"}
    ) == (
        "Q",
        0,
    )
    assert row_question_and_gold("not a dict") == (None, None)
    assert row_question_and_gold({}) == (None, None)


def test_validate_config_non_int_knob_fails_loudly(monkeypatch):
    # A str budget in VALIDATE_CONFIG must fail loudly here, not crash deep in the SDK.
    monkeypatch.setattr(
        validate, "load_project", lambda **k: _validate_config_project(max_turns="7")
    )
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: None)
    with pytest.raises(SystemExit, match="must be an int"):
        validate._cmd_validate(_validate_ns(max_turns=None))


def test_read_config_rejects_non_dict():
    import types

    from benchmax.cli._project import ProjectError, _read_config

    m = types.ModuleType("m")
    m.LAUNCH_CONFIG = [1, 2]  # not a dict → fail loudly
    with pytest.raises(ProjectError, match="must be a dict"):
        _read_config(m, "LAUNCH_CONFIG")
    assert _read_config(m, "MISSING") == {}  # absent → {}
    m.VALIDATE_CONFIG = None
    assert _read_config(m, "VALIDATE_CONFIG") == {}  # None (unset) → {}
    m.OK = {"a": 1}
    assert _read_config(m, "OK") == {"a": 1}


# --- inconsistent reward shape (ragged keys across examples) --------------


def test_inconsistent_components_helper():
    # 'cite' present in only 1 of 2 rollouts → ragged; 'acc' in both → not flagged.
    assert validate._inconsistent_components(
        [{"acc": 1.0, "cite": 0.3}, {"acc": 0.0}]
    ) == [("cite", 1)]
    assert (
        validate._inconsistent_components([{"a": 1.0}, {"a": 0.0}]) == []
    )  # consistent
    assert validate._inconsistent_components([{"a": 1.0}]) == []  # <2 rollouts


def test_validate_flags_inconsistent_reward_shape(monkeypatch, capsys):
    # A soft ⚠ (report still passes) when a component is missing from some rollouts.
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"acc": 1.0, "cite": 0.3}),
            ExampleValidation(index=1, ok=True, rewards={"acc": 0.0}),  # 'cite' missing
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0  # soft warning, not a failure
    out = capsys.readouterr().out
    assert "reward shape inconsistent" in out
    assert "'cite' in 1/2" in out


def test_validate_json_includes_inconsistent_shape_warning(monkeypatch, capsys):
    report = _report(
        examples=[
            ExampleValidation(index=0, ok=True, rewards={"acc": 1.0, "cite": 0.3}),
            ExampleValidation(index=1, ok=True, rewards={"acc": 0.0}),
        ],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns(json=True)) == 0
    out = capsys.readouterr().out
    assert '"inconsistent_reward_shape"' in out and '"component": "cite"' in out
    assert '"present": 1' in out


# --- sft mode (slice 5) --------------------------------------------------

_SFT_ROW = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "yo"}]}\n'
)


def _write_sft_project(tmp_path, *, train, eval=None):
    (tmp_path / "main.py").write_text('TRAINING_MODE = "sft"\n')
    (tmp_path / "train_dataset.jsonl").write_text(train)
    if eval is not None:
        (tmp_path / "eval_dataset.jsonl").write_text(eval)
    return tmp_path


def test_validate_sft_valid_dataset_exits_0(tmp_path, capsys):
    _write_sft_project(tmp_path, train=_SFT_ROW)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "sft" in out
    assert "train 1" in out
    assert "validate passed" in out


def test_validate_sft_malformed_json_shows_physical_line_numbers(tmp_path, capsys):
    # blank at physical line 1, malformed JSON at physical line 2, a valid row at
    # physical line 3 -- proves _load_jsonl (which drops blanks + reindexes) is
    # bypassed: load_sft_dataset's own physical-line count must survive intact.
    _write_sft_project(tmp_path, train="\nnot valid json\n" + _SFT_ROW)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 1
    out = capsys.readouterr().out
    assert "train_dataset.jsonl:2" in out
    assert "invalid JSON" in out
    assert "validate failed" in out


def test_validate_sft_json_output(tmp_path, capsys):
    import json as _json

    _write_sft_project(tmp_path, train=_SFT_ROW)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path), json=True)) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["train_row_count"] == 1
    # only the expected "no eval dataset provided" notice, no error issues
    assert [i["severity"] for i in payload["issues"]] == ["notice"]


def test_validate_sft_json_malformed_line_reports_physical_line(tmp_path, capsys):
    import json as _json

    _write_sft_project(tmp_path, train="\nnot valid json\n" + _SFT_ROW)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path), json=True)) == 1
    payload = _json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    issue = next(i for i in payload["issues"] if i["severity"] == "error")
    assert issue["physical_line"] == 2


def test_validate_sft_weight_notice_does_not_fail(tmp_path, capsys):
    weighted_row = (
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "content": "yo", "weight": 1}]}\n'
    )
    _write_sft_project(tmp_path, train=weighted_row)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "rows with weight         1" in out


def test_validate_sft_empty_train_fails(tmp_path, capsys):
    _write_sft_project(tmp_path, train="\n")
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 1
    out = capsys.readouterr().out
    assert "validate failed" in out


def test_validate_sft_uses_project_validate_config_max_seq_len(tmp_path, capsys):
    """`castform validate` (sft mode) must resolve max_seq_len from the project's
    VALIDATE_CONFIG the same way `_cmd_validate` resolves it for RL envs -- not
    silently validate with the library default (regression: `_cmd_validate_sft`
    used to call `validate_sft_dataset` with no kwargs at all, so a project's
    declared budget was never honored)."""
    (tmp_path / "main.py").write_text('TRAINING_MODE = "sft"\n')
    (tmp_path / "train_dataset.jsonl").write_text(_SFT_ROW)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "exceed max_seq_len" not in out  # default max_seq_len (8192) not tripped

    # Same tiny row, but the project now declares a max_seq_len far below it --
    # only honoring VALIDATE_CONFIG makes this row cross the budget.
    (tmp_path / "main.py").write_text(
        'TRAINING_MODE = "sft"\nVALIDATE_CONFIG = {"max_seq_len": 0}\n'
    )
    assert (
        validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 0
    )  # notice, not error
    out = capsys.readouterr().out
    assert "exceed max_seq_len" in out


def test_validate_sft_rejects_non_int_max_seq_len(tmp_path, capsys):
    """A malformed VALIDATE_CONFIG value (a string, not an int) must produce a
    clean configuration error -- not a raw TypeError raised deep inside
    validation. `_cmd_validate` is `@handle_errors`-wrapped, so the
    `SftConfigError` (a `RuntimeError`) surfaces as a clean stderr line."""
    (tmp_path / "main.py").write_text(
        'TRAINING_MODE = "sft"\nVALIDATE_CONFIG = {"max_seq_len": "100"}\n'
    )
    (tmp_path / "train_dataset.jsonl").write_text(_SFT_ROW)
    assert validate._cmd_validate(_validate_ns(dir=str(tmp_path))) == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "max_seq_len" in err


def test_reward_audit_shows_inconsistent_shape(monkeypatch, capsys):
    report = _report(
        examples=[
            ExampleValidation(
                index=0,
                ok=True,
                rewards={"answer_correctness": 1.0, "cite": 0.3},
                messages=[{"role": "assistant", "content": "a"}],
            ),
            ExampleValidation(
                index=1,
                ok=True,
                rewards={"answer_correctness": 0.0},  # 'cite' missing
                messages=[{"role": "assistant", "content": "b"}],
            ),
        ],
        group=None,
    )
    monkeypatch.setattr(validate, "load_project", lambda **k: _RagProject())
    monkeypatch.setattr("benchmax.platform.validation.validate_env", lambda **k: report)
    assert validate._cmd_validate(_validate_ns(reward_audit=True)) == 0
    out = capsys.readouterr().out
    assert "inconsistent reward shape" in out
    assert "cite: present in 1/2" in out
