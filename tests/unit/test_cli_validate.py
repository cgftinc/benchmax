"""Slice 1.4 CLI: project loading + validate report rendering. Offline."""

from __future__ import annotations

import argparse

import pytest

from benchmax.cli import validate
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
    p = tmp_path / "run.py"
    p.write_text(body)
    return p


def test_discover_single_env(tmp_path):
    mod = _load_module_from_file(_write_run(tmp_path, "MyEnv"))
    assert discover_env_class(mod).__name__ == "MyEnv"


def test_discover_no_env_raises(tmp_path):
    p = tmp_path / "run.py"
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
    env_class = type("E", (), {})
    train_dataset = [{"prompt": "x"}]
    eval_dataset = []
    module = None
    from_file = True


def _validate_ns(**over) -> argparse.Namespace:
    base = dict(
        dir=".",
        run_file="run.py",
        module=None,
        env_class=None,
        train="train_dataset.jsonl",
        eval="eval_dataset.jsonl",
        env_arg=None,
        model=None,
        examples=2,
        group_samples=2,
        local_only=False,
        verbose=False,
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


def test_env_arg_parsing():
    assert validate._parse_env_args(["a=1", "b=hi", "c=true"]) == {
        "a": 1,
        "b": "hi",
        "c": True,
    }


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


def test_validate_constant_needs_two_rollouts(monkeypatch, capsys):
    # A single rollout can't establish variance — don't warn.
    report = _report(
        examples=[ExampleValidation(index=0, ok=True, rewards={"acc": 0.0})],
        group=None,
    )
    _patch(monkeypatch, report)
    assert validate._cmd_validate(_validate_ns()) == 0
    assert "never vary" not in capsys.readouterr().out


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
