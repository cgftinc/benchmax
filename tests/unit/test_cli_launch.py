"""Slice 1.5 offline: launcher-arg validation against the platform schema."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from benchmax.cli import launch
from benchmax.cli.launch import _build_launcher_args, _coerce_arg
from benchmax.platform.client import LaunchArgSpec


def _spec(name, type_, **kw):
    return LaunchArgSpec(
        name=name, label=name, type=type_, required=False, description="", **kw
    )


SPECS = [
    _spec("model", "string", enum=("qwen-4b", "qwen-35b")),
    _spec("max_rollout_len", "integer", min=128, max=8000, warn_above=4000),
    _spec("max_turns", "integer"),
]


def test_coerce_types():
    assert _coerce_arg(_spec("n", "integer"), "5") == 5
    assert _coerce_arg(_spec("x", "number"), "1.5") == 1.5
    assert _coerce_arg(_spec("b", "boolean"), "true") is True
    assert _coerce_arg(_spec("b", "boolean"), "no") is False
    assert _coerce_arg(_spec("s", "string"), "hi") == "hi"


def test_build_ok():
    out = _build_launcher_args(SPECS, ["model=qwen-4b", "max_rollout_len=2000"])
    assert out == {"model": "qwen-4b", "max_rollout_len": 2000}


def test_build_rejects_unknown_key():
    # max_response_len is the classic wrong knob — must be rejected, not silently sent.
    with pytest.raises(SystemExit, match="Unknown launch arg 'max_response_len'"):
        _build_launcher_args(SPECS, ["max_response_len=2000"])


def test_build_rejects_bad_enum():
    with pytest.raises(SystemExit, match="must be one of"):
        _build_launcher_args(SPECS, ["model=gpt-9"])


def test_build_rejects_out_of_range():
    with pytest.raises(SystemExit, match="above max"):
        _build_launcher_args(SPECS, ["max_rollout_len=99999"])
    with pytest.raises(SystemExit, match="below min"):
        _build_launcher_args(SPECS, ["max_rollout_len=1"])


def test_build_warns_above_soft_cap(capsys):
    _build_launcher_args(SPECS, ["max_rollout_len=6000"])
    assert "soft cap" in capsys.readouterr().err


def test_build_bad_pair():
    with pytest.raises(SystemExit, match="key=value"):
        _build_launcher_args(SPECS, ["model"])


# --- pip merge reaches BOTH launch sites (the B1 fix) -------------------------


@dataclasses.dataclass
class _Uploaded:
    env_blob_path: str = "envs/x"


class _SlotEnv:
    PIP_DEPENDENCIES = ["myorg-search>=2.0"]


class _FakeProject:
    env_class = _SlotEnv
    train_dataset = [{"prompt": "x"}]
    eval_dataset = [{"prompt": "y"}]
    module = None
    from_file = True
    launch_config: dict = {}
    validate_config: dict = {}


class _FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def list_launch_args(self):
        return SPECS

    def launch_training_run(self, **kw):
        return "run-123"


def _launch_ns(**over):
    base = dict(
        list_args=False,
        dir=".",
        run_file="run.py",
        module=None,
        env_class=None,
        train="train_dataset.jsonl",
        eval="eval_dataset.jsonl",
        env_arg=None,
        set=None,
        name=None,
        type="simple",
        yes=True,
        skip_validate=False,
        pip=["mydep"],
        provider="chroma",
        model=None,
        json=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_launch_merges_pip_into_both_sites(monkeypatch):
    # The merged deps (--pip + the env's PIP_DEPENDENCIES slot + --provider's SDK)
    # MUST reach the pre-flight validate_env AND the upload — else the pre-flight
    # validates with zero deps while the uploaded run has them (the B1 disagreement).
    captured: dict = {"validate": "unset", "upload": "unset"}

    def _fake_validate(**k):
        captured["validate"] = k.get("pip_dependencies")
        return type("R", (), {"ok": True})()

    def _fake_upload(**k):
        captured["upload"] = k.get("pip_dependencies")
        return _Uploaded()

    monkeypatch.setattr(launch, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr(launch, "TrainerClient", _FakeClient)
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _fake_validate)
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", _fake_upload
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")

    assert launch._cmd_launch(_launch_ns()) == 0
    merged = ["mydep", "myorg-search>=2.0", "chromadb>=1.0.0", "snowballstemmer>=2.2.0"]
    assert captured["validate"] == merged  # pre-flight got deps (was NOT passed before)
    assert captured["upload"] == merged


class _PlainEnv:  # no PIP_DEPENDENCIES slot
    pass


class _PlainProject(_FakeProject):
    env_class = _PlainEnv


def test_launch_plain_env_no_provider_both_sites_match(monkeypatch):
    # The COMMON real launch: no --provider, a plain env. Both sites must still get
    # the SAME value — the --pip list verbatim, and None when there are no deps.
    captured: dict = {"validate": "unset", "upload": "unset"}

    def _fake_validate(**k):
        captured["validate"] = k.get("pip_dependencies")
        return type("R", (), {"ok": True})()

    def _fake_upload(**k):
        captured["upload"] = k.get("pip_dependencies")
        return _Uploaded()

    monkeypatch.setattr(launch, "load_project", lambda **k: _PlainProject())
    monkeypatch.setattr(launch, "TrainerClient", _FakeClient)
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _fake_validate)
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", _fake_upload
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")

    assert launch._cmd_launch(_launch_ns(pip=["mydep"], provider=None)) == 0
    assert captured["validate"] == ["mydep"] and captured["upload"] == ["mydep"]

    assert launch._cmd_launch(_launch_ns(pip=None, provider=None)) == 0
    assert captured["validate"] is None and captured["upload"] is None


def test_launch_preflight_honors_set_max_turns(monkeypatch):
    # The pre-flight smoke must validate at the SAME turn budget the run will use
    # (from --set max_turns), so a multi-turn env isn't truncated and falsely flagged.
    captured: dict = {}

    def _fake_validate(**k):
        captured["max_turns"] = k.get("max_turns")
        return type("R", (), {"ok": True})()

    monkeypatch.setattr(launch, "load_project", lambda **k: _PlainProject())
    monkeypatch.setattr(launch, "TrainerClient", _FakeClient)
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _fake_validate)
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")

    assert launch._cmd_launch(_launch_ns(set=["max_turns=11"])) == 0
    assert captured["max_turns"] == 11
    # default when --set max_turns is omitted
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    assert captured["max_turns"] == 4


# --- LAUNCH_CONFIG: run.py bakes in launcher args + type -------------------


def _config_project(**cfg):
    class _P(_FakeProject):
        launch_config = cfg

    return _P()


class _CapturingClient(_FakeClient):
    captured: dict = {}

    def launch_training_run(self, **kw):
        _CapturingClient.captured = kw
        return "run-123"


def _patch_launch(monkeypatch, project):
    monkeypatch.setattr(launch, "load_project", lambda **k: project)
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr(
        "benchmax.platform.validation.validate_env",
        lambda **k: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")


def test_launch_config_supplies_launcher_args(monkeypatch):
    _patch_launch(monkeypatch, _config_project(max_turns=7, max_rollout_len=2000))
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    la = _CapturingClient.captured["launcher_args"]
    assert la["max_turns"] == 7 and la["max_rollout_len"] == 2000


def test_launch_cli_set_overrides_config(monkeypatch):
    _patch_launch(monkeypatch, _config_project(max_turns=7))
    assert launch._cmd_launch(_launch_ns(set=["max_turns=11"])) == 0
    assert _CapturingClient.captured["launcher_args"]["max_turns"] == 11


def test_launch_config_type_used_when_flag_omitted(monkeypatch):
    _patch_launch(monkeypatch, _config_project(type="simple-cpu"))
    assert launch._cmd_launch(_launch_ns(type=None, set=None)) == 0
    assert _CapturingClient.captured["training_run_type"] == "simple-cpu"


def test_launch_config_max_turns_reaches_preflight(monkeypatch):
    captured: dict = {}

    def _fake_validate(**k):
        captured["max_turns"] = k.get("max_turns")
        return type("R", (), {"ok": True})()

    monkeypatch.setattr(
        launch, "load_project", lambda **k: _config_project(max_turns=9)
    )
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _fake_validate)
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    assert captured["max_turns"] == 9  # preflight smoke-tests at the config budget


def test_launcher_args_from_config_unknown_key_warns_and_skips(capsys):
    from benchmax.cli.launch import _launcher_args_from_config

    out = _launcher_args_from_config(SPECS, {"max_turns": 5, "bogus_knob": 1})
    assert out == {"max_turns": 5}  # unknown skipped
    assert "unknown launch arg 'bogus_knob'" in capsys.readouterr().err


def test_launcher_args_from_config_out_of_range_fails():
    from benchmax.cli.launch import _launcher_args_from_config

    with pytest.raises(SystemExit, match="above max"):
        _launcher_args_from_config(SPECS, {"max_rollout_len": 99999})


def test_launcher_args_from_config_ignores_reserved_keys():
    from benchmax.cli.launch import _launcher_args_from_config

    # type/name/model are resolved separately, not sent as launcher args
    assert _launcher_args_from_config(SPECS, {"type": "simple", "max_turns": 4}) == {
        "max_turns": 4
    }
