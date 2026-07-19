"""Slice 1.5 offline: launcher-arg validation against the platform schema."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from castform.cli import launch
from castform.cli.launch import _build_launcher_args, _coerce_arg
from castform.platform.client import LaunchArgSpec


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
        run_file="main.py",
        module=None,
        env_class=None,
        train="train_dataset.jsonl",
        eval="eval_dataset.jsonl",
        env_arg=None,
        set=None,
        name=None,
        yes=True,
        pip=["mydep"],
        provider="chroma",
        json=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_launch_merges_pip_for_upload(monkeypatch):
    # The merged deps (--pip + env declarations + provider SDK) reach the upload.
    captured: dict = {"upload": "unset"}

    def _fake_upload(**k):
        captured["upload"] = k.get("pip_dependencies")
        return _Uploaded()

    monkeypatch.setattr(launch, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr(launch, "TrainerClient", _FakeClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", _fake_upload
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns()) == 0
    merged = ["mydep", "myorg-search>=2.0", "chromadb>=1.0.0", "snowballstemmer>=2.2.0"]
    assert captured["upload"] == merged


class _PlainEnv:  # no PIP_DEPENDENCIES slot
    pass


class _PlainProject(_FakeProject):
    env_class = _PlainEnv


def test_launch_plain_env_upload_dependencies(monkeypatch):
    # A plain env gets --pip verbatim, or None when no dependency is declared.
    captured: dict = {"upload": "unset"}

    def _fake_upload(**k):
        captured["upload"] = k.get("pip_dependencies")
        return _Uploaded()

    monkeypatch.setattr(launch, "load_project", lambda **k: _PlainProject())
    monkeypatch.setattr(launch, "TrainerClient", _FakeClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", _fake_upload
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns(pip=["mydep"], provider=None)) == 0
    assert captured["upload"] == ["mydep"]

    assert launch._cmd_launch(_launch_ns(pip=None, provider=None)) == 0
    assert captured["upload"] is None


def test_launch_set_max_turns_reaches_server(monkeypatch):
    monkeypatch.setattr(launch, "load_project", lambda **k: _PlainProject())
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns(set=["max_turns=11"])) == 0
    assert _CapturingClient.captured["launcher_args"]["max_turns"] == 11


# --- LAUNCH_CONFIG: main.py bakes in launcher args -------------------


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
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)


def test_launch_config_supplies_launcher_args(monkeypatch):
    _patch_launch(monkeypatch, _config_project(max_turns=7, max_rollout_len=2000))
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    la = _CapturingClient.captured["launcher_args"]
    assert la["max_turns"] == 7 and la["max_rollout_len"] == 2000


def test_launch_cli_set_overrides_config(monkeypatch):
    _patch_launch(monkeypatch, _config_project(max_turns=7))
    assert launch._cmd_launch(_launch_ns(set=["max_turns=11"])) == 0
    assert _CapturingClient.captured["launcher_args"]["max_turns"] == 11


def test_launch_config_model_is_training_arg(monkeypatch):
    """LAUNCH_CONFIG['model'] is sent to the training launcher."""
    monkeypatch.setattr(
        launch, "load_project", lambda **k: _config_project(model="qwen-4b")
    )
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    assert _CapturingClient.captured["launcher_args"]["model"] == "qwen-4b"


def test_launch_config_max_turns_reaches_server(monkeypatch):
    monkeypatch.setattr(
        launch, "load_project", lambda **k: _config_project(max_turns=9)
    )
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    assert _CapturingClient.captured["launcher_args"]["max_turns"] == 9


def test_launcher_args_from_config_unknown_key_warns_and_skips(capsys):
    from castform.cli.launch import _launcher_args_from_config

    out = _launcher_args_from_config(SPECS, {"max_turns": 5, "bogus_knob": 1})
    assert out == {"max_turns": 5}  # unknown skipped
    assert "unknown launch arg 'bogus_knob'" in capsys.readouterr().err


def test_launcher_args_from_config_out_of_range_fails():
    from castform.cli.launch import _launcher_args_from_config

    with pytest.raises(SystemExit, match="above max"):
        _launcher_args_from_config(SPECS, {"max_rollout_len": 99999})


def test_launcher_args_from_config_ignores_reserved_keys():
    from castform.cli.launch import _launcher_args_from_config

    # reserved keys ('name', 'type') are filtered from launcher args; 'model' is NOT
    # reserved — it flows through as the training arg (see the model-routing test above)
    assert _launcher_args_from_config(SPECS, {"type": "simple", "max_turns": 4}) == {
        "max_turns": 4
    }


# --- token-budget guard + in-repo manifest (Slice 6) -----------------------


def test_launch_warns_when_estimate_exceeds_budget(monkeypatch, capsys):
    """The pre-confirm truncation guard warns when the env's estimated rollout
    tokens exceed max_rollout_len (a truncated rollout is dropped from the loss)."""
    from benchmax.envs import BaseEnv

    class _BigEnv(BaseEnv):
        async def create_dataset(self, split, base_dir):
            raise NotImplementedError

        async def compute_reward(
            self, rollout_id, messages, example_args, *, termination_reason
        ):
            return {}

        def estimate_rollout_tokens(self):
            return 999_999  # >> the budget below

    class _BigProject(_FakeProject):
        env_class = _BigEnv
        launch_config = {"max_rollout_len": 2000, "max_turns": 4}

    _patch_launch(monkeypatch, _BigProject())
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    err = capsys.readouterr().err
    assert "EXCEEDS max_rollout_len" in err and "DROP from the loss" in err


def test_launch_writes_in_repo_manifest(monkeypatch, tmp_path):
    """After a successful launch, an in-repo manifest records the env + dataset
    hashes, row counts, launcher args, and run id."""
    import json as _json

    (tmp_path / "main.py").write_text("# env definition\n")
    monkeypatch.setattr(launch, "load_project", lambda **k: _FakeProject())
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")

    assert launch._cmd_launch(_launch_ns(dir=str(tmp_path), json=True)) == 0
    manifest = tmp_path / ".castform" / "runs" / "run-123.json"
    assert manifest.exists()
    data = _json.loads(manifest.read_text())
    assert data["run_id"] == "run-123"
    assert data["train_rows"] == 1 and data["eval_rows"] == 1
    assert data["env_sha256"] is not None  # main.py was hashed
    assert data["train_sha256"] and data["eval_sha256"]
    assert "launcher_args" in data


def test_estimate_rollout_tokens_prefers_env_then_falls_back():
    from castform.cli.launch import _estimate_rollout_tokens

    # no env estimate + a max_turns → coarse per-turn fallback
    est, src = _estimate_rollout_tokens(_SlotEnv, {}, {"max_turns": 5})
    assert est == 5 * launch._GENERIC_TOKENS_PER_TURN and "per-turn" in src
    # no env estimate + no max_turns → no estimate at all
    assert _estimate_rollout_tokens(_SlotEnv, {}, {}) == (None, "")


def test_launch_guard_uses_schema_default_when_budget_omitted(monkeypatch, capsys):
    """When max_rollout_len is omitted, the guard must fall back to the schema
    default (the effective view), not silently skip — so a big env still warns."""
    from benchmax.envs import BaseEnv

    class _Big(BaseEnv):
        async def create_dataset(self, split, base_dir):
            raise NotImplementedError

        async def compute_reward(
            self, rollout_id, messages, example_args, *, termination_reason
        ):
            return {}

        def estimate_rollout_tokens(self):
            return 999_999

    class _BigProject(_FakeProject):
        env_class = _Big
        launch_config: dict = {}  # NO max_rollout_len set

    specs_with_defaults = [
        _spec("max_rollout_len", "integer", min=128, max=100000, default=4096),
        _spec("max_turns", "integer", default=4),
    ]

    class _DefaultsClient(_CapturingClient):
        def list_launch_args(self):
            return specs_with_defaults

    monkeypatch.setattr(launch, "load_project", lambda **k: _BigProject())
    monkeypatch.setattr(launch, "TrainerClient", _DefaultsClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    assert "EXCEEDS max_rollout_len 4096" in capsys.readouterr().err


def test_launch_manifest_records_module_source_for_module_launch(monkeypatch, tmp_path):
    """A --module launch records the module path (not main.py) and no env_sha256."""
    import json as _json

    class _ModuleProject(_FakeProject):
        from_file = False  # loaded from an importable module, not a file

    monkeypatch.setattr(launch, "load_project", lambda **k: _ModuleProject())
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr(
        "castform.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")

    assert (
        launch._cmd_launch(
            _launch_ns(dir=str(tmp_path), module="myorg.envs.search", json=True)
        )
        == 0
    )
    data = _json.loads((tmp_path / ".castform" / "runs" / "run-123.json").read_text())
    assert data["env_from_file"] is False
    assert data["env_source"] == "myorg.envs.search"
    assert data["env_sha256"] is None


def test_launch_help_hides_training_run_type():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    launch.register(subparsers)

    help_text = subparsers.choices["launch"].format_help()
    assert "--type" not in help_text
    assert "Training run type" not in help_text
    assert "--trainer-ref" not in help_text


def test_launch_rejects_training_run_type_arg():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    launch.register(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "--type", "simple-cpu"])


def test_launch_rejects_trainer_ref_arg():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    launch.register(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "--trainer-ref", "main"])
