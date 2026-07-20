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
    training_mode = "rl"
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
        skip_validate=False,
        pip=["mydep"],
        provider="chroma",
        model=None,
        allow_experimental_weights=False,
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
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

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
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

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
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns(set=["max_turns=11"])) == 0
    assert captured["max_turns"] == 11
    # default when --set max_turns is omitted
    assert launch._cmd_launch(_launch_ns(set=None)) == 0
    assert captured["max_turns"] == 4


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
        "benchmax.platform.validation.validate_env",
        lambda **k: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
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


def test_launch_config_model_is_training_arg_not_preflight(monkeypatch):
    """LAUNCH_CONFIG['model'] is the TRAINING model → sent to the server as a launcher
    arg; the pre-flight validate model comes from VALIDATE_CONFIG/--model, not from
    LAUNCH_CONFIG (regression: model was reserved out + reused as the validate model)."""
    seen: dict = {}

    def _fake_validate(**k):
        seen["llm_model"] = k.get("llm_model")
        return type("R", (), {"ok": True})()

    monkeypatch.setattr(
        launch, "load_project", lambda **k: _config_project(model="qwen-4b")
    )
    monkeypatch.setattr(launch, "TrainerClient", _CapturingClient)
    monkeypatch.setattr("benchmax.platform.validation.validate_env", _fake_validate)
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)

    assert launch._cmd_launch(_launch_ns(model=None, set=None)) == 0
    # training model reaches the server as a launcher arg
    assert _CapturingClient.captured["launcher_args"]["model"] == "qwen-4b"
    # pre-flight did NOT borrow the training model (no --model, empty VALIDATE_CONFIG)
    assert seen["llm_model"] is None


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
    monkeypatch.setattr(launch, "_write_run_manifest", lambda **k: None)
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

    # reserved keys ('name', 'type') are filtered from launcher args; 'model' is NOT
    # reserved — it flows through as the training arg (see the model-routing test above)
    assert _launcher_args_from_config(SPECS, {"type": "simple", "max_turns": 4}) == {
        "max_turns": 4
    }


# --- token-budget guard + in-repo manifest (Slice 6) -----------------------


def test_launch_warns_when_estimate_exceeds_budget(monkeypatch, capsys):
    """The pre-confirm truncation guard warns when the env's estimated rollout
    tokens exceed max_rollout_len (a truncated rollout is dropped from the loss)."""
    from benchmax.envs.base_env import BaseEnv

    class _BigEnv(BaseEnv):
        async def list_tools(self):
            return []

        async def run_tool(self, rollout_id, tool_name, **k):
            return ""

        async def compute_reward(self, rollout_id, messages, task, **k):
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
        "benchmax.platform.validation.validate_env",
        lambda **k: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
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
    from benchmax.cli.launch import _estimate_rollout_tokens

    # no env estimate + a max_turns → coarse per-turn fallback
    est, src = _estimate_rollout_tokens(_SlotEnv, {}, {"max_turns": 5})
    assert est == 5 * launch._GENERIC_TOKENS_PER_TURN and "per-turn" in src
    # no env estimate + no max_turns → no estimate at all
    assert _estimate_rollout_tokens(_SlotEnv, {}, {}) == (None, "")


def test_launch_guard_uses_schema_default_when_budget_omitted(monkeypatch, capsys):
    """When max_rollout_len is omitted, the guard must fall back to the schema
    default (the effective view), not silently skip — so a big env still warns."""
    from benchmax.envs.base_env import BaseEnv

    class _Big(BaseEnv):
        async def list_tools(self):
            return []

        async def run_tool(self, rollout_id, tool_name, **k):
            return ""

        async def compute_reward(self, rollout_id, messages, task, **k):
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
        "benchmax.platform.validation.validate_env",
        lambda **k: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
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
        "benchmax.platform.validation.validate_env",
        lambda **k: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr(
        "benchmax.platform.training_run.upload_training_run", lambda **k: _Uploaded()
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


# --- sft mode (slice 5) -----------------------------------------------------

_SFT_ROW = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "yo"}]}\n'
)
_WEIGHTED_ROW = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "yo", "weight": 1}]}\n'
)


def _write_sft_project(tmp_path, *, train=_SFT_ROW, eval=_SFT_ROW):
    (tmp_path / "main.py").write_text('TRAINING_MODE = "sft"\n')
    (tmp_path / "train_dataset.jsonl").write_text(train)
    (tmp_path / "eval_dataset.jsonl").write_text(eval)
    return tmp_path


def _sft_launch_ns(tmp_path, **over):
    base = dict(
        list_args=False,
        dir=str(tmp_path),
        run_file="main.py",
        module=None,
        env_class=None,
        train="train_dataset.jsonl",
        eval="eval_dataset.jsonl",
        env_arg=None,
        set=None,
        name=None,
        yes=True,
        skip_validate=False,
        pip=None,
        provider=None,
        model=None,
        allow_experimental_weights=False,
        json=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _fake_uploaded_sft_run(**kw):
    from benchmax.platform.training_run import UploadedSftRun

    return UploadedSftRun(
        train_dataset_path="blob://train", eval_dataset_path="blob://eval"
    )


class _SftCapturingClient:
    captured_launch: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def list_launch_args(self):
        return SPECS

    def launch_sft_run(self, **kw):
        _SftCapturingClient.captured_launch = kw
        return "sft-run-1"


def _patch_sft_launch(monkeypatch, *, upload, launch_supported):
    monkeypatch.setattr(launch, "TrainerClient", _SftCapturingClient)
    monkeypatch.setattr("benchmax.platform.training_run.upload_sft_run", upload)
    monkeypatch.setattr(
        "benchmax.platform.client.SFT_LAUNCH_SUPPORTED", launch_supported
    )
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")


def test_launch_sft_not_supported_blocks_before_upload(monkeypatch, tmp_path, capsys):
    upload_calls: list = []

    def _fake_upload(**kw):
        upload_calls.append(kw)
        return _fake_uploaded_sft_run(**kw)

    _write_sft_project(tmp_path)
    _patch_sft_launch(monkeypatch, upload=_fake_upload, launch_supported=False)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path)) == 1
    assert upload_calls == []  # no orphaned storage artifacts
    err = capsys.readouterr().err
    assert "does not accept env-less sft runs yet" in err


def test_launch_sft_supported_uploads_then_launches_in_order(monkeypatch, tmp_path):
    order: list[str] = []

    def _fake_upload(**kw):
        order.append("upload")
        return _fake_uploaded_sft_run(**kw)

    class _OrderedClient(_SftCapturingClient):
        def launch_sft_run(self, **kw):
            order.append("launch")
            return super().launch_sft_run(**kw)

    _write_sft_project(tmp_path)
    monkeypatch.setattr(launch, "TrainerClient", _OrderedClient)
    monkeypatch.setattr("benchmax.platform.training_run.upload_sft_run", _fake_upload)
    monkeypatch.setattr("benchmax.platform.client.SFT_LAUNCH_SUPPORTED", True)
    monkeypatch.setattr(launch.config, "web_app_url", lambda: "http://x")

    assert launch._cmd_launch(_sft_launch_ns(tmp_path)) == 0
    assert order == ["upload", "launch"]
    assert _OrderedClient.captured_launch["train_dataset_path"] == "blob://train"
    assert _OrderedClient.captured_launch["eval_dataset_path"] == "blob://eval"


def test_launch_sft_weight_bearing_blocks_before_upload_without_override(
    monkeypatch, tmp_path, capsys
):
    upload_calls: list = []

    def _fake_upload(**kw):
        upload_calls.append(kw)
        return _fake_uploaded_sft_run(**kw)

    _write_sft_project(tmp_path, train=_WEIGHTED_ROW)
    _patch_sft_launch(monkeypatch, upload=_fake_upload, launch_supported=True)

    assert (
        launch._cmd_launch(_sft_launch_ns(tmp_path, allow_experimental_weights=False))
        == 1
    )
    assert upload_calls == []
    err = capsys.readouterr().err
    assert "masking" in err
    assert "--allow-experimental-weights" in err


def test_launch_sft_weight_bearing_proceeds_with_override(monkeypatch, tmp_path):
    upload_calls: list = []

    def _fake_upload(**kw):
        upload_calls.append(kw)
        return _fake_uploaded_sft_run(**kw)

    _write_sft_project(tmp_path, train=_WEIGHTED_ROW)
    _patch_sft_launch(monkeypatch, upload=_fake_upload, launch_supported=True)

    assert (
        launch._cmd_launch(_sft_launch_ns(tmp_path, allow_experimental_weights=True))
        == 0
    )
    assert len(upload_calls) == 1


def test_launch_sft_invalid_dataset_blocks_before_upload(monkeypatch, tmp_path):
    upload_calls: list = []

    def _fake_upload(**kw):
        upload_calls.append(kw)
        return _fake_uploaded_sft_run(**kw)

    _write_sft_project(tmp_path, train="\n")  # empty train -> not ok
    _patch_sft_launch(monkeypatch, upload=_fake_upload, launch_supported=True)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path)) == 1
    assert upload_calls == []


def test_launch_sft_weight_bearing_proceeds_with_launch_config_override(
    monkeypatch, tmp_path
):
    """LAUNCH_CONFIG['allow_experimental_weights'] must clear the weight gate the
    same way --allow-experimental-weights does -- the CLI flag and the config key
    are equivalent overrides (mirrors sft_main.py's own launch(), which already
    checks LAUNCH_CONFIG.get("allow_experimental_weights")). Regression:
    `_cmd_launch_sft` used to check only `args.allow_experimental_weights`."""
    upload_calls: list = []

    def _fake_upload(**kw):
        upload_calls.append(kw)
        return _fake_uploaded_sft_run(**kw)

    (tmp_path / "main.py").write_text(
        'TRAINING_MODE = "sft"\nLAUNCH_CONFIG = {"allow_experimental_weights": True}\n'
    )
    (tmp_path / "train_dataset.jsonl").write_text(_WEIGHTED_ROW)
    (tmp_path / "eval_dataset.jsonl").write_text(_WEIGHTED_ROW)
    _patch_sft_launch(monkeypatch, upload=_fake_upload, launch_supported=True)

    # the CLI flag is left at its default False -- only the project config opts in
    assert (
        launch._cmd_launch(_sft_launch_ns(tmp_path, allow_experimental_weights=False))
        == 0
    )
    assert len(upload_calls) == 1
    # the key is consumed client-side by the weight gate; it must never reach the
    # server as a launcher arg (regression: _LAUNCH_CONFIG_RESERVED omitted it)
    assert _SftCapturingClient.captured_launch["launcher_args"] is None


def test_launch_sft_uses_project_validate_config(monkeypatch, tmp_path):
    """`castform launch` (sft mode) must resolve max_seq_len/max_row_bytes from
    the project's VALIDATE_CONFIG, same as `castform validate` -- regression:
    `_cmd_launch_sft` called `validate_sft_dataset` with no kwargs at all, so
    launch never honored the project's declared budget."""
    from benchmax.sft.validate import validate_sft_dataset as real_validate_sft_dataset

    captured: dict = {}

    def _spy(*a, **kw):
        captured.update(kw)
        return real_validate_sft_dataset(*a, **kw)

    (tmp_path / "main.py").write_text(
        'TRAINING_MODE = "sft"\nVALIDATE_CONFIG = {"max_seq_len": 5}\n'
    )
    (tmp_path / "train_dataset.jsonl").write_text(_SFT_ROW)
    (tmp_path / "eval_dataset.jsonl").write_text(_SFT_ROW)
    monkeypatch.setattr("benchmax.sft.validate_sft_dataset", _spy)
    _patch_sft_launch(monkeypatch, upload=_fake_uploaded_sft_run, launch_supported=True)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path)) == 0
    assert captured["max_seq_len"] == 5


def test_launch_sft_validate_config_max_seq_len_rejects_non_int(monkeypatch, tmp_path):
    """A malformed VALIDATE_CONFIG value (a string instead of an int) must raise a
    clean configuration error -- not a raw TypeError deep inside validation."""
    (tmp_path / "main.py").write_text(
        'TRAINING_MODE = "sft"\nVALIDATE_CONFIG = {"max_seq_len": "100"}\n'
    )
    (tmp_path / "train_dataset.jsonl").write_text(_SFT_ROW)
    (tmp_path / "eval_dataset.jsonl").write_text(_SFT_ROW)
    _patch_sft_launch(monkeypatch, upload=_fake_uploaded_sft_run, launch_supported=True)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path)) == 1


def test_launch_sft_allow_experimental_weights_rejects_non_bool_string(
    monkeypatch, tmp_path
):
    """`LAUNCH_CONFIG["allow_experimental_weights"] = "false"` must be rejected
    outright -- the string "false" is truthy in Python, so resolving it by
    truthiness would silently CLEAR the experimental-weight safety gate (the
    opposite of what the user almost certainly meant)."""
    upload_calls: list = []

    def _fake_upload(**kw):
        upload_calls.append(kw)
        return _fake_uploaded_sft_run(**kw)

    (tmp_path / "main.py").write_text(
        'TRAINING_MODE = "sft"\nLAUNCH_CONFIG = {"allow_experimental_weights": "false"}\n'
    )
    (tmp_path / "train_dataset.jsonl").write_text(_WEIGHTED_ROW)
    (tmp_path / "eval_dataset.jsonl").write_text(_WEIGHTED_ROW)
    _patch_sft_launch(monkeypatch, upload=_fake_upload, launch_supported=True)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path)) == 1
    assert not upload_calls  # rejected before upload, not silently treated as False


def test_launch_sft_launcher_args_flow_through(monkeypatch, tmp_path):
    _write_sft_project(tmp_path)
    _patch_sft_launch(monkeypatch, upload=_fake_uploaded_sft_run, launch_supported=True)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path, set=["max_turns=11"])) == 0
    assert _SftCapturingClient.captured_launch["launcher_args"]["max_turns"] == 11


def test_launch_sft_json_output(monkeypatch, tmp_path, capsys):
    # Progress lines ("Validating…", "Uploading…") share stdout with the JSON
    # payload (same pre-existing convention as the rl launch path) — check the
    # payload landed rather than parsing the whole stream as one JSON document.
    _write_sft_project(tmp_path)
    _patch_sft_launch(monkeypatch, upload=_fake_uploaded_sft_run, launch_supported=True)

    assert launch._cmd_launch(_sft_launch_ns(tmp_path, json=True)) == 0
    out = capsys.readouterr().out
    assert '"run_id": "sft-run-1"' in out
