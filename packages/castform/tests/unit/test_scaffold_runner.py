"""The scaffold seed's `main.py` staged runner: argparse dispatch + the
upload-once path (data → bundle → upload → validate → launch), with the SDK
monkeypatched (no network)."""

from __future__ import annotations

import asyncio
import dataclasses
import types
from pathlib import Path

import castform.cli.scaffold as scaffold_pkg
import pytest

from ._scaffold import discover_env_class, load_module

_SCAFFOLD_DIR = Path(scaffold_pkg.__file__).parent
_SEED = _SCAFFOLD_DIR / "generic_main.py"


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Load a scaffold seed as a module (its `__main__` block does not fire)."""
    monkeypatch.chdir(tmp_path)
    return load_module(_SEED)


def _fake_report(ok: bool = True):
    outcome = types.SimpleNamespace(
        rewards={"score": 1.0},
        termination_reason="finished",
        error=None,
    )
    return types.SimpleNamespace(
        ok=ok,
        local={"validate-0": outcome, "validate-1": outcome},
        remote=None,
        local_errors={},
        remote_errors={},
    )


# ── import safety: loading defines the stages, runs none of them ────────────────


def test_import_defines_stages_without_running(mod, tmp_path, monkeypatch):
    """Loading a seed defines the runner but executes NO stage (the `__main__`
    block is import-safe)."""
    assert all(hasattr(mod, n) for n in ("main", "generate_data", "validate", "launch"))
    assert not (tmp_path / "train.jsonl").exists()


def test_constructor_args_follow_the_example_shape(mod):
    assert mod._constructor_args(types.SimpleNamespace()) == {}


def test_launch_config_names_the_training_model(mod):
    assert mod.LAUNCH_CONFIG["model"] == "Qwen/Qwen3.5-4B"


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"prompt": "", "ground_truth": "Paris"},
        {"prompt": 123, "ground_truth": "Paris"},
        {"prompt": "Capital of France?"},
        {"prompt": "Capital of France?", "ground_truth": ""},
        {"prompt": "Capital of France?", "ground_truth": 123},
    ],
)
def test_row_converter_rejects_malformed_seed_rows(mod, row):
    with pytest.raises((TypeError, ValueError)):
        mod.CustomEnv()._example_from_row(row)


def test_row_converter_owns_reserved_prompt_messages(mod):
    example = mod.CustomEnv()._example_from_row(
        {
            "prompt": "Capital of France?",
            "ground_truth": "Paris",
            "prompt_messages": [{"role": "user", "content": "override"}],
        }
    )

    assert example.payload["prompt_messages"][-1]["content"] == "Capital of France?"


def test_dataset_path_uses_the_safe_resolver(mod, tmp_path, monkeypatch):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"prompt":"question","ground_truth":"answer"}\n')
    calls: list[tuple[Path, str]] = []

    def fake_resolve(base_dir, relative_path):
        calls.append((base_dir, relative_path))
        return data_path

    monkeypatch.setattr(mod, "resolve_dataset_path", fake_resolve)
    dataset = asyncio.run(mod.CustomEnv().create_dataset("train", tmp_path))

    assert len(dataset) == 1
    assert calls == [(tmp_path, "train.jsonl")]


def test_generate_data_preserves_an_existing_split_without_force(mod, tmp_path):
    train_path = tmp_path / mod.TRAIN_FILE
    train_path.write_text("curated train data\n")

    assert mod.generate_data(force=False)

    assert train_path.read_text() == "curated train data\n"
    assert (tmp_path / mod.EVAL_FILE).exists()


# ── argparse dispatch: the staged upload-once flow ──────────────────────────────


def _patch_staged_recorders(mod, monkeypatch, *, report_ok: bool = True):
    calls: list = []
    bundle = object()
    uploaded = types.SimpleNamespace(
        env_cls_path="envs/test/env-cls.pkl",
        env_metadata_path="envs/test/env-metadata.json",
        dataset_path=None,
    )
    monkeypatch.setattr(mod, "ensure_session", lambda *a, **k: calls.append(("session",)))

    def _data(**k):
        calls.append(("data", k))
        return True

    def _dump_bundle(env_class, **k):
        calls.append(("bundle", env_class))
        return bundle

    def _upload_assets(**k):
        calls.append(("upload", k))
        return uploaded

    def _validate(env, received):
        calls.append(("validate", received))
        return _fake_report(ok=report_ok)

    def _launch(received, **k):
        calls.append(("launch", received, k))
        return "run-x"

    monkeypatch.setattr(mod, "generate_data", _data)
    monkeypatch.setattr(mod, "dump_bundle", _dump_bundle)
    monkeypatch.setattr(mod, "upload_assets", _upload_assets)
    monkeypatch.setattr(mod, "validate", _validate)
    monkeypatch.setattr(mod, "launch", _launch)
    monkeypatch.setattr(mod, "_load_jsonl", lambda path: [{"row": 1}])
    return calls, bundle, uploaded


def test_dispatch_data_only(mod, monkeypatch):
    calls, _, _ = _patch_staged_recorders(mod, monkeypatch)
    assert mod.main(["data", "--force"]) == 0
    assert calls == [("data", {"force": True})]


def test_data_stage_does_not_require_login(mod, monkeypatch):
    monkeypatch.setattr(mod, "generate_data", lambda **kwargs: True)
    monkeypatch.setattr(
        mod,
        "ensure_session",
        lambda: (_ for _ in ()).throw(AssertionError("data requested login")),
    )
    assert mod.main(["data"]) == 0


def test_bare_and_validate_run_the_upload_once_path_and_stop(mod, monkeypatch):
    """Bare `python main.py` (and `validate`) run data → bundle → upload →
    validate on the SAME uploaded assets, and never launch (GPU spend)."""
    env_class = discover_env_class(mod)
    for argv in ([], ["validate"]):
        calls, bundle, uploaded = _patch_staged_recorders(mod, monkeypatch)
        assert mod.main(argv) == 0
        assert [c[0] for c in calls] == ["data", "session", "bundle", "upload", "validate"]
        assert calls[2][1] is env_class
        assert calls[3][1]["bundle"] is bundle
        assert calls[4][1] is uploaded


def test_dispatch_launch_reuses_validated_assets_and_passes_yes(mod, monkeypatch):
    for argv, assume_yes in ((["launch"], False), (["launch", "-y"], True)):
        calls, _, uploaded = _patch_staged_recorders(mod, monkeypatch)
        assert mod.main(argv) == 0
        assert calls[-1] == ("launch", uploaded, {"assume_yes": assume_yes})


def test_main_exit_1_on_validate_failure_and_gates_launch(mod, monkeypatch):
    """A failed validation exits non-zero and blocks the launch stage."""
    calls, _, _ = _patch_staged_recorders(mod, monkeypatch, report_ok=False)
    assert mod.main(["launch"]) == 1
    assert not any(c[0] == "launch" for c in calls)
    assert mod.main([]) == 1


def test_main_exit_1_when_launch_aborted(mod, monkeypatch):
    _patch_staged_recorders(mod, monkeypatch)
    monkeypatch.setattr(mod, "launch", lambda *a, **k: None)
    assert mod.main(["launch"]) == 1


# ── validate stage: script calls the public group-native function ───────────────


def test_validate_passes_uploaded_assets_and_config(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_class = discover_env_class(mod)
    captured: dict = {}
    remote_assets = object()

    async def fake_validate_environment(env, **kw):
        captured["env"] = env
        captured.update(kw)
        return _fake_report(ok=True)

    monkeypatch.setattr(mod, "validate_environment", fake_validate_environment)
    report = mod.validate(
        env_class(**mod._constructor_args(types.SimpleNamespace())), remote_assets
    )

    assert report.ok
    assert isinstance(captured["env"], env_class)
    assert captured["split"] == "train"
    assert captured["base_dir"] == Path(".")
    assert captured["remote_assets"] is remote_assets
    assert captured["model"] == str(mod.VALIDATE_CONFIG["model"])
    assert captured["max_context_tokens"] == 2048
    assert captured["local_timeout_seconds"] == 120.0


def test_validate_surfaces_public_dataset_materialization_failure(mod, monkeypatch):
    async def fail_validation(*args, **kwargs):
        raise RuntimeError("dataset materialization failed")

    monkeypatch.setattr(mod, "validate_environment", fail_validation)
    env_class = discover_env_class(mod)

    with pytest.raises(RuntimeError, match="dataset materialization failed"):
        mod.validate(env_class(**mod._constructor_args(types.SimpleNamespace())), None)


def test_scorecard_marks_error_settlements_and_shows_messages(mod, capsys):
    good = types.SimpleNamespace(
        rewards={"correct": 1.0},
        termination_reason="finished",
        error=None,
    )
    masked = types.SimpleNamespace(
        rewards={},
        termination_reason="max_turns_exceeded",
        error="KeyError: 'ground_truth'",
    )
    bad = types.SimpleNamespace(
        rewards={"correct": 0.0},
        termination_reason="judge_error",
        error=None,
    )
    report = types.SimpleNamespace(
        ok=False,
        local={"validate-0": good, "validate-1": masked, "validate-2": bad},
        remote=None,
        local_errors={},
        remote_errors={},
    )

    mod._print_validation(report)

    output = capsys.readouterr().out
    assert "✅ local validate-0: finished" in output
    assert "error=KeyError: 'ground_truth'" in output
    assert "❌ local validate-2: judge_error" in output
    assert "❌ validation failed" in output


def test_scorecard_surfaces_static_and_runtime_contract_warnings(mod, capsys):
    outcome = types.SimpleNamespace(
        rewards={"correct": 1.0},
        termination_reason="finished",
        error=None,
    )
    report = types.SimpleNamespace(
        ok=True,
        local={"validate-0": outcome},
        remote=None,
        static_warnings={"agent.kwargs.max_tokens": "output cap is trainer-clamped"},
        local_warnings={
            "validate-0": ["max_tokens requested 4096 but the effective output cap was 1024"]
        },
        remote_warnings={},
        local_errors={},
        remote_errors={},
    )

    mod._print_validation(report)

    output = capsys.readouterr().out
    assert "agent.kwargs.max_tokens" in output
    assert "effective output cap was 1024" in output
    assert "✅ validation passed" in output


@pytest.mark.parametrize(
    "relative_path",
    [
        "CLAUDE.md",
        "STARTER.md",
        "skills/design-environment/SKILL.md",
    ],
)
def test_scaffold_design_guidance_documents_model_parameter_ownership(
    relative_path: str,
) -> None:
    guidance = (_SCAFFOLD_DIR / relative_path).read_text().lower()

    assert "max_tokens" in guidance
    assert "max_completion_tokens" in guidance
    assert "temperature" in guidance
    assert "top_p" in guidance
    assert "warning" in guidance


@pytest.mark.parametrize(
    "relative_path",
    [
        "skills/verify-environment/SKILL.md",
        "skills/launch-run/SKILL.md",
    ],
)
def test_scaffold_validation_guidance_requires_training_contract_checks(
    relative_path: str,
) -> None:
    guidance = (_SCAFFOLD_DIR / relative_path).read_text().lower()

    assert "sampling" in guidance
    assert "history" in guidance
    assert "validate_environment" in guidance
    assert "do not launch" in guidance or "never launch" in guidance


def test_launch_skill_uses_explicit_constructor_args_not_legacy_global() -> None:
    guidance = (_SCAFFOLD_DIR / "skills/launch-run/SKILL.md").read_text()

    assert "constructor_args=constructor_args" in guidance
    assert "constructor_args=ENV_ARGS" not in guidance


# ── launch stage: [y/N] confirm and the asdict spread ───────────────────────────

_Uploaded = dataclasses.make_dataclass(
    "Uploaded",
    ["env_cls_path", "env_metadata_path", "dataset_path"],
)


def _patch_launch_client(mod, monkeypatch, launched: dict):
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def launch_training_run(self, **kw):
            launched.update(kw)
            return "run-123"

    monkeypatch.setattr(mod, "TrainerClient", lambda: FakeClient())


def test_launch_declined_at_confirm(mod, monkeypatch):
    launched: dict = {}
    _patch_launch_client(mod, monkeypatch, launched)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert mod.launch(_Uploaded("e", "m", "d")) is None
    assert not launched  # aborted before spending credits


def test_launch_confirmed_spreads_uploaded_paths(mod, monkeypatch):
    launched: dict = {}
    _patch_launch_client(mod, monkeypatch, launched)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert mod.launch(_Uploaded("e", "m", "d")) == "run-123"
    # the 3 UploadedEnvironmentAssets fields spread through **dataclasses.asdict
    assert launched["env_cls_path"] == "e"
    assert launched["dataset_path"] == "d"
    assert launched["name"] == mod._run_name()
    # LAUNCH_CONFIG feeds launcher_args, minus reserved keys
    assert "type" not in (launched["launcher_args"] or {})
    assert "name" not in (launched["launcher_args"] or {})
    assert (
        launched["launcher_args"]["max_context_tokens"] == mod.LAUNCH_CONFIG["max_context_tokens"]
    )


def test_launch_assume_yes_skips_prompt(mod, monkeypatch):
    launched: dict = {}
    _patch_launch_client(mod, monkeypatch, launched)
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError))
    assert mod.launch(_Uploaded("e", "m", "d"), assume_yes=True) == "run-123"
    assert launched["name"] == mod._run_name()


def test_runtime_dependencies_are_script_owned(mod):
    env_class = discover_env_class(mod)
    assert isinstance(mod.RUNTIME_DEPENDENCIES, list)
    assert not hasattr(env_class, "PIP_DEPENDENCIES")
