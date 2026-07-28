"""Every scaffold seed's `main.py` runnable stage-runner: argparse dispatch +
SDK-direct stages (data → validate → launch), with the SDK monkeypatched (no
network). Parametrized over both seeds — the runner is env-type-agnostic."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import castform.cli.scaffold as scaffold_pkg
import pytest

from ._scaffold import discover_env_class, load_module

_SCAFFOLD_DIR = Path(scaffold_pkg.__file__).parent
_SEEDS = {
    "generic": _SCAFFOLD_DIR / "generic_main.py",
    "rag": _SCAFFOLD_DIR / "rag_main.py",
}


@pytest.fixture(params=["generic", "rag"])
def mod(request, tmp_path, monkeypatch):
    """Load a scaffold seed as a module (its `__main__` block does not fire)."""
    monkeypatch.chdir(tmp_path)
    return load_module(_SEEDS[request.param])


def _fake_report(ok: bool = True):
    outcome = types.SimpleNamespace(
        rewards={"score": 1.0}, termination_reason="finished"
    )
    return types.SimpleNamespace(
        ok=ok,
        local={"validate-0": outcome, "validate-1": outcome},
        remote=None,
    )


# ── import safety: loading defines the stages, runs none of them ────────────────


def test_import_defines_stages_without_running(mod, tmp_path, monkeypatch):
    """Loading a seed defines the runner but executes NO stage (the `__main__`
    block is import-safe)."""
    assert all(hasattr(mod, n) for n in ("main", "generate_data", "validate", "launch"))
    assert not (tmp_path / "train.jsonl").exists()


def test_rag_runtime_dependency_matches_the_generating_sdk(mod):
    if not hasattr(mod, "CustomSearchEnv"):
        pytest.skip("generic scaffold has no Castform runtime dependency")

    from importlib.metadata import version

    assert mod.RUNTIME_DEPENDENCIES == [f"castform=={version('castform')}"]


# ── argparse dispatch: the right stage fns, and the safe-prefix STOP ────────────


def _patch_stage_recorders(mod, monkeypatch):
    calls: list = []
    monkeypatch.setattr(mod, "ensure_session", lambda *a, **k: None)

    def _data(**k):
        calls.append(("data", k))
        return True

    def _validate():  # returns a passing report so main() exits 0
        calls.append(("validate",))
        return _fake_report(ok=True)

    def _launch(**k):  # returns a run id so main() exits 0
        calls.append(("launch", k))
        return "run-x"

    monkeypatch.setattr(mod, "generate_data", _data)
    monkeypatch.setattr(mod, "validate", _validate)
    monkeypatch.setattr(mod, "launch", _launch)
    return calls


def test_dispatch_validate_only(mod, monkeypatch):
    calls = _patch_stage_recorders(mod, monkeypatch)
    assert mod.main(["validate"]) == 0
    assert calls == [("validate",)]


def test_dispatch_data_only(mod, monkeypatch):
    calls = _patch_stage_recorders(mod, monkeypatch)
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


def test_validate_stage_ensures_session(mod, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(mod, "ensure_session", lambda: calls.append("login"))
    monkeypatch.setattr(mod, "validate", lambda: _fake_report(ok=True))
    assert mod.main(["validate"]) == 0
    assert calls == ["login"]


def test_bare_and_all_run_data_then_validate_then_stop(mod, monkeypatch):
    """Bare `python main.py` (and `all`) run the SAFE prefix data → validate and
    STOP — launch is never automatic (GPU spend)."""
    for argv in ([], ["all"]):
        calls = _patch_stage_recorders(mod, monkeypatch)
        assert mod.main(argv) == 0
        assert calls == [("data", {"force": False}), ("validate",)]
        assert not any(c[0] == "launch" for c in calls)


def test_dispatch_launch_passes_yes(mod, monkeypatch):
    calls = _patch_stage_recorders(mod, monkeypatch)
    assert mod.main(["launch"]) == 0
    assert calls == [("launch", {"assume_yes": False})]
    calls.clear()
    assert mod.main(["launch", "-y"]) == 0
    assert calls == [("launch", {"assume_yes": True})]


def test_main_exit_1_on_validate_failure(mod, monkeypatch):
    """A failed baseline must exit non-zero — CI / an agent can't treat a broken
    validate as success."""
    monkeypatch.setattr(mod, "ensure_session", lambda *a, **k: None)
    monkeypatch.setattr(mod, "generate_data", lambda **k: None)
    monkeypatch.setattr(mod, "validate", lambda: _fake_report(ok=False))
    assert mod.main(["validate"]) == 1
    assert mod.main([]) == 1  # bare = data → validate; a failing validate is non-zero


def test_main_exit_1_when_launch_gated(mod, monkeypatch):
    """launch() → None (validate-gate failed / user aborted / non-TTY refused) must
    exit non-zero."""
    monkeypatch.setattr(mod, "ensure_session", lambda *a, **k: None)
    monkeypatch.setattr(mod, "launch", lambda **k: None)
    assert mod.main(["launch"]) == 1


# ── validate stage: script calls the public group-native function ───────────────


def test_validate_loads_the_public_dataset_then_runs_group_validation(
    mod, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    field = "question" if hasattr(mod, "CustomSearchEnv") else "prompt"
    (tmp_path / "train.jsonl").write_text(f'{{"{field}": "q"}}\n')
    (tmp_path / "eval.jsonl").write_text(f'{{"{field}": "q2"}}\n')
    captured: dict = {}
    env_class = discover_env_class(mod)
    original_create_dataset = env_class.create_dataset

    async def recording_create_dataset(self, split, base_dir):
        captured["dataset_call"] = (split, base_dir)
        return await original_create_dataset(self, split, base_dir)

    async def fake_validate_environment(env, **kw):
        captured["env"] = env
        captured.update(kw)
        return _fake_report(ok=True)

    monkeypatch.setattr(env_class, "create_dataset", recording_create_dataset)
    monkeypatch.setattr(mod, "validate_environment", fake_validate_environment)
    report = mod.validate()

    assert report.ok
    assert isinstance(captured["env"], env_class)
    assert captured["dataset_call"] == ("train", Path("."))
    assert captured["example"].id
    assert captured["model"] == str(mod.VALIDATE_CONFIG["model"])
    assert captured["include_remote"] is bool(
        mod.VALIDATE_CONFIG.get("include_remote", False)
    )


def test_validate_surfaces_public_dataset_materialization_failure(mod, monkeypatch):
    env_class = discover_env_class(mod)

    async def fail_create_dataset(self, split, base_dir):
        raise RuntimeError("dataset materialization failed")

    async def should_not_validate(*args, **kwargs):
        raise AssertionError("validation ran without a materialized example")

    monkeypatch.setattr(env_class, "create_dataset", fail_create_dataset)
    monkeypatch.setattr(mod, "validate_environment", should_not_validate)

    with pytest.raises(RuntimeError, match="dataset materialization failed"):
        mod.validate()


def test_scorecard_prints_termination_reason_and_complete_reward_shape(mod, capsys):
    outcome = types.SimpleNamespace(
        rewards={"correct": 0.0, "format": 0.0},
        termination_reason="judge_error",
    )
    report = types.SimpleNamespace(
        ok=False,
        local={"validate-0": outcome},
    )

    mod._print_scorecard(report)

    output = capsys.readouterr().out
    assert "termination_reason=judge_error" in output
    assert "rewards={'correct': 0.0, 'format': 0.0}" in output
    assert "validate: FAIL" in output


# ── launch stage: validate-gate, [y/N] confirm, and the asdict spread ───────────


def _patch_launch_sdk(mod, monkeypatch, launched: dict, *, validate_ok: bool = True):
    monkeypatch.setattr(mod, "validate", lambda: _fake_report(ok=validate_ok))
    bundle = object()

    def fake_dump_bundle(env_class, **kwargs):
        launched["_bundle_call"] = {"env_class": env_class, **kwargs}
        return bundle

    monkeypatch.setattr(mod, "dump_bundle", fake_dump_bundle)
    Uploaded = dataclasses.make_dataclass(
        "Uploaded",
        [
            "env_cls_path",
            "env_metadata_path",
            "dataset_path",
        ],
    )

    def fake_upload_training_run(**kwargs):
        launched["_upload_call"] = kwargs
        return Uploaded("e", "m", "d")

    monkeypatch.setattr(mod, "upload_training_run", fake_upload_training_run)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def launch_training_run(self, **kw):
            launched.update(kw)
            return "run-123"

    monkeypatch.setattr(mod, "TrainerClient", lambda: FakeClient())


def _seed_datasets(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"prompt": "q"}\n')
    (tmp_path / "eval.jsonl").write_text('{"prompt": "q2"}\n')


def test_launch_blocked_by_failing_validate(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched, validate_ok=False)
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    assert mod.launch() is None
    assert not launched  # never reached upload/launch


def test_launch_declined_at_confirm(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_datasets(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert mod.launch() is None
    assert not launched  # aborted before spending credits


def test_launch_confirmed_spreads_uploaded_paths(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_datasets(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert mod.launch() == "run-123"
    # the 3 UploadedTrainingRun fields spread through **dataclasses.asdict
    assert launched["env_cls_path"] == "e"
    assert launched["dataset_path"] == "d"
    assert launched["name"] == mod._run_name()
    assert launched["_bundle_call"] == {
        "env_class": discover_env_class(mod),
        "constructor_args": mod.ENV_ARGS,
        "pip_dependencies": mod.RUNTIME_DEPENDENCIES,
    }
    assert launched["_upload_call"]["bundle"] is not None
    assert "env_class" not in launched["_upload_call"]
    assert "pip_dependencies" not in launched["_upload_call"]
    # LAUNCH_CONFIG feeds launcher_args, minus reserved keys
    assert "type" not in (launched["launcher_args"] or {})
    assert "name" not in (launched["launcher_args"] or {})
    assert (
        launched["launcher_args"]["max_rollout_len"]
        == mod.LAUNCH_CONFIG["max_rollout_len"]
    )


def test_launch_assume_yes_skips_prompt(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_datasets(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched)
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    assert mod.launch(assume_yes=True) == "run-123"
    assert launched["name"] == mod._run_name()


def test_runtime_dependencies_are_script_owned(mod):
    env_class = discover_env_class(mod)
    assert isinstance(mod.RUNTIME_DEPENDENCIES, list)
    assert not hasattr(env_class, "PIP_DEPENDENCIES")


def test_rag_seed_does_not_import_workspace_showcase():
    source = _SEEDS["rag"].read_text()
    assert "postgres_search_env" not in source
