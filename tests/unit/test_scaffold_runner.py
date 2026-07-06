"""Every scaffold seed's `main.py` runnable stage-runner: argparse dispatch +
SDK-direct stages (data → validate → launch), with the SDK monkeypatched (no
network). Parametrized over both seeds — the runner is env-type-agnostic."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import pytest

import benchmax.cli.scaffold as scaffold_pkg
from benchmax.cli._project import _load_module_from_file, discover_env_class

_SCAFFOLD_DIR = Path(scaffold_pkg.__file__).parent
_SEEDS = {
    "generic": _SCAFFOLD_DIR / "generic_main.py",
    "rag": _SCAFFOLD_DIR / "rag_main.py",
    "traces": _SCAFFOLD_DIR / "traces_main.py",
    "judge": _SCAFFOLD_DIR / "judge_main.py",
}


@pytest.fixture(params=["generic", "rag", "traces", "judge"])
def mod(request):
    """Load a scaffold seed as a module (its `__main__` block does not fire)."""
    return _load_module_from_file(_SEEDS[request.param])


def _fake_report(ok: bool = True):
    ex = types.SimpleNamespace(index=0, ok=True, rewards={"score": 1.0}, error=None)
    remote = types.SimpleNamespace(examples=[ex], group_reward=None)
    return types.SimpleNamespace(ok=ok, remote=remote)


# ── import safety: loading defines the stages, runs none of them ────────────────


def test_import_defines_stages_without_running(mod, tmp_path, monkeypatch):
    """Loading a seed defines the runner but executes NO stage (the `__main__`
    block is import-safe)."""
    assert all(hasattr(mod, n) for n in ("main", "generate_data", "validate", "launch"))
    assert not (tmp_path / "train_dataset.jsonl").exists()


# ── argparse dispatch: the right stage fns, and the safe-prefix STOP ────────────


def _patch_stage_recorders(mod, monkeypatch):
    calls: list = []
    monkeypatch.setattr(mod, "ensure_session", lambda *a, **k: None)

    def _data(**k):
        calls.append(("data", k))

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


# ── validate stage: config-derived args reach validate_env ──────────────────────


def test_validate_passes_config_derived_args(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    (tmp_path / "eval_dataset.jsonl").write_text('{"prompt": "q2"}\n')
    captured: dict = {}

    def fake_validate_env(**kw):
        captured.update(kw)
        return _fake_report(ok=True)

    monkeypatch.setattr(mod, "validate_env", fake_validate_env)
    report = mod.validate()

    assert report.ok
    assert captured["env_class"] is discover_env_class(mod)
    assert captured["local"] is False  # the remote real-rollout subset
    # each seed's VALIDATE_CONFIG feeds validate_env (defaults where a key is absent)
    assert captured["max_turns"] == mod.VALIDATE_CONFIG.get("max_turns", 4)
    assert captured["max_tool_calls"] == mod.VALIDATE_CONFIG.get("max_tool_calls", 8)
    assert captured["remote_examples"] == mod.VALIDATE_CONFIG.get("examples", 2)
    assert len(captured["train_dataset"]) == 1  # loaded off disk, not passed empty


# ── launch stage: validate-gate, [y/N] confirm, and the asdict spread ───────────


def _patch_launch_sdk(mod, monkeypatch, launched: dict, *, validate_ok: bool = True):
    monkeypatch.setattr(mod, "validate", lambda: _fake_report(ok=validate_ok))
    Uploaded = dataclasses.make_dataclass(
        "Uploaded",
        [
            "env_cls_path",
            "env_metadata_path",
            "train_dataset_path",
            "eval_dataset_path",
        ],
    )
    monkeypatch.setattr(
        mod, "upload_training_run", lambda **k: Uploaded("e", "m", "t", "v")
    )

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
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    (tmp_path / "eval_dataset.jsonl").write_text('{"prompt": "q2"}\n')


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
    # the 4 UploadedTrainingRun fields spread through **dataclasses.asdict
    assert launched["env_cls_path"] == "e"
    assert launched["train_dataset_path"] == "t"
    assert launched["name"] == mod._run_name()
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
