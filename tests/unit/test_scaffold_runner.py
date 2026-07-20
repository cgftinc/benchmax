"""Every scaffold seed's `main.py` runnable stage-runner: argparse dispatch +
SDK-direct stages (data → validate → launch), with the SDK monkeypatched (no
network). Parametrized over every seed for the mode-agnostic assertions (import
safety, dispatch); RL-only and sft-only assertions get their own fixture, since
the sft seed is env-less and uses a different SDK surface (see `_RL_SEEDS`)."""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import pytest

import benchmax.cli.scaffold as scaffold_pkg
from benchmax.cli._project import _load_module_from_file, discover_env_class

_SCAFFOLD_DIR = Path(scaffold_pkg.__file__).parent
_SEEDS = {
    "generic": _SCAFFOLD_DIR / "generic_main.py",
    "rag": _SCAFFOLD_DIR / "rag_main.py",
    "sft": _SCAFFOLD_DIR / "sft_main.py",
}
# sft is env-less with a different SDK surface than generic/rag — see mod/rl_mod/sft_mod below.
_RL_SEEDS = ("generic", "rag")


@pytest.fixture(params=list(_SEEDS))
def mod(request, tmp_path, monkeypatch):
    """Load a scaffold seed as a module (its `__main__` block does not fire).
    Parametrized over EVERY seed — use this only for mode-agnostic assertions
    (import safety, argparse dispatch). RL-only assertions belong on `rl_mod`."""
    monkeypatch.chdir(tmp_path)
    return _load_module_from_file(_SEEDS[request.param])


@pytest.fixture(params=_RL_SEEDS)
def rl_mod(request, tmp_path, monkeypatch):
    """Like `mod`, but parametrized over the RL (env-class) seeds only — for
    assertions that reach `discover_env_class`/`validate_env`/`upload_training_run`/
    `launch_training_run`, none of which the env-less sft seed has."""
    monkeypatch.chdir(tmp_path)
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


# ── validate stage: config-derived args reach validate_env (RL seeds only) ──────


def test_validate_passes_config_derived_args(rl_mod, tmp_path, monkeypatch):
    mod = rl_mod
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


# ── launch stage: validate-gate, [y/N] confirm, and the asdict spread (RL seeds) ─


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


def test_launch_blocked_by_failing_validate(rl_mod, tmp_path, monkeypatch):
    mod = rl_mod
    monkeypatch.chdir(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched, validate_ok=False)
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    assert mod.launch() is None
    assert not launched  # never reached upload/launch


def test_launch_declined_at_confirm(rl_mod, tmp_path, monkeypatch):
    mod = rl_mod
    monkeypatch.chdir(tmp_path)
    _seed_datasets(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert mod.launch() is None
    assert not launched  # aborted before spending credits


def test_launch_confirmed_spreads_uploaded_paths(rl_mod, tmp_path, monkeypatch):
    mod = rl_mod
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


def test_launch_assume_yes_skips_prompt(rl_mod, tmp_path, monkeypatch):
    mod = rl_mod
    monkeypatch.chdir(tmp_path)
    _seed_datasets(tmp_path)
    launched: dict = {}
    _patch_launch_sdk(mod, monkeypatch, launched)
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    assert mod.launch(assume_yes=True) == "run-123"
    assert launched["name"] == mod._run_name()


# ── sft seed: env-less validate/launch surface ──────────────────────────────────


@pytest.fixture
def sft_mod(tmp_path, monkeypatch):
    """The sft seed loaded on its own (not parametrized — its SDK surface differs
    from the RL seeds', see `_RL_SEEDS` above)."""
    monkeypatch.chdir(tmp_path)
    return _load_module_from_file(_SEEDS["sft"])


def test_sft_seed_multimodal_row_validates(sft_mod):
    """`_SEED_MULTIMODAL` must validate cleanly through `validate_sft_dataset` on
    EVERY test run (not just when a user manually enables it) so the opt-in sample
    can't silently rot. Checked against the module's own dataset functions so this
    breaks the moment the seed or the schema drifts."""
    from benchmax.sft.dataset import SftDataset, SftRow

    dataset = SftDataset(
        path="<_SEED_MULTIMODAL>",
        rows=[SftRow("<_SEED_MULTIMODAL>", 1, sft_mod._SEED_MULTIMODAL)],
        load_issues=[],
    )
    report = sft_mod.validate_sft_dataset(dataset)
    assert report.ok, [
        (i.severity, i.message) for i in report.issues if i.severity == "error"
    ]


def test_sft_validate_reads_real_dataset(sft_mod, tmp_path):
    """`validate()` runs the real `load_sft_dataset`/`validate_sft_dataset` pair
    against the seed data written by `generate_data()` — no network involved."""
    sft_mod.generate_data()
    report = sft_mod.validate()
    assert report.ok
    assert report.train_row_count == len(sft_mod._SEED_TRAIN)
    assert report.eval_row_count == len(sft_mod._SEED_EVAL)


def test_sft_launch_blocked_by_failing_validate(sft_mod, tmp_path, monkeypatch):
    """A real invalid dataset (empty train) must block launch through the real
    validate_sft_dataset gate -- launch() no longer delegates to the module-level
    `validate()` wrapper (see the TOCTOU fix: it loads once and validates those
    same objects), so the gate is exercised directly rather than mocked."""
    (tmp_path / sft_mod.TRAIN_FILE).write_text("")  # empty -> train_row_count == 0
    calls: list = []
    monkeypatch.setattr(sft_mod, "upload_sft_run", lambda **k: calls.append(k))
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    assert sft_mod.launch() is None
    assert not calls  # never reached upload


def test_sft_launch_no_upload_while_unsupported(sft_mod, monkeypatch):
    """The pre-upload capability guard: with `SFT_LAUNCH_SUPPORTED` false (its real
    value as of writing — the live platform doesn't accept env-less sft runs yet),
    `launch()` must record NO upload call and return None, `--yes` included."""
    sft_mod.generate_data()
    assert sft_mod.SFT_LAUNCH_SUPPORTED is False
    calls: list = []
    monkeypatch.setattr(sft_mod, "upload_sft_run", lambda **k: calls.append(k))
    assert sft_mod.launch(assume_yes=True) is None
    assert not calls


def _fake_uploaded_sft_run(train_dataset_path="t", eval_dataset_path="v"):
    Uploaded = dataclasses.make_dataclass(
        "UploadedSftRun", ["train_dataset_path", "eval_dataset_path"]
    )
    return Uploaded(train_dataset_path, eval_dataset_path)


def test_sft_launch_uploads_and_launches_when_supported(sft_mod, monkeypatch):
    """With the capability flag patched true, `launch()` runs validate -> weight
    gate -> upload_sft_run -> `TrainerClient.launch_sft_run`, in that order, and
    spreads the uploaded paths into the launch call."""
    sft_mod.generate_data()
    monkeypatch.setattr(sft_mod, "SFT_LAUNCH_SUPPORTED", True)
    monkeypatch.setattr(sft_mod, "upload_sft_run", lambda **k: _fake_uploaded_sft_run())
    launched: dict = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def launch_sft_run(self, **kw):
            launched.update(kw)
            return "run-123"

    monkeypatch.setattr(sft_mod, "TrainerClient", lambda: FakeClient())
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    assert sft_mod.launch() == "run-123"
    assert launched["train_dataset_path"] == "t"
    assert launched["eval_dataset_path"] == "v"
    assert launched["name"] == sft_mod._run_name()
    assert "allow_experimental_weights" not in (launched["launcher_args"] or {})
    assert "type" not in (launched["launcher_args"] or {})
    assert (
        launched["launcher_args"]["num_epochs"] == sft_mod.LAUNCH_CONFIG["num_epochs"]
    )


def test_sft_launch_blocked_by_weight_gate(sft_mod, tmp_path, monkeypatch):
    """A weight-bearing dataset blocks launch before the SFT_LAUNCH_SUPPORTED guard
    (and before upload) unless LAUNCH_CONFIG opts in via
    `allow_experimental_weights`."""
    weighted_row = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "masked", "weight": 0},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "trained"},
        ]
    }
    (tmp_path / sft_mod.TRAIN_FILE).write_text(json.dumps(weighted_row) + "\n")
    (tmp_path / sft_mod.EVAL_FILE).write_text(json.dumps(weighted_row) + "\n")

    monkeypatch.setattr(sft_mod, "SFT_LAUNCH_SUPPORTED", True)
    calls: list = []
    monkeypatch.setattr(sft_mod, "upload_sft_run", lambda **k: calls.append(k))
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )

    assert sft_mod.launch(assume_yes=True) is None
    assert not calls  # blocked before upload

    sft_mod.LAUNCH_CONFIG["allow_experimental_weights"] = True
    try:
        monkeypatch.setattr(
            sft_mod, "upload_sft_run", lambda **k: _fake_uploaded_sft_run()
        )

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def launch_sft_run(self, **kw):
                return "run-456"

        monkeypatch.setattr(sft_mod, "TrainerClient", lambda: FakeClient())
        assert sft_mod.launch(assume_yes=True) == "run-456"
    finally:
        sft_mod.LAUNCH_CONFIG.pop("allow_experimental_weights", None)


# ── fail-closed config resolution: clean errors, no truthy-coercion ────────────


def test_sft_validate_rejects_non_int_max_seq_len(sft_mod, monkeypatch):
    """A malformed VALIDATE_CONFIG value (a string, not an int) must raise a
    clean config error, not a raw TypeError deep inside validate_sft_dataset."""
    from benchmax.sft import SftConfigError

    sft_mod.generate_data()
    monkeypatch.setitem(sft_mod.VALIDATE_CONFIG, "max_seq_len", "100")
    with pytest.raises(SftConfigError, match="max_seq_len"):
        sft_mod.validate()


def test_sft_launch_rejects_non_bool_allow_experimental_weights_string(
    sft_mod, tmp_path, monkeypatch
):
    """LAUNCH_CONFIG['allow_experimental_weights'] = "false" (a string, truthy in
    Python) must be rejected outright -- resolving it by truthiness would
    silently CLEAR the experimental-weight safety gate, the opposite of what a
    user writing "false" almost certainly meant."""
    from benchmax.sft import SftConfigError

    weighted_row = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "masked", "weight": 0},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "trained"},
        ]
    }
    (tmp_path / sft_mod.TRAIN_FILE).write_text(json.dumps(weighted_row) + "\n")
    monkeypatch.setitem(sft_mod.LAUNCH_CONFIG, "allow_experimental_weights", "false")
    calls: list = []
    monkeypatch.setattr(sft_mod, "upload_sft_run", lambda **k: calls.append(k))
    monkeypatch.setattr(
        "builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError)
    )

    with pytest.raises(SftConfigError):
        sft_mod.launch(assume_yes=True)
    assert not calls  # rejected before upload, not silently treated as False


# ── generate_data: the 5-case existence state machine ──────────────────────────

_SENTINEL_TRAIN = "SENTINEL_TRAIN_ROW\n"
_SENTINEL_EVAL = "SENTINEL_EVAL_ROW\n"


def test_generate_data_neither_exists_creates_both(sft_mod, tmp_path):
    sft_mod.generate_data()
    assert (tmp_path / sft_mod.TRAIN_FILE).exists()
    assert (tmp_path / sft_mod.EVAL_FILE).exists()


def test_generate_data_both_exist_skips_both(sft_mod, tmp_path):
    (tmp_path / sft_mod.TRAIN_FILE).write_text(_SENTINEL_TRAIN)
    (tmp_path / sft_mod.EVAL_FILE).write_text(_SENTINEL_EVAL)

    sft_mod.generate_data()

    assert (tmp_path / sft_mod.TRAIN_FILE).read_text() == _SENTINEL_TRAIN
    assert (tmp_path / sft_mod.EVAL_FILE).read_text() == _SENTINEL_EVAL


def test_generate_data_train_only_skips_and_never_fabricates_eval(sft_mod, tmp_path):
    """A train-only project is a valid, intentional state (eval is optional in
    SFT) -- `generate_data()` must not inject an unrelated seed eval file into it
    (regression: the per-file existence fix still unconditionally wrote the
    missing counterpart)."""
    (tmp_path / sft_mod.TRAIN_FILE).write_text(_SENTINEL_TRAIN)

    sft_mod.generate_data()

    assert (tmp_path / sft_mod.TRAIN_FILE).read_text() == _SENTINEL_TRAIN
    assert not (tmp_path / sft_mod.EVAL_FILE).exists()


def test_generate_data_eval_only_without_train_fails_loudly(sft_mod, tmp_path):
    """eval without train has no legitimate explanation (a corrupted project, not
    a first run) -- `generate_data()` must refuse outright rather than fabricate
    train data to match, and must leave the directory untouched."""
    (tmp_path / sft_mod.EVAL_FILE).write_text(_SENTINEL_EVAL)

    with pytest.raises(sft_mod.SftScaffoldError):
        sft_mod.generate_data()

    assert not (tmp_path / sft_mod.TRAIN_FILE).exists()
    assert (tmp_path / sft_mod.EVAL_FILE).read_text() == _SENTINEL_EVAL


def test_generate_data_force_regenerates_both_regardless_of_state(sft_mod, tmp_path):
    (tmp_path / sft_mod.TRAIN_FILE).write_text(_SENTINEL_TRAIN)
    (tmp_path / sft_mod.EVAL_FILE).write_text(_SENTINEL_EVAL)

    sft_mod.generate_data(force=True)

    assert (tmp_path / sft_mod.TRAIN_FILE).read_text() != _SENTINEL_TRAIN
    assert (tmp_path / sft_mod.EVAL_FILE).read_text() != _SENTINEL_EVAL


def test_generate_data_dangling_symlink_is_refused_not_followed(sft_mod, tmp_path):
    """A dangling symlink at a target path must be treated as "already there" --
    refused (skip, no crash) rather than silently followed/clobbered. Exercises
    the exclusive-create path's O_EXCL semantics, which never follow a symlink."""
    (tmp_path / sft_mod.TRAIN_FILE).symlink_to(tmp_path / "does_not_exist_target")

    sft_mod.generate_data()  # must not raise / crash

    # the dangling symlink itself is untouched -- no target was ever created
    assert (tmp_path / sft_mod.TRAIN_FILE).is_symlink()
    assert not (tmp_path / sft_mod.TRAIN_FILE).exists()  # still dangling (unfollowed)
    # train "exists" (as a path entry) from generate_data's point of view, so
    # eval is treated as the train-only-preserved case -- never fabricated
    assert not (tmp_path / sft_mod.EVAL_FILE).exists()


# ── launch: validate-then-upload TOCTOU — load once, never reload from disk ────


def test_sft_launch_uses_loaded_dataset_not_reloaded_after_confirm(
    sft_mod, tmp_path, monkeypatch
):
    """regression: launch() must validate and upload the SAME loaded objects. An
    edit/swap of the on-disk file between the initial validate/load and the launch
    confirmation must NOT reach `upload_sft_run` -- otherwise what gets uploaded was
    never actually the thing that was validated (a TOCTOU gap: `launch()` used to
    reload from disk right before `upload_sft_run`, after the user confirmed)."""
    sft_mod.generate_data()
    monkeypatch.setattr(sft_mod, "SFT_LAUNCH_SUPPORTED", True)
    captured: dict = {}

    def _fake_upload(train, eval, run_name):
        captured["train_rows"] = [r.data for r in train.rows]
        return _fake_uploaded_sft_run()

    monkeypatch.setattr(sft_mod, "upload_sft_run", _fake_upload)

    def _swap_then_confirm(*a):
        # An attacker/race edits the on-disk file after the initial validate/load,
        # right before the user types 'y' at the confirmation prompt.
        (tmp_path / sft_mod.TRAIN_FILE).write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "SWAPPED"},
                        {"role": "assistant", "content": "SWAPPED"},
                    ]
                }
            )
            + "\n"
        )
        return "y"

    monkeypatch.setattr("builtins.input", _swap_then_confirm)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def launch_sft_run(self, **kw):
            return "run-789"

    monkeypatch.setattr(sft_mod, "TrainerClient", lambda: FakeClient())

    assert sft_mod.launch() == "run-789"
    # the ORIGINAL, validated seed content reached upload -- not the swapped rows
    assert captured["train_rows"] == sft_mod._SEED_TRAIN
    assert not any("SWAPPED" in json.dumps(row) for row in captured["train_rows"])
    # the swapped file really is on disk -- proving a reload (the bug) would have
    # picked it up instead
    assert "SWAPPED" in (tmp_path / sft_mod.TRAIN_FILE).read_text()
