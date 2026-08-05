"""The Harbor scaffold's explicit constructor and runtime-data workflow."""

from __future__ import annotations

import types
from pathlib import Path

import castform.cli.scaffold as scaffold_pkg
import pytest

from ._scaffold import load_module

_SEED = Path(scaffold_pkg.__file__).parent / "harbor_main.py"


@pytest.fixture
def mod(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return load_module(_SEED)


def _argv(action: str = "validate") -> list[str]:
    return [
        action,
        "--dataset",
        "org/tasks",
        "--modal-token-id",
        "modal-id",
        "--modal-token-secret",
        "modal-secret",
        "--verifier-env",
        "OPENAI_API_KEY=judge-key",
    ]


def _report(ok: bool = True):
    return types.SimpleNamespace(
        ok=ok,
        static_warnings={},
        static_errors={},
        local=None,
        remote=None,
        local_errors={},
        remote_errors={},
    )


def test_constructor_uses_stock_harness_without_model_overrides(mod):
    args = types.SimpleNamespace(
        dataset="org/tasks",
        dataset_ref="latest",
        modal_token_id="modal-id",
        modal_token_secret="modal-secret",
        verifier_env=["OPENAI_API_KEY=judge-key"],
    )

    constructor_args = mod._constructor_args(args)
    env = mod.CustomHarborEnv(**constructor_args)

    assert env._trial.agent.name == "mini-swe-agent"
    assert env._trial.agent.model_name is None
    assert env._trial.agent.kwargs == {}
    assert env._trial.verifier.env == {"OPENAI_API_KEY": "judge-key"}
    assert env.validation_diagnostics() == ()


def test_harbor_seed_bundles_with_its_explicit_constructor_args(mod):
    args = types.SimpleNamespace(
        dataset="org/tasks",
        dataset_ref="latest",
        modal_token_id="modal-id",
        modal_token_secret="modal-secret",
        verifier_env=[],
    )

    bundle = mod.dump_bundle(
        mod.CustomHarborEnv,
        constructor_args=mod._constructor_args(args),
        pip_dependencies=mod.RUNTIME_DEPENDENCIES,
    )

    assert bundle.pickled
    assert bundle.metadata.pip_dependencies == ("harbor[modal]<0.19,>=0.18.0",)


def test_data_action_needs_dataset_but_not_credentials(mod, capsys):
    assert mod.main(["data", "--dataset", "org/tasks"]) == 0

    output = capsys.readouterr().out
    assert "org/tasks@latest resolves through Harbor at runtime" in output
    assert "no JSONL upload is needed" in output


def test_validate_reuses_constructor_values_and_uploads_no_jsonl(mod, monkeypatch):
    captured: dict[str, object] = {}
    uploaded = types.SimpleNamespace(
        env_cls_path="env.pkl",
        env_metadata_path="metadata.json",
        dataset_path=None,
    )

    monkeypatch.setattr(mod, "ensure_session", lambda: None)

    def fake_bundle(env_class, **kwargs):
        captured["env_class"] = env_class
        captured["constructor_args"] = kwargs["constructor_args"]
        return object()

    def fake_upload(**kwargs):
        captured["upload"] = kwargs
        return uploaded

    def fake_validate(env, assets):
        captured["local_env"] = env
        captured["validated_assets"] = assets
        return _report()

    monkeypatch.setattr(mod, "dump_bundle", fake_bundle)
    monkeypatch.setattr(mod, "upload_assets", fake_upload)
    monkeypatch.setattr(mod, "validate", fake_validate)

    assert mod.main(_argv()) == 0

    constructor_args = captured["constructor_args"]
    local_env = captured["local_env"]
    assert constructor_args["dataset_name"] == local_env._dataset.name
    assert constructor_args["dataset_ref"] == local_env._dataset.ref
    assert constructor_args["verifier_env"] == local_env._trial.verifier.env
    assert set(captured["upload"]) == {"bundle", "run_name"}
    assert captured["validated_assets"] is uploaded


def test_validate_requires_explicit_modal_credentials_before_network(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "ensure_session",
        lambda: (_ for _ in ()).throw(AssertionError("network stage reached")),
    )

    with pytest.raises(SystemExit):
        mod.main(["validate", "--dataset", "org/tasks"])
