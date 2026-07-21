from pathlib import Path

import pytest
from benchmax.bundle import dump_bundle, load_bundle
from benchmax.envs.harbor import (
    BundledHarborAgent,
    ModalCredentials,
)
from harbor import EnvironmentType, TrialVerifierConfig

from harvey_agent import HarveyHarnessAgent
from main import HarveyLabHarborEnv


def test_harvey_constructor_uses_latest_dataset_and_native_harness() -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")

    env = HarveyLabHarborEnv(
        sandbox_credentials=credentials,
        judge_api_key="judge-key",
    )

    assert env._dataset.name == "harveyai/lab"
    assert env._dataset.ref == "latest"
    assert env._eval_ratio == 0.1
    assert env._sandbox_credentials is credentials
    trial = env._trial
    assert isinstance(trial.agent, BundledHarborAgent)
    assert trial.agent.config.import_path == "harvey_agent:HarveyHarnessAgent"
    assert trial.environment.type == EnvironmentType.MODAL
    assert trial.trials_dir == Path("/tmp/castform-harvey-harbor-trials")
    assert isinstance(trial.verifier, TrialVerifierConfig)
    assert trial.verifier.env["OPENAI_API_KEY"] == "judge-key"
    assert trial.verifier.env["OPENAI_BASE_URL"] == "https://llm.castform.dev/v1"


def test_harvey_bundles_carry_the_fixed_judge_key() -> None:
    """Judge and Modal credentials are fixed keys that ride in bundles."""

    bundle = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args={
            "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
            "judge_api_key": "judge-key",
        },
        pip_dependencies=["harbor[modal]>=0.18.0,<0.19"],
    )
    _, constructor_args = load_bundle(bundle, instantiate=False)
    assert constructor_args["judge_api_key"] == "judge-key"


def test_harvey_constructor_rejects_empty_judge_key() -> None:
    with pytest.raises(ValueError, match="judge_api_key"):
        HarveyLabHarborEnv(
            sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
            judge_api_key="",
        )


def test_harvey_agent_builds_harbor_task_command(tmp_path: Path) -> None:
    agent = HarveyHarnessAgent(
        logs_dir=tmp_path / "trial" / "agent",
        model_name="openai/gemma-model",
        extra_env={
            "OPENAI_API_KEY": "agent-key",
            "OPENAI_BASE_URL": "https://model.example/v1",
        },
    )

    command = agent._run_command("test-run", "Review the documents")

    assert "harvey_runtime.py" in command
    assert "--documents-dir /workspace/documents" in command
    assert "--output-dir /workspace/output" in command
    assert "--model gemma-model" in command
    assert "--max-turns 30" in command
    assert "/workspace/output/." in command


@pytest.mark.parametrize(
    ("extra_env", "missing_name"),
    [
        ({"OPENAI_BASE_URL": "https://model.example/v1"}, "OPENAI_API_KEY"),
        ({"OPENAI_API_KEY": "agent-key"}, "OPENAI_BASE_URL"),
    ],
)
def test_harvey_agent_requires_explicit_model_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_env: dict[str, str],
    missing_name: str,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = HarveyHarnessAgent(
        logs_dir=tmp_path / "trial" / "agent",
        model_name="openai/gemma-model",
        extra_env=extra_env,
    )

    with pytest.raises(RuntimeError, match=missing_name):
        agent._execution_env()


@pytest.mark.parametrize("judge_concurrency", [0, -1])
def test_harvey_constructor_rejects_invalid_judge_concurrency(
    judge_concurrency: int,
) -> None:
    with pytest.raises(ValueError, match="judge_concurrency must be positive"):
        HarveyLabHarborEnv(
            sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
            judge_api_key="judge-key",
            judge_concurrency=judge_concurrency,
        )
