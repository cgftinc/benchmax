from pathlib import Path

import harvey_env
import pytest
from benchmax.bundle import dump_bundle, load_bundle
from benchmax.envs.harbor import HarborEnv, ModalCredentials
from harbor import EnvironmentType

from harvey_agent import HarveyHarnessAgent
from harvey_env import HarveyLabHarborEnv, harvey_harbor_constructor_args


def test_harvey_constructor_uses_latest_dataset_and_native_harness() -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")

    config = harvey_harbor_constructor_args(
        credentials,
        judge_api_key="judge-key",
    )

    assert config["dataset"].name == "harveyai/lab"
    assert config["dataset"].ref == "latest"
    assert config["eval_ratio"] == 0.1
    assert config["sandbox_credentials"] is credentials
    assert config["max_concurrent_trials"] == 1000
    trial = config["trial"]
    assert trial.agent.import_path == "harvey_agent:HarveyHarnessAgent"
    assert trial.environment.type == EnvironmentType.MODAL
    assert trial.trials_dir == Path("/tmp/castform-harvey-harbor-trials")
    assert trial.verifier.env == {
        "REWARDKIT_JUDGE": "openai/gpt-5.4-nano",
        "OPENAI_API_KEY": "judge-key",
        "OPENAI_BASE_URL": "https://llm.castform.dev/v1",
        "OPENAI_API_BASE": "https://llm.castform.dev/v1",
        "ANTHROPIC_API_KEY": "judge-key",
        "JUDGE_CONCURRENCY": "2",
    }


def test_harvey_environment_survives_by_value_bundle() -> None:
    constructor_args = {
        "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
        "judge_api_key": "judge-key",
    }

    bundle = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args=constructor_args,
        local_modules=[harvey_env],
    )
    restored_class, restored_args = load_bundle(bundle, instantiate=False)
    restored = restored_class(**restored_args)

    assert isinstance(restored, HarborEnv)
    assert len(bundle.pickled) > 50_000


def test_harvey_agent_builds_harbor_task_command(tmp_path: Path) -> None:
    agent = HarveyHarnessAgent(
        logs_dir=tmp_path / "trial" / "agent",
        model_name="openai/gemma-model",
        extra_env={
            "OPENAI_API_KEY": "agent-key",
            "OPENAI_BASE_URL": "https://model.example/v1",
            "HARBOR_HARVEY_BOOTSTRAP_PIP_PACKAGES": "",
        },
    )

    command = agent._run_command("test-run", "Review the documents")

    assert "run-harbor-task" in command
    assert "--documents-dir /workspace/documents" in command
    assert "--output-dir /workspace/output" in command
    assert "--model gemma-model" in command
    assert "--max-turns 30" in command


@pytest.mark.parametrize("judge_concurrency", [0, -1])
def test_harvey_constructor_rejects_invalid_judge_concurrency(
    judge_concurrency: int,
) -> None:
    with pytest.raises(ValueError, match="judge_concurrency must be positive"):
        harvey_harbor_constructor_args(
            ModalCredentials("modal-id", "modal-secret"),
            judge_api_key="judge-key",
            judge_concurrency=judge_concurrency,
        )
