from pathlib import Path

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
        "JUDGE_CONCURRENCY": "1",
    }


def test_harvey_environment_survives_by_value_bundle() -> None:
    constructor_args = {
        "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
        "judge_api_key": "judge-key",
    }

    bundle = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args=constructor_args,
        pip_dependencies=["harbor[modal]>=0.18.0,<0.19"],
    )
    restored_class, restored_args = load_bundle(bundle, instantiate=False)
    restored = restored_class(**restored_args)

    assert isinstance(restored, HarborEnv)
    assert bundle.metadata.pip_dependencies == ("harbor[modal]<0.19,>=0.18.0",)
    captured_sources = restored_class.__init__.__globals__["_AGENT_SOURCES"]
    assert set(captured_sources) == {"harvey_agent.py", "harvey_runtime.py"}
    assert "class HarveyHarnessAgent" in captured_sources["harvey_agent.py"]
    assert "class HarborOwnedSandbox" in captured_sources["harvey_runtime.py"]


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
        harvey_harbor_constructor_args(
            ModalCredentials("modal-id", "modal-secret"),
            judge_api_key="judge-key",
            judge_concurrency=judge_concurrency,
        )
