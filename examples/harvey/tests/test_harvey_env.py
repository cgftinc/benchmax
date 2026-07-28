from pathlib import Path
from types import SimpleNamespace

import main as harvey_main
import pytest
from benchmax.bundle import dump_bundle, load_bundle
from benchmax.envs.harbor import (
    BundledHarborAgent,
    ModalCredentials,
)
from harbor import EnvironmentType, TrialVerifierConfig

from harvey_agent import HarveyHarnessAgent
from main import (
    HarveyLabHarborEnv,
    _modal_credentials_from_process,
    _verifier_env_for_provider,
)


def test_harvey_constructor_uses_latest_dataset_and_native_harness() -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")

    env = HarveyLabHarborEnv(
        sandbox_credentials=credentials,
        verifier_env={"ANTHROPIC_API_KEY": "anthropic-key"},
        judge_model="anthropic/claude-sonnet-4-6",
    )

    assert env._dataset.name == "harveyai/lab"
    assert env._dataset.ref == "latest"
    assert env._eval_ratio == 0.1
    assert env._sandbox_credentials is credentials
    trial = env._trial
    assert isinstance(trial.agent, BundledHarborAgent)
    assert trial.agent.config.import_path == "harvey_agent:HarveyHarnessAgent"
    assert trial.environment.type == EnvironmentType.MODAL
    assert trial.environment.kwargs["app_name"] == "harbor-benchmax"
    assert trial.environment.kwargs["sandbox_timeout_secs"] == 10800
    assert trial.environment.kwargs["sandbox_idle_timeout_secs"] == 1800
    assert trial.trials_dir == Path("/tmp/castform-harvey-harbor-trials")
    assert isinstance(trial.verifier, TrialVerifierConfig)
    assert trial.verifier.env == {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "REWARDKIT_JUDGE": "anthropic/claude-sonnet-4-6",
        "JUDGE_CONCURRENCY": "1",
    }


def test_harvey_constructor_overrides_modal_sandbox_lifecycle() -> None:
    env = HarveyLabHarborEnv(
        sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
        verifier_env={"ANTHROPIC_API_KEY": "anthropic-key"},
        judge_model="anthropic/claude-sonnet-4-6",
        modal_app_name="harbor-long-docs",
        sandbox_timeout_secs=7200,
        sandbox_idle_timeout_secs=900,
    )

    assert env._trial.environment.kwargs == {
        "app_name": "harbor-long-docs",
        "sandbox_timeout_secs": 7200,
        "sandbox_idle_timeout_secs": 900,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sandbox_timeout_secs", 0),
        ("sandbox_timeout_secs", True),
        ("sandbox_idle_timeout_secs", -1),
    ],
)
def test_harvey_constructor_rejects_invalid_modal_timeouts(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        HarveyLabHarborEnv(
            sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
            verifier_env={"ANTHROPIC_API_KEY": "anthropic-key"},
            judge_model="anthropic/claude-sonnet-4-6",
            **{name: value},
        )


def test_harvey_bundles_carry_the_explicit_verifier_environment() -> None:
    """Verifier and Modal credentials are fixed values that ride in bundles."""

    bundle = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args={
            "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
            "verifier_env": {
                "OPENAI_API_KEY": "judge-key",
                "OPENAI_BASE_URL": "https://llm.castform.dev/v1",
            },
            "judge_model": "openai/gpt-5.4-nano",
        },
        pip_dependencies=["harbor[modal]>=0.18.0,<0.19"],
    )
    _, constructor_args = load_bundle(bundle, instantiate=False)
    assert constructor_args["verifier_env"] == {
        "OPENAI_API_KEY": "judge-key",
        "OPENAI_BASE_URL": "https://llm.castform.dev/v1",
    }


def test_harvey_constructor_does_not_copy_credentials_between_providers() -> None:
    env = HarveyLabHarborEnv(
        sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
        verifier_env={"OPENAI_API_KEY": "judge-key"},
        judge_model="openai/gpt-5.4-nano",
    )

    assert env._trial.verifier.env["OPENAI_API_KEY"] == "judge-key"
    assert "ANTHROPIC_API_KEY" not in env._trial.verifier.env


def test_harvey_constructor_rejects_empty_verifier_environment() -> None:
    with pytest.raises(ValueError, match="verifier_env"):
        HarveyLabHarborEnv(
            sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
            verifier_env={},
            judge_model="anthropic/claude-sonnet-4-6",
        )


@pytest.mark.parametrize(
    "verifier_env",
    [
        {"REWARDKIT_JUDGE": "anthropic/model"},
        {"JUDGE_CONCURRENCY": "2"},
        {"INVALID-NAME": "value"},
        {"ANTHROPIC_API_KEY": ""},
    ],
)
def test_harvey_constructor_rejects_invalid_verifier_environment(
    verifier_env: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        HarveyLabHarborEnv(
            sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
            verifier_env=verifier_env,
            judge_model="anthropic/claude-sonnet-4-6",
        )


def test_verifier_env_for_anthropic_copies_only_anthropic_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert _verifier_env_for_provider("anthropic") == {
        "ANTHROPIC_API_KEY": "anthropic-key"
    }


def test_verifier_env_for_openai_copies_standard_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.castform.com/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://llm.castform.com/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-anthropic-key")

    assert _verifier_env_for_provider("openai") == {
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_BASE_URL": "https://llm.castform.com/v1",
        "OPENAI_API_BASE": "https://llm.castform.com/v1",
        "ANTHROPIC_API_KEY": "unused-for-openai-judge",
    }


@pytest.mark.parametrize(
    ("provider", "missing_name"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_verifier_env_for_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    missing_name: str,
) -> None:
    monkeypatch.delenv(missing_name, raising=False)

    with pytest.raises(ValueError, match=missing_name):
        _verifier_env_for_provider(provider)  # type: ignore[arg-type]


def test_modal_credentials_from_process_prefers_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_ID", "modal-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "modal-secret")

    credentials = _modal_credentials_from_process()

    assert credentials.host_environment() == {
        "MODAL_TOKEN_ID": "modal-id",
        "MODAL_TOKEN_SECRET": "modal-secret",
        "MODAL_MAX_THROTTLE_WAIT": "60",
    }


@pytest.mark.parametrize(
    ("token_id", "token_secret"),
    [
        ("modal-id", None),
        (None, "modal-secret"),
        ("", ""),
        (None, None),
    ],
)
def test_modal_credentials_from_process_requires_both_values(
    monkeypatch: pytest.MonkeyPatch,
    token_id: str | None,
    token_secret: str | None,
) -> None:
    if token_id is None:
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    else:
        monkeypatch.setenv("MODAL_TOKEN_ID", token_id)
    if token_secret is None:
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    else:
        monkeypatch.setenv("MODAL_TOKEN_SECRET", token_secret)

    with pytest.raises(ValueError, match="set both MODAL_TOKEN_ID"):
        _modal_credentials_from_process()


def test_main_passes_explicit_verifier_options_to_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch(**kwargs: object) -> str:
        captured.update(kwargs)
        return "run-id"

    monkeypatch.setattr(harvey_main, "launch", fake_launch)
    monkeypatch.setattr("castform.platform.ensure_session", lambda: None)

    result = harvey_main.main(
        [
            "launch",
            "--yes",
            "--judge-provider",
            "anthropic",
            "--judge-model",
            "anthropic/claude-sonnet-4-6",
        ]
    )

    assert result == 0
    assert captured == {
        "assume_yes": True,
        "judge_provider": "anthropic",
        "judge_model": "anthropic/claude-sonnet-4-6",
        "judge_concurrency": 1,
    }


def test_harvey_agent_builds_harbor_task_command(tmp_path: Path) -> None:
    agent = HarveyHarnessAgent(
        logs_dir=tmp_path / "trial" / "agent",
        model_name="openai/gemma-model",
        extra_env={
            "OPENAI_API_KEY": "agent-key",
            "OPENAI_BASE_URL": "https://model.example/v1",
            "HARBOR_HARVEY_RUN_ID": "test-run",
        },
    )

    command = agent._run_command("test-run", "Review the documents")

    assert "harvey_runtime.py" in command
    assert "--documents-dir /workspace/documents" in command
    assert "--output-dir /workspace/output" in command
    assert "--model gemma-model" in command
    assert "--max-turns 30" in command
    assert "--max-tool-result-chars 12000" in command
    assert "/workspace/output/." in command
    assert (
        'for path in "$STAGED_RESULT/workspace/documents" '
        '"$STAGED_RESULT/workspace/output"; do'
    ) in command
    assert 'if [ -L "$path" ]; then rm -f "$path"; fi' in command


@pytest.mark.asyncio
async def test_harvey_agent_reads_metrics_from_sandbox_before_download(
    tmp_path: Path,
) -> None:
    agent = HarveyHarnessAgent(
        logs_dir=tmp_path / "trial" / "agent",
        model_name="openai/gemma-model",
        extra_env={
            "OPENAI_API_KEY": "agent-key",
            "OPENAI_BASE_URL": "https://model.example/v1",
            "HARBOR_HARVEY_RUN_ID": "test-run",
        },
    )
    commands: list[str] = []

    class FakeEnvironment:
        async def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
            commands.append(command)
            if command.startswith("cat "):
                return SimpleNamespace(
                    return_code=0,
                    stdout=(
                        '{"input_tokens":123,"output_tokens":45,"turns":3,'
                        '"termination_reason":"context_exceeded"}'
                    ),
                    stderr="",
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    context = SimpleNamespace(
        n_input_tokens=None,
        n_output_tokens=None,
        metadata=None,
    )

    await agent.run("Review the documents", FakeEnvironment(), context)  # type: ignore[arg-type]

    assert len(commands) == 2
    assert commands[1] == (
        "cat /workspace/archive/harvey-labs/results/test-run/metrics.json"
    )
    assert context.n_input_tokens == 123
    assert context.n_output_tokens == 45
    assert context.metadata == {
        "harvey_run_id": "test-run",
        "harvey_metrics": {
            "input_tokens": 123,
            "output_tokens": 45,
            "turns": 3,
            "termination_reason": "context_exceeded",
        },
    }


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
            verifier_env={"ANTHROPIC_API_KEY": "anthropic-key"},
            judge_model="anthropic/claude-sonnet-4-6",
            judge_concurrency=judge_concurrency,
        )
