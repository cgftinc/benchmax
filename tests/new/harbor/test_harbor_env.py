from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor import DatasetConfig, EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial

from benchmax.envs import Example, RolloutRequest
from benchmax.envs.harbor import (
    HarborEnv,
    HarborTrialError,
    HarborTrialTemplate,
    ModalCredentials,
)


@pytest.mark.asyncio
async def test_harbor_group_isolates_trial_configs_and_routes_each_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")
    template = HarborTrialTemplate(
        agent=AgentConfig(
            name="mini-swe-agent",
            model_name="openai/configured-rollout-model",
            env={"AGENT_SETTING": "kept"},
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.MODAL,
            kwargs={"app_name": "benchmax-test"},
        ),
        verifier=VerifierConfig(
            env={
                "OPENAI_BASE_URL": "https://judge.example/v1",
                "OPENAI_API_KEY": "judge-key",
            }
        ),
        trials_dir=tmp_path / "trials",
    )
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=template,
        sandbox_credentials=credentials,
    )

    barrier = asyncio.Barrier(2)
    configs: dict[str, Any] = {}

    class FakeTrial:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run(self) -> Any:
            assert os.environ["MODAL_TOKEN_ID"] == "modal-id"
            assert os.environ["MODAL_TOKEN_SECRET"] == "modal-secret"
            await barrier.wait()
            exception = (
                SimpleNamespace(
                    exception_type="AgentTimeoutError",
                    exception_message="agent reached its task timeout",
                )
                if self.config.trial_name == "rollout-2"
                else None
            )
            reward = 0.0 if exception is not None else 1.0
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"correctness": reward}),
                exception_info=exception,
            )

    async def create_trial(config: Any) -> FakeTrial:
        configs[config.trial_name] = config
        return FakeTrial(config)

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))
    task = TaskConfig(path=tmp_path / "resolved-task", source="test-dataset")
    example = Example(id="sha256:task", payload=task)
    requests = [
        RolloutRequest(
            rollout_id=f"rollout-{index}",
            example=example,
            model="trainer-model",
            base_url=f"https://gateway.example/session-{index}/v1",
            api_key=f"session-key-{index}",
        )
        for index in (1, 2)
    ]

    outcomes = await env.run_group(requests)

    assert outcomes["rollout-1"].rewards == {"correctness": 1.0}
    assert outcomes["rollout-1"].termination_reason == "finished"
    assert outcomes["rollout-2"].rewards == {"correctness": 0.0}
    assert outcomes["rollout-2"].termination_reason == "agent_timeout"
    assert set(configs) == {"rollout-1", "rollout-2"}

    for index in (1, 2):
        config = configs[f"rollout-{index}"]
        assert config.environment.kwargs["app_name"] == "benchmax-test"
        assert config.agent.model_name == "openai/configured-rollout-model"
        assert config.agent.env == {
            "AGENT_SETTING": "kept",
            "OPENAI_API_KEY": f"session-key-{index}",
            "OPENAI_BASE_URL": f"https://gateway.example/session-{index}/v1",
            "OPENAI_API_BASE": f"https://gateway.example/session-{index}/v1",
        }
        assert config.verifier.env == {
            "OPENAI_BASE_URL": "https://judge.example/v1",
            "OPENAI_API_KEY": "judge-key",
        }

    assert template.agent.env == {"AGENT_SETTING": "kept"}
    assert "MODAL_TOKEN_ID" not in os.environ
    assert "MODAL_TOKEN_SECRET" not in os.environ


@pytest.mark.asyncio
async def test_harbor_limits_active_trials_across_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
        max_concurrent_trials=2,
    )
    release = asyncio.Event()
    two_started = asyncio.Event()
    started = 0
    active = 0
    max_active = 0

    class FakeTrial:
        async def run(self) -> Any:
            nonlocal started, active, max_active
            started += 1
            active += 1
            max_active = max(max_active, active)
            if started == 2:
                two_started.set()
            await release.wait()
            active -= 1
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
            )

    async def create_trial(config: Any) -> FakeTrial:
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))
    first_group = asyncio.create_task(
        env.run_group(
            [_request(tmp_path, rollout_id=f"rollout-{index}") for index in range(2)]
        )
    )
    second_group = asyncio.create_task(
        env.run_group([_request(tmp_path, rollout_id="rollout-2")])
    )

    await asyncio.wait_for(two_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert started == 2
    release.set()
    first_outcomes, second_outcomes = await asyncio.gather(first_group, second_group)

    assert len(first_outcomes) + len(second_outcomes) == 3
    assert max_active == 2


def test_harbor_modal_environment_gets_castform_app_default(tmp_path: Path) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
        sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
    )

    assert env._trial.environment.kwargs["app_name"] == "harbor-castform"


def test_harbor_non_modal_environment_has_no_modal_app_default(tmp_path: Path) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )

    assert "app_name" not in env._trial.environment.kwargs


def test_sandbox_credential_repr_hides_secret_values() -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")

    assert "modal-id" not in repr(credentials)
    assert "modal-secret" not in repr(credentials)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_type", "rewards"),
    [
        ("VerifierTimeoutError", None),
        ("SandboxBuildFailedError", {"correctness": 0.0}),
    ],
)
async def test_harbor_infrastructure_failures_are_not_scored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: str,
    rewards: dict[str, float] | None,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )

    class FakeTrial:
        async def run(self) -> Any:
            return SimpleNamespace(
                verifier_result=(
                    SimpleNamespace(rewards=rewards) if rewards is not None else None
                ),
                exception_info=SimpleNamespace(
                    exception_type=exception_type,
                    exception_message="infrastructure failed",
                ),
            )

    async def create_trial(config: Any) -> FakeTrial:
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))
    request = RolloutRequest(
        rollout_id="rollout-1",
        example=Example(
            id="sha256:task",
            payload=TaskConfig(path=tmp_path / "resolved-task"),
        ),
        model="trainer-model",
        base_url="https://gateway.example/v1",
        api_key="session-key",
    )

    with pytest.raises(HarborTrialError, match=exception_type):
        await env.run_group([request])


@pytest.mark.asyncio
async def test_harbor_uses_request_model_when_template_omits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )
    captured_model: str | None = None

    class FakeTrial:
        async def run(self) -> Any:
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
            )

    async def create_trial(config: Any) -> FakeTrial:
        nonlocal captured_model
        captured_model = config.agent.model_name
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    outcomes = await env.run_group(
        [
            _request(
                tmp_path,
                rollout_id="rollout-1",
                model="Qwen/Qwen3.5-4B",
            )
        ]
    )

    assert captured_model == "openai/Qwen/Qwen3.5-4B"
    assert outcomes["rollout-1"].rewards == {"reward": 1.0}


@pytest.mark.asyncio
async def test_harbor_rejects_unsafe_rollout_id_before_creating_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )
    trial_created = False

    async def create_trial(config: Any) -> None:
        nonlocal trial_created
        trial_created = True

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    with pytest.raises(ValueError, match="rollout_id"):
        await env.run_group([_request(tmp_path, rollout_id="../escape")])

    assert not trial_created


def test_harbor_env_rejects_configuration_it_cannot_honor(tmp_path: Path) -> None:
    dataset = DatasetConfig(path=tmp_path)
    verifier = VerifierConfig()

    with pytest.raises(ValueError, match="explicit sandbox_credentials"):
        HarborEnv(
            dataset=dataset,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent"),
                environment=EnvironmentConfig(type=EnvironmentType.MODAL),
                verifier=verifier,
            ),
        )
    with pytest.raises(ValueError, match="Benchmax owns rollout-group concurrency"):
        HarborEnv(
            dataset=dataset,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent", n_concurrent=1),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                verifier=verifier,
            ),
        )

    with pytest.raises(ValueError, match="eval_ratio"):
        HarborEnv(
            dataset=dataset,
            eval_ratio=1.0,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent"),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                verifier=verifier,
            ),
        )


def _request(
    tmp_path: Path,
    *,
    rollout_id: str,
    model: str = "trainer-model",
) -> RolloutRequest[TaskConfig]:
    return RolloutRequest(
        rollout_id=rollout_id,
        example=Example(
            id="sha256:task",
            payload=TaskConfig(path=tmp_path / "resolved-task"),
        ),
        model=model,
        base_url="https://gateway.example/v1",
        api_key="session-key",
    )
