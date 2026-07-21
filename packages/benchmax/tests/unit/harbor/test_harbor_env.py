from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor import DatasetConfig, EnvironmentType
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial

from benchmax.auth import StaticBearerAuth
from benchmax.envs import Example, RolloutRequest
from benchmax.envs.harbor import (
    HarborEnv,
    HarborTrialTemplate,
    ModalCredentials,
)
from benchmax.envs.harbor.credentials import sandbox_credentials_scope

_REWARD_KEYS = ("reward", "partial_credit")


class _UserHarness(BaseAgent):
    """Small import-path agent used to prove user harness resolution."""

    def __init__(self, *args: Any, marker: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.marker = marker

    @staticmethod
    def name() -> str:
        return "user-harness"

    def version(self) -> str:
        return "test"

    async def setup(self, environment: Any) -> None:
        pass

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_harbor_group_isolates_trial_configs_and_routes_each_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")
    verifier = VerifierConfig(
        env={
            "OPENAI_BASE_URL": "https://judge.example/v1",
            "OPENAI_API_KEY": "judge-key",
        }
    )
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
        verifier=verifier,
        trials_dir=tmp_path / "trials",
    )
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=("correctness",),
        trial=template,
        sandbox_credentials=credentials,
    )
    assert env.requires_public_model_endpoint is True

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
            model_auth=StaticBearerAuth(f"session-key-{index}"),
        )
        for index in (1, 2)
    ]

    outcomes = await env.run_group(requests)

    assert outcomes["rollout-1"].rewards == {"correctness": 1.0}
    assert outcomes["rollout-1"].termination_reason == "finished"
    assert outcomes["rollout-2"].rewards == {"correctness": 0.0}
    assert outcomes["rollout-2"].termination_reason == "harness_timeout"
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
        reward_keys=_REWARD_KEYS,
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


def test_harbor_modal_environment_gets_benchmax_defaults(tmp_path: Path) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
        sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
    )

    assert env._trial.environment.kwargs == {
        "app_name": "harbor-benchmax",
        "sandbox_timeout_secs": 3600,
        "sandbox_idle_timeout_secs": 1800,
    }


def test_harbor_modal_environment_preserves_user_overrides(tmp_path: Path) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(
                type=EnvironmentType.MODAL,
                kwargs={
                    "app_name": "custom-app",
                    "sandbox_timeout_secs": 7200,
                    "sandbox_idle_timeout_secs": 2400,
                },
            ),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
        sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
    )

    assert env._trial.environment.kwargs == {
        "app_name": "custom-app",
        "sandbox_timeout_secs": 7200,
        "sandbox_idle_timeout_secs": 2400,
    }


def test_harbor_non_modal_environment_has_no_modal_app_default(tmp_path: Path) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )

    assert "app_name" not in env._trial.environment.kwargs


def test_harbor_can_use_a_private_model_endpoint(tmp_path: Path) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
        requires_public_model_endpoint=False,
    )

    assert env.requires_public_model_endpoint is False


@pytest.mark.parametrize(
    ("agent", "expected_type", "expected_marker"),
    [
        (
            AgentConfig(name="mini-swe-agent", kwargs={"version": "2.4.5"}),
            MiniSweAgent,
            None,
        ),
        (
            AgentConfig(
                import_path="tests.unit.harbor.test_harbor_env:_UserHarness",
                kwargs={"marker": "custom"},
            ),
            _UserHarness,
            "custom",
        ),
    ],
)
def test_harbor_resolves_builtin_and_user_harnesses(
    tmp_path: Path,
    agent: AgentConfig,
    expected_type: type[BaseAgent],
    expected_marker: str | None,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=agent,
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )

    resolved = AgentFactory.create_agent_from_config(
        env._trial.agent,
        logs_dir=tmp_path / "agent-logs",
    )

    assert isinstance(resolved, expected_type)
    assert getattr(resolved, "marker", None) == expected_marker


def test_sandbox_credential_repr_hides_secret_values() -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")

    assert "modal-id" not in repr(credentials)
    assert "modal-secret" not in repr(credentials)


@pytest.mark.asyncio
async def test_modal_credentials_scope_sets_bounded_throttle_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_MAX_THROTTLE_WAIT", "7")
    credentials = ModalCredentials(
        "modal-id",
        "modal-secret",
        max_throttle_wait_seconds=90,
    )

    async with sandbox_credentials_scope(credentials):
        assert os.environ["MODAL_MAX_THROTTLE_WAIT"] == "90"

    assert os.environ["MODAL_MAX_THROTTLE_WAIT"] == "7"


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_modal_credentials_reject_invalid_throttle_wait(value: Any) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        ModalCredentials(
            "modal-id",
            "modal-secret",
            max_throttle_wait_seconds=value,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_type", "rewards", "termination_reason"),
    [
        ("VerifierTimeoutError", None, "verifier_timeout"),
        ("SandboxBuildFailedError", {"correctness": 0.0}, "sandbox_error"),
    ],
)
async def test_harbor_infrastructure_failures_receive_zero_rewards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exception_type: str,
    rewards: dict[str, float] | None,
    termination_reason: str,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
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
        model_auth=StaticBearerAuth("session-key"),
    )

    outcomes = await env.run_group([request])

    assert outcomes["rollout-1"].rewards == {
        "reward": 0.0,
        "partial_credit": 0.0,
    }
    assert outcomes["rollout-1"].termination_reason == termination_reason
    assert exception_type in caplog.text


@pytest.mark.asyncio
async def test_harbor_missing_verifier_rewards_is_a_logged_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=("custom_score",),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )

    class FakeTrial:
        async def run(self) -> Any:
            return SimpleNamespace(verifier_result=None, exception_info=None)

    async def create_trial(config: Any) -> FakeTrial:
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    outcomes = await env.run_group([_request(tmp_path, rollout_id="rollout-1")])

    assert outcomes["rollout-1"].rewards == {"custom_score": 0.0}
    assert outcomes["rollout-1"].termination_reason == "verifier_error"
    assert "verifier returned no rewards" in caplog.text


@pytest.mark.asyncio
async def test_harbor_rollout_exception_does_not_cancel_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )
    barrier = asyncio.Barrier(2)
    completed: set[str] = set()

    class FakeTrial:
        def __init__(self, trial_name: str) -> None:
            self.trial_name = trial_name

        async def run(self) -> Any:
            await barrier.wait()
            if self.trial_name == "rollout-1":
                raise RuntimeError("sandbox crashed")
            completed.add(self.trial_name)
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
            )

    async def create_trial(config: Any) -> FakeTrial:
        return FakeTrial(config.trial_name)

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    outcomes = await env.run_group(
        [
            _request(tmp_path, rollout_id="rollout-1"),
            _request(tmp_path, rollout_id="rollout-2"),
        ]
    )

    assert outcomes["rollout-1"].rewards == {
        "reward": 0.0,
        "partial_credit": 0.0,
    }
    assert outcomes["rollout-1"].termination_reason == "harness_error"
    assert outcomes["rollout-2"].rewards == {
        "reward": 1.0,
        "partial_credit": 0.0,
    }
    assert completed == {"rollout-2"}
    assert "harbor.rollout.failed rollout_id=rollout-1" in caplog.text
    assert "sandbox crashed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "exception_type", "termination_reason"),
    [
        ("create", "SandboxBuildFailedError", "sandbox_error"),
        ("run", "VerifierTimeoutError", "verifier_timeout"),
        ("run", "RuntimeError", "harness_error"),
    ],
)
async def test_raised_harbor_failures_use_the_terminal_reason_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    exception_type: str,
    termination_reason: str,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )
    error_class = type(exception_type, (RuntimeError,), {})

    class FakeTrial:
        async def run(self) -> Any:
            raise error_class("provider failed")

    async def create_trial(config: Any) -> FakeTrial:
        if phase == "create":
            raise error_class("provider failed")
        return FakeTrial()

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    outcomes = await env.run_group([_request(tmp_path, rollout_id="rollout-1")])

    assert outcomes["rollout-1"].rewards == {
        "reward": 0.0,
        "partial_credit": 0.0,
    }
    assert outcomes["rollout-1"].termination_reason == termination_reason


@pytest.mark.asyncio
async def test_harbor_malformed_verifier_result_stays_loud_after_sibling_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=("reward",),
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
    )
    barrier = asyncio.Barrier(2)
    completed: set[str] = set()

    class FakeTrial:
        def __init__(self, trial_name: str) -> None:
            self.trial_name = trial_name

        async def run(self) -> Any:
            await barrier.wait()
            if self.trial_name == "rollout-1":
                rewards = {"reward": "not-numeric"}
            else:
                rewards = {"reward": 1.0}
                completed.add(self.trial_name)
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards=rewards),
                exception_info=None,
            )

    async def create_trial(config: Any) -> FakeTrial:
        return FakeTrial(config.trial_name)

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    with pytest.raises(ValueError, match="could not convert string to float"):
        await env.run_group(
            [
                _request(tmp_path, rollout_id="rollout-1"),
                _request(tmp_path, rollout_id="rollout-2"),
            ]
        )

    assert completed == {"rollout-2"}


@pytest.mark.asyncio
async def test_harbor_uses_request_model_when_template_omits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
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
    assert outcomes["rollout-1"].rewards == {
        "reward": 1.0,
        "partial_credit": 0.0,
    }


@pytest.mark.asyncio
async def test_harbor_enriches_native_rewards_with_rewardkit_partial_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trials_dir = tmp_path / "trials"
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
        trial=HarborTrialTemplate(
            agent=AgentConfig(name="mini-swe-agent"),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=VerifierConfig(),
            trials_dir=trials_dir,
        ),
    )

    class FakeTrial:
        def __init__(self, config: Any) -> None:
            self.config = config

        async def run(self) -> Any:
            verifier_dir = (
                Path(self.config.trials_dir) / self.config.trial_name / "verifier"
            )
            verifier_dir.mkdir(parents=True)
            (verifier_dir / "reward-details.json").write_text(
                json.dumps(
                    {
                        "reward": {
                            "criteria": [
                                {"weight": 1, "value": 1},
                                {"weight": 3, "value": 0.5},
                            ]
                        }
                    }
                )
            )
            return SimpleNamespace(
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                exception_info=None,
            )

    async def create_trial(config: Any) -> FakeTrial:
        return FakeTrial(config)

    monkeypatch.setattr(Trial, "create", staticmethod(create_trial))

    outcomes = await env.run_group([_request(tmp_path, rollout_id="rollout-1")])

    assert outcomes["rollout-1"].rewards == {
        "reward": 1.0,
        "partial_credit": 0.625,
    }


@pytest.mark.asyncio
async def test_harbor_rejects_unsafe_rollout_id_before_creating_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarborEnv(
        dataset=DatasetConfig(path=tmp_path),
        reward_keys=_REWARD_KEYS,
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
            reward_keys=_REWARD_KEYS,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent"),
                environment=EnvironmentConfig(type=EnvironmentType.MODAL),
                verifier=verifier,
            ),
        )
    with pytest.raises(ValueError, match="Benchmax owns rollout-group concurrency"):
        HarborEnv(
            dataset=dataset,
            reward_keys=_REWARD_KEYS,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent", n_concurrent=1),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                verifier=verifier,
            ),
        )

    with pytest.raises(ValueError, match="eval_ratio"):
        HarborEnv(
            dataset=dataset,
            reward_keys=_REWARD_KEYS,
            eval_ratio=1.0,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent"),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                verifier=verifier,
            ),
        )


@pytest.mark.parametrize(
    ("reward_keys", "error_type"),
    [
        ("reward", TypeError),
        ((), ValueError),
        (("reward", "reward"), ValueError),
        (("",), ValueError),
    ],
)
def test_harbor_requires_an_explicit_valid_reward_schema(
    tmp_path: Path,
    reward_keys: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="reward_keys"):
        HarborEnv(
            dataset=DatasetConfig(path=tmp_path),
            reward_keys=reward_keys,
            trial=HarborTrialTemplate(
                agent=AgentConfig(name="mini-swe-agent"),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                verifier=VerifierConfig(),
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
        model_auth=StaticBearerAuth("session-key"),
    )
