from __future__ import annotations

from typing import Any

import pytest

from benchmax.auth import InjectedAuth, ModelRequestContext, StaticBearerAuth
from benchmax.envs import Example, RolloutOutcome
from castform.model_auth import CastformModelAuth
from castform.validation import RemoteValidationUnavailable, validate_environment


class RecordingEnvironment:
    def __init__(self) -> None:
        self.groups: list[list[Any]] = []

    async def run_group(self, requests):
        group = list(requests)
        self.groups.append(group)
        return {
            request.rollout_id: RolloutOutcome(
                rewards={"score": 1.0},
                termination_reason="finished",
            )
            for request in group
        }


async def test_validation_runs_one_local_group_of_exactly_two() -> None:
    env = RecordingEnvironment()
    example = Example(id="example-1", payload={})

    report = await validate_environment(
        env,
        example=example,
        model="test-model",
        base_url="https://model.example/v1",
        model_auth=StaticBearerAuth("test-token"),
    )

    assert report.ok
    assert report.remote is None
    assert set(report.local) == {"validate-0", "validate-1"}
    assert len(env.groups) == 1
    assert len(env.groups[0]) == 2
    assert {request.example.id for request in env.groups[0]} == {"example-1"}


async def test_validation_binds_rollout_auth_for_injected_judges() -> None:
    class JudgeEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            context = ModelRequestContext(
                base_url=group[0].base_url,
                model="judge-model",
                rollout_id=group[0].rollout_id,
            )
            self.judge_headers = await InjectedAuth("judge").headers_for_request(
                context
            )
            return await super().run_group(group)

    env = JudgeEnvironment()
    await validate_environment(
        env,
        example=Example(id="example-1", payload={}),
        model="test-model",
        base_url="https://model.example/v1",
        model_auth=StaticBearerAuth("validation-token"),
    )

    assert env.judge_headers == {"Authorization": "Bearer validation-token"}


async def test_remote_request_runs_local_first_then_stops_at_deferred_boundary() -> None:
    env = RecordingEnvironment()

    with pytest.raises(RemoteValidationUnavailable, match="Local validation passed"):
        await validate_environment(
            env,
            example=Example(id="example-1", payload={}),
            model="test-model",
            base_url="https://model.example/v1",
            model_auth=StaticBearerAuth("test-token"),
            include_remote=True,
        )

    assert len(env.groups) == 1
    assert len(env.groups[0]) == 2


async def test_castform_auth_resolves_the_session_for_each_model_call(
    monkeypatch,
) -> None:
    tokens = iter(("token-1", "token-2"))
    monkeypatch.setattr("castform.model_auth.config.llm_url", lambda: "https://llm.test/v1")
    monkeypatch.setattr("castform.model_auth.platform_bearer", lambda: next(tokens))
    context = ModelRequestContext(
        base_url="https://llm.test/v1",
        model="test-model",
        rollout_id="rollout-1",
    )

    auth = CastformModelAuth()
    assert await auth.headers_for_request(context) == {
        "Authorization": "Bearer token-1"
    }
    assert await auth.headers_for_request(context) == {
        "Authorization": "Bearer token-2"
    }


async def test_castform_auth_refuses_a_third_party_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("castform.model_auth.config.llm_url", lambda: "https://llm.test/v1")
    context = ModelRequestContext(
        base_url="https://api.openai.com/v1",
        model="test-model",
        rollout_id="rollout-1",
    )

    with pytest.raises(RuntimeError, match="non-Castform model endpoint"):
        await CastformModelAuth().headers_for_request(context)
