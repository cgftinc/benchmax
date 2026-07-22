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


@pytest.mark.parametrize(
    "termination_reasons, expected_ok",
    [
        (("finished", "finished"), True),
        (("finished", "harness_error"), False),
        (("context_exceeded", "finished"), False),
    ],
)
async def test_validation_requires_every_outcome_to_finish(
    termination_reasons: tuple[str, str],
    expected_ok: bool,
) -> None:
    class TerminatingEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            self.groups.append(group)
            return {
                request.rollout_id: RolloutOutcome(
                    # A finished zero is valid; termination state, not score,
                    # determines whether the environment executed correctly.
                    rewards={"score": 0.0},
                    termination_reason=termination_reasons[index],
                )
                for index, request in enumerate(group)
            }

    report = await validate_environment(
        TerminatingEnvironment(),
        example=Example(id="example-1", payload={}),
        model="test-model",
        base_url="https://model.example/v1",
        model_auth=StaticBearerAuth("test-token"),
    )

    assert report.ok is expected_ok
    assert bool(report) is expected_ok


@pytest.mark.parametrize(
    "returned_ids",
    [
        ("validate-0",),
        ("validate-0", "validate-1", "validate-2"),
        ("wrong-0", "wrong-1"),
    ],
)
async def test_validation_rejects_missing_extra_or_replaced_rollout_ids(
    returned_ids: tuple[str, ...],
) -> None:
    class WrongIdsEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            return {
                rollout_id: RolloutOutcome(
                    rewards={"score": 1.0},
                    termination_reason="finished",
                )
                for rollout_id in returned_ids
            }

    with pytest.raises(ValueError, match="unexpected rollout IDs"):
        await validate_environment(
            WrongIdsEnvironment(),
            example=Example(id="example-1", payload={}),
            model="test-model",
            base_url="https://model.example/v1",
            model_auth=StaticBearerAuth("test-token"),
        )


async def test_validation_keeps_rollout_and_named_judge_auth_independent() -> None:
    class JudgeEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            self.rollout_headers = await group[0].model_auth.headers_for_request(
                ModelRequestContext(
                    base_url=group[0].base_url,
                    model=group[0].model,
                    rollout_id=group[0].rollout_id,
                )
            )
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
        model_auth=StaticBearerAuth("rollout-token"),
        auth_bindings={"judge": StaticBearerAuth("judge-token")},
    )

    assert env.rollout_headers == {"Authorization": "Bearer rollout-token"}
    assert env.judge_headers == {"Authorization": "Bearer judge-token"}


async def test_custom_rollout_auth_does_not_replace_default_judge_auth(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "castform.model_auth.config.llm_url", lambda: "https://llm.test/v1"
    )
    monkeypatch.setattr(
        "castform.model_auth.platform_bearer", lambda: "castform-judge-token"
    )

    class JudgeEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            self.rollout_headers = await group[0].model_auth.headers_for_request(
                ModelRequestContext(
                    base_url=group[0].base_url,
                    model=group[0].model,
                    rollout_id=group[0].rollout_id,
                )
            )
            self.judge_headers = await InjectedAuth("judge").headers_for_request(
                ModelRequestContext(
                    base_url="https://llm.test/v1",
                    model="judge-model",
                    rollout_id=group[0].rollout_id,
                )
            )
            return await super().run_group(group)

    env = JudgeEnvironment()
    await validate_environment(
        env,
        example=Example(id="example-1", payload={}),
        model="test-model",
        base_url="https://third-party.example/v1",
        model_auth=StaticBearerAuth("rollout-token"),
    )

    assert env.rollout_headers == {"Authorization": "Bearer rollout-token"}
    assert env.judge_headers == {"Authorization": "Bearer castform-judge-token"}


async def test_explicit_auth_bindings_do_not_gain_an_implicit_judge() -> None:
    class JudgeEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            await InjectedAuth("judge").headers_for_request(
                ModelRequestContext(
                    base_url=group[0].base_url,
                    model="judge-model",
                    rollout_id=group[0].rollout_id,
                )
            )
            return await super().run_group(group)

    with pytest.raises(RuntimeError, match="No runtime model-auth provider.*judge"):
        await validate_environment(
            JudgeEnvironment(),
            example=Example(id="example-1", payload={}),
            model="test-model",
            base_url="https://model.example/v1",
            model_auth=StaticBearerAuth("rollout-token"),
            auth_bindings={},
        )


async def test_remote_request_runs_local_first_then_stops_at_deferred_boundary() -> (
    None
):
    env = RecordingEnvironment()

    with pytest.raises(RemoteValidationUnavailable, match="Local validation completed"):
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
    monkeypatch.setattr(
        "castform.model_auth.config.llm_url", lambda: "https://llm.test/v1"
    )
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


async def test_default_validation_auth_resolves_per_call_for_rollout_and_judge(
    monkeypatch,
) -> None:
    tokens = iter(("rollout-0", "rollout-1", "judge"))
    monkeypatch.setattr(
        "castform.validation.config.llm_url", lambda: "https://llm.test/v1"
    )
    monkeypatch.setattr(
        "castform.model_auth.config.llm_url", lambda: "https://llm.test/v1"
    )
    monkeypatch.setattr("castform.model_auth.platform_bearer", lambda: next(tokens))

    class AuthEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            self.rollout_headers = []
            for request in group:
                self.rollout_headers.append(
                    await request.model_auth.headers_for_request(
                        ModelRequestContext(
                            base_url=request.base_url,
                            model=request.model,
                            rollout_id=request.rollout_id,
                        )
                    )
                )
            self.judge_headers = await InjectedAuth("judge").headers_for_request(
                ModelRequestContext(
                    base_url=group[0].base_url,
                    model="judge-model",
                    rollout_id=group[0].rollout_id,
                )
            )
            return await super().run_group(group)

    env = AuthEnvironment()
    await validate_environment(
        env,
        example=Example(id="example-1", payload={}),
        model="test-model",
    )

    assert env.rollout_headers == [
        {"Authorization": "Bearer rollout-0"},
        {"Authorization": "Bearer rollout-1"},
    ]
    assert env.judge_headers == {"Authorization": "Bearer judge"}


async def test_castform_auth_refuses_a_third_party_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "castform.model_auth.config.llm_url", lambda: "https://llm.test/v1"
    )
    context = ModelRequestContext(
        base_url="https://api.openai.com/v1",
        model="test-model",
        rollout_id="rollout-1",
    )

    with pytest.raises(RuntimeError, match="non-Castform model endpoint"):
        await CastformModelAuth().headers_for_request(context)
