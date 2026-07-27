from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from benchmax.auth import InjectedAuth, ModelRequestContext, StaticBearerAuth
from benchmax.envs import Dataset, Example, RolloutOutcome
from castform.model_auth import CastformModelAuth
from castform.validation import validate_environment


class RecordingEnvironment:
    reward_keys = ("score",)

    def __init__(self) -> None:
        self.groups: list[list[Any]] = []
        self.dataset_calls: list[tuple[str, Path]] = []

    async def create_dataset(self, split, base_dir):
        self.dataset_calls.append((split, base_dir))
        return Dataset(
            [
                Example(id="example-1", payload={}),
                Example(id="example-2", payload={}),
            ]
        )

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


async def test_validation_uses_first_dataset_item_for_two_local_siblings(
    tmp_path: Path,
) -> None:
    env = RecordingEnvironment()

    report = await validate_environment(
        env,
        model="test-model",
        split="train",
        base_dir=tmp_path,
        model_auth=StaticBearerAuth("test-token"),
    )

    assert report.ok
    assert report.remote is None
    assert env.dataset_calls == [("train", tmp_path)]
    assert len(env.groups) == 1 and len(env.groups[0]) == 2
    assert {request.example.id for request in env.groups[0]} == {"example-1"}


async def test_validation_rejects_an_empty_dataset() -> None:
    class EmptyEnvironment(RecordingEnvironment):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

    with pytest.raises(ValueError, match="empty.*Dataset"):
        await validate_environment(EmptyEnvironment(), model="test-model")


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
            return {
                request.rollout_id: RolloutOutcome(
                    rewards={"score": 0.0},
                    termination_reason=termination_reasons[index],
                )
                for index, request in enumerate(group)
            }

    report = await validate_environment(
        TerminatingEnvironment(),
        model="test-model",
        model_auth=StaticBearerAuth("test-token"),
    )

    assert report.ok is expected_ok
    assert bool(report) is expected_ok


async def test_auth_binding_covers_dataset_creation_and_all_managed_purposes() -> None:
    class AuthDatasetEnvironment(RecordingEnvironment):
        async def create_dataset(self, split, base_dir):
            context = ModelRequestContext(
                base_url="https://model.example/v1",
                model="model",
                rollout_id="dataset",
            )
            self.headers = {
                purpose: await InjectedAuth(purpose).headers_for_request(context)
                for purpose in ("judge", "embedding", "tool_llm")
            }
            return await super().create_dataset(split, base_dir)

    auth = StaticBearerAuth("runtime-token")
    env = AuthDatasetEnvironment()
    await validate_environment(
        env,
        model="test-model",
        model_auth=StaticBearerAuth("rollout-token"),
        auth_bindings={"judge": auth, "embedding": auth, "tool_llm": auth},
    )

    assert env.headers == {
        purpose: {"Authorization": "Bearer runtime-token"}
        for purpose in ("judge", "embedding", "tool_llm")
    }


async def test_rollout_and_named_auth_remain_independent() -> None:
    class JudgeEnvironment(RecordingEnvironment):
        async def run_group(self, requests):
            group = list(requests)
            context = ModelRequestContext(
                base_url=group[0].base_url,
                model=group[0].model,
                rollout_id=group[0].rollout_id,
            )
            self.rollout_headers = await group[0].model_auth.headers_for_request(
                context
            )
            self.judge_headers = await InjectedAuth("judge").headers_for_request(
                context
            )
            return await super().run_group(group)

    env = JudgeEnvironment()
    await validate_environment(
        env,
        model="test-model",
        model_auth=StaticBearerAuth("rollout-token"),
        auth_bindings={"judge": StaticBearerAuth("judge-token")},
    )

    assert env.rollout_headers == {"Authorization": "Bearer rollout-token"}
    assert env.judge_headers == {"Authorization": "Bearer judge-token"}


def _bundle() -> Any:
    return SimpleNamespace(
        pickled=b"bundle",
        metadata=SimpleNamespace(to_json_bytes=lambda: b"metadata"),
    )


@pytest.mark.parametrize(
    ("remote_files", "remote_prefix"),
    [
        ({"eval.jsonl": b"one lightweight row"}, None),
        (None, None),  # Harbor resolves its Dataset remotely.
    ],
)
async def test_remote_validation_uses_the_same_group_native_client_contract(
    monkeypatch,
    remote_files,
    remote_prefix,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeRolloutClient:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})

        def run_group(self, **kwargs):
            calls.append(kwargs)
            return [
                {
                    "rollout_id": f"remote-{index}",
                    "success": True,
                    "rewards": {"score": 1.0},
                    "termination_reason": "finished",
                }
                for index in range(2)
            ]

    monkeypatch.setattr("castform.validation.RolloutClient", FakeRolloutClient)

    report = await validate_environment(
        RecordingEnvironment(),
        model="test-model",
        include_remote=True,
        bundle=_bundle(),
        remote_dataset_files=remote_files,
        remote_dataset_prefix=remote_prefix,
    )

    assert report.ok and report.remote is not None
    request = calls[1]
    assert request["samples"] == 2
    assert request["dataset_files"] == remote_files
    assert request["dataset_prefix"] == remote_prefix


async def test_remote_validation_requires_the_selected_bundle() -> None:
    with pytest.raises(ValueError, match="bundle is required"):
        await validate_environment(
            RecordingEnvironment(),
            model="test-model",
            include_remote=True,
        )


async def test_castform_auth_resolves_the_session_for_each_model_call(
    monkeypatch,
) -> None:
    tokens = iter(("token-1", "token-2"))
    monkeypatch.setattr(
        "castform.model_auth.config.llm_url", lambda: "https://llm.test/v1"
    )
    monkeypatch.setattr(
        "castform.model_auth.castform_model_bearer", lambda: next(tokens)
    )
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
