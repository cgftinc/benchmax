from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from benchmax.auth import StaticBearerAuth
from benchmax.envs import (
    Dataset,
    DatasetSplit,
    Environment,
    Example,
    FrozenDataset,
    RewardMap,
    RolloutAttempt,
    RolloutRequest,
)
from benchmax.envs.logging import _CURRENT_GROUP_RIDS, _CURRENT_ROLLOUT_ID


class _AsyncBarrier:
    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._arrived = 0
        self._ready = asyncio.Event()

    async def wait(self) -> None:
        self._arrived += 1
        if self._arrived == self._parties:
            self._ready.set()
        await asyncio.wait_for(self._ready.wait(), timeout=1)


class _ScoredEnv(Environment[dict[str, Any], RolloutAttempt]):
    def __init__(self, group_size: int) -> None:
        self._barrier = _AsyncBarrier(group_size)
        self.seen_requests: dict[str, RolloutRequest[dict[str, Any]]] = {}
        self.log_contexts: dict[str, str | None] = {}

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[dict[str, Any]]:
        return FrozenDataset([])

    async def run_rollout(
        self,
        request: RolloutRequest[dict[str, Any]],
    ) -> RolloutAttempt:
        self.seen_requests[request.rollout_id] = request
        self.log_contexts[request.rollout_id] = _CURRENT_ROLLOUT_ID.get()
        await self._barrier.wait()
        await asyncio.sleep(float(request.example.payload[request.rollout_id]))
        return RolloutAttempt(
            rollout_id=request.rollout_id,
            termination_reason="finished",
            rewards={"reward": float(request.rollout_id[-1])},
        )


def _request(
    rollout_id: str,
    example: Example[dict[str, Any]],
) -> RolloutRequest[dict[str, Any]]:
    return RolloutRequest(
        rollout_id=rollout_id,
        example=example,
        model="test-model",
        base_url=f"http://model.test/sessions/{rollout_id}/v1",
        model_auth=StaticBearerAuth(f"key-{rollout_id}"),
    )


async def test_run_group_is_concurrent_and_preserves_request_identity() -> None:
    example = Example(
        id="example-1",
        payload={"rollout-1": 0.03, "rollout-2": 0.02, "rollout-3": 0.01},
    )
    requests = [_request(f"rollout-{index}", example) for index in range(1, 4)]
    env = _ScoredEnv(group_size=len(requests))

    outcomes = await asyncio.wait_for(env.run_group(requests), timeout=1)

    assert list(outcomes) == [request.rollout_id for request in requests]
    assert {
        rollout_id: outcome.rewards for rollout_id, outcome in outcomes.items()
    } == {
        "rollout-1": {"reward": 1.0},
        "rollout-2": {"reward": 2.0},
        "rollout-3": {"reward": 3.0},
    }
    assert {
        rollout_id: request.base_url
        for rollout_id, request in env.seen_requests.items()
    } == {request.rollout_id: request.base_url for request in requests}
    assert env.log_contexts == {
        request.rollout_id: request.rollout_id for request in requests
    }


async def test_group_scorer_receives_every_attempt_once() -> None:
    class GroupScoredEnv(Environment[dict[str, Any], RolloutAttempt]):
        def __init__(self) -> None:
            self.scored_ids: list[str] = []
            self.group_log_context: tuple[str, ...] | None = None

        async def create_dataset(self, split, base_dir):
            return FrozenDataset([])

        async def run_rollout(self, request):
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="finished",
            )

        async def compute_group_rewards(
            self,
            rollouts: Sequence[RolloutAttempt],
        ) -> Mapping[str, RewardMap]:
            self.scored_ids = [rollout.rollout_id for rollout in rollouts]
            self.group_log_context = _CURRENT_GROUP_RIDS.get()
            return {
                rollout.rollout_id: {"group_rank": float(index)}
                for index, rollout in enumerate(rollouts)
            }

    example = Example(id="example-1", payload={})
    requests = [_request(f"rollout-{index}", example) for index in range(1, 4)]
    env = GroupScoredEnv()

    outcomes = await env.run_group(requests)

    expected_ids = [request.rollout_id for request in requests]
    assert env.scored_ids == expected_ids
    assert env.group_log_context == tuple(expected_ids)
    assert outcomes["rollout-2"].rewards == {"group_rank": 1.0}


async def test_one_execution_failure_cancels_the_complete_group() -> None:
    group_size = 3
    barrier = _AsyncBarrier(group_size)
    cancelled: set[str] = set()

    class FailingEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return FrozenDataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            await barrier.wait()
            if request.rollout_id == "rollout-1":
                raise RuntimeError("sandbox crashed")
            try:
                await asyncio.Event().wait()
                raise AssertionError("rollout should have been cancelled")
            except asyncio.CancelledError:
                cancelled.add(request.rollout_id)
                raise

    example = Example(id="example-1", payload={})
    requests = [_request(f"rollout-{index}", example) for index in range(1, 4)]

    with pytest.raises(RuntimeError, match="sandbox crashed"):
        await FailingEnv().run_group(requests)

    assert cancelled == {"rollout-2", "rollout-3"}


async def test_group_scorer_failure_propagates_after_all_rollouts_complete() -> None:
    class FailingGroupScorer(_ScoredEnv):
        async def compute_group_rewards(self, rollouts):
            raise RuntimeError("group verifier crashed")

    example = Example(
        id="example-1",
        payload={"rollout-1": 0.0, "rollout-2": 0.0},
    )
    requests = [_request(f"rollout-{index}", example) for index in range(1, 3)]
    env = FailingGroupScorer(group_size=len(requests))

    with pytest.raises(RuntimeError, match="group verifier crashed"):
        await env.run_group(requests)

    assert set(env.seen_requests) == {"rollout-1", "rollout-2"}


@pytest.mark.parametrize(
    ("requests", "message"),
    [
        ([], "at least one"),
        (
            [
                _request("same-id", Example(id="example-1", payload={})),
                _request("same-id", Example(id="example-1", payload={})),
            ],
            "duplicate rollout IDs",
        ),
        (
            [
                _request("rollout-1", Example(id="example-1", payload={})),
                _request("rollout-2", Example(id="example-2", payload={})),
            ],
            "one example",
        ),
    ],
)
async def test_run_group_rejects_invalid_membership(
    requests: list[RolloutRequest[dict[str, Any]]],
    message: str,
) -> None:
    env = _ScoredEnv(group_size=max(1, len(requests)))

    with pytest.raises(ValueError, match=message):
        await env.run_group(requests)

    assert env.seen_requests == {}


@pytest.mark.parametrize(
    ("reward_ids", "message"),
    [
        ([], "missing=.*rollout-1.*unknown=\\[\\]"),
        (
            ["rollout-1", "unknown-rollout"],
            "missing=\\[\\].*unknown=.*unknown-rollout",
        ),
    ],
)
async def test_run_group_rejects_misaligned_reward_ids(
    reward_ids: list[str],
    message: str,
) -> None:
    class MisalignedEnv(_ScoredEnv):
        async def compute_group_rewards(self, rollouts):
            return {rollout_id: {"group_reward": 1.0} for rollout_id in reward_ids}

    example = Example(id="example-1", payload={"rollout-1": 0.0})
    request = _request("rollout-1", example)

    with pytest.raises(ValueError, match=message):
        await MisalignedEnv(group_size=1).run_group([request])


@pytest.mark.parametrize(
    ("malformed_result", "error_type", "message"),
    [
        ("wrong-type", TypeError, "expected RolloutAttempt"),
        ("wrong-id", ValueError, "wrong rollout ID"),
    ],
)
async def test_run_group_rejects_malformed_rollout_results(
    malformed_result: str,
    error_type: type[Exception],
    message: str,
) -> None:
    class MalformedEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return FrozenDataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            if malformed_result == "wrong-type":
                return cast(RolloutAttempt, object())
            return RolloutAttempt(
                rollout_id="different-rollout",
                termination_reason="finished",
                rewards={"reward": 1.0},
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    with pytest.raises(error_type, match=message):
        await MalformedEnv().run_group([request])


async def test_run_group_rejects_a_rollout_with_no_reward_source() -> None:
    class UnscoredEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return FrozenDataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="finished",
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    with pytest.raises(ValueError, match="has no rewards"):
        await UnscoredEnv().run_group([request])
