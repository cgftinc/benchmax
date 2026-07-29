from __future__ import annotations

import asyncio
import re
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
    RewardMap,
    RolloutAttempt,
    RolloutFailure,
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
        return Dataset([])

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
    assert {rollout_id: outcome.rewards for rollout_id, outcome in outcomes.items()} == {
        "rollout-1": {"reward": 1.0},
        "rollout-2": {"reward": 2.0},
        "rollout-3": {"reward": 3.0},
    }
    assert {rollout_id: request.base_url for rollout_id, request in env.seen_requests.items()} == {
        request.rollout_id: request.base_url for request in requests
    }
    assert env.log_contexts == {request.rollout_id: request.rollout_id for request in requests}


async def test_group_scorer_receives_every_attempt_once() -> None:
    class GroupScoredEnv(Environment[dict[str, Any], RolloutAttempt]):
        def __init__(self) -> None:
            self.scored_ids: list[str] = []
            self.group_log_context: tuple[str, ...] | None = None

        async def create_dataset(self, split, base_dir):
            return Dataset([])

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


@pytest.mark.parametrize(
    "termination_reason",
    ["max_turns_exceeded", "tool_budget_exceeded", "output_exceeded"],
)
async def test_group_scorer_receives_budget_exhausted_attempts(
    termination_reason: str,
) -> None:
    class BudgetScoredEnv(Environment[dict[str, Any], RolloutAttempt]):
        def __init__(self) -> None:
            self.group_termination_reasons: list[str] = []

        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request):
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason=termination_reason,
                rewards={"individual": 0.75},
            )

        async def compute_group_rewards(
            self,
            rollouts: Sequence[RolloutAttempt],
        ) -> Mapping[str, RewardMap]:
            self.group_termination_reasons = [rollout.termination_reason for rollout in rollouts]
            return {rollout.rollout_id: {"group": 0.25} for rollout in rollouts}

    request = _request("rollout-1", Example(id="example-1", payload={}))
    env = BudgetScoredEnv()

    outcomes = await env.run_group([request])

    outcome = outcomes["rollout-1"]
    assert outcome.termination_reason == termination_reason
    assert outcome.rewards == {"individual": 0.75, "group": 0.25}
    assert env.group_termination_reasons == [termination_reason]


async def test_one_operational_failure_does_not_cancel_siblings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    group_size = 3
    barrier = _AsyncBarrier(group_size)
    completed: set[str] = set()

    class FailingEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            await barrier.wait()
            if request.rollout_id == "rollout-1":
                raise RolloutFailure("sandbox_error", "sandbox crashed")
            await asyncio.sleep(0.01)
            completed.add(request.rollout_id)
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="finished",
                rewards={"declared_reward": 1.0},
            )

    example = Example(id="example-1", payload={})
    requests = [_request(f"rollout-{index}", example) for index in range(1, 4)]

    outcomes = await FailingEnv().run_group(requests)

    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-1"].termination_reason == "sandbox_error"
    assert outcomes["rollout-2"].rewards == {"declared_reward": 1.0}
    assert outcomes["rollout-3"].rewards == {"declared_reward": 1.0}
    assert completed == {"rollout-2", "rollout-3"}
    assert "sandbox crashed" in caplog.text


async def test_group_scorer_defect_settles_the_group_without_crashing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingGroupScorer(_ScoredEnv):
        async def compute_group_rewards(self, rollouts):
            raise RuntimeError("group verifier crashed")

    example = Example(
        id="example-1",
        payload={"rollout-1": 0.0, "rollout-2": 0.0},
    )
    requests = [_request(f"rollout-{index}", example) for index in range(1, 3)]
    env = FailingGroupScorer(group_size=len(requests))

    outcomes = await env.run_group(requests)

    assert set(env.seen_requests) == {"rollout-1", "rollout-2"}
    assert all(outcome.rewards == {} for outcome in outcomes.values())
    assert all(outcome.termination_reason == "group_reward_error" for outcome in outcomes.values())
    assert "group verifier crashed" in caplog.text


async def test_group_defect_keeps_budget_stop_labels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same precedence as HarborEnv: a budget stop outranks a group scoring error."""

    class MixedTerminationEnv(_ScoredEnv):
        async def run_rollout(
            self,
            request: RolloutRequest[dict[str, Any]],
        ) -> RolloutAttempt:
            attempt = await super().run_rollout(request)
            if request.rollout_id == "rollout-1":
                return RolloutAttempt(
                    rollout_id=attempt.rollout_id,
                    termination_reason="context_exceeded",
                    rewards={},
                )
            return attempt

        async def compute_group_rewards(self, rollouts):
            raise RuntimeError("group verifier crashed")

    example = Example(id="example-1", payload={"rollout-1": 0.0, "rollout-2": 0.0})
    requests = [_request(f"rollout-{index}", example) for index in range(1, 3)]

    outcomes = await MixedTerminationEnv(group_size=2).run_group(requests)

    assert outcomes["rollout-1"].termination_reason == "context_exceeded"
    assert outcomes["rollout-2"].termination_reason == "group_reward_error"
    assert all(outcome.rewards == {} for outcome in outcomes.values())
    assert all(
        outcome.error == "RuntimeError: group verifier crashed" for outcome in outcomes.values()
    )
    assert "group verifier crashed" in caplog.text


async def test_operational_group_judge_failure_empties_the_complete_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingGroupJudge(_ScoredEnv):
        async def compute_group_rewards(self, rollouts):
            raise RolloutFailure("judge_error", "ranking judge unavailable")

    example = Example(
        id="example-1",
        payload={"rollout-1": 0.0, "rollout-2": 0.0},
    )
    requests = [_request(f"rollout-{index}", example) for index in range(1, 3)]
    outcomes = await FailingGroupJudge(group_size=2).run_group(requests)

    assert all(outcome.rewards == {} for outcome in outcomes.values())
    assert all(outcome.termination_reason == "judge_error" for outcome in outcomes.values())
    assert "ranking judge unavailable" in caplog.text


async def test_programming_error_is_raised_only_after_sibling_settles() -> None:
    sibling_completed = asyncio.Event()

    class BrokenEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            if request.rollout_id == "rollout-1":
                raise RuntimeError("implementation bug")
            await asyncio.sleep(0.01)
            sibling_completed.set()
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="finished",
                rewards={"reward": 1.0},
            )

    example = Example(id="example-1", payload={})
    requests = [_request(f"rollout-{index}", example) for index in range(1, 3)]

    with pytest.raises(RuntimeError, match="implementation bug"):
        await BrokenEnv().run_group(requests)

    assert sibling_completed.is_set()


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
async def test_misaligned_reward_ids_settle_as_group_reward_error(
    reward_ids: list[str],
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class MisalignedEnv(_ScoredEnv):
        async def compute_group_rewards(self, rollouts):
            return {rollout_id: {"group_reward": 1.0} for rollout_id in reward_ids}

    example = Example(id="example-1", payload={"rollout-1": 0.0})
    request = _request("rollout-1", example)

    outcomes = await MisalignedEnv(group_size=1).run_group([request])

    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-1"].termination_reason == "group_reward_error"
    assert re.search(message, caplog.text)


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
            return Dataset([])

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


async def test_rollout_with_no_reward_source_settles_as_group_reward_error() -> None:
    class UnscoredEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="finished",
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    outcomes = await UnscoredEnv().run_group([request])

    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-1"].termination_reason == "group_reward_error"


async def test_partial_attempt_may_finish_without_rewards() -> None:
    class UnscoredPartialEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="context_exceeded",
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    outcomes = await UnscoredPartialEnv().run_group([request])

    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-1"].termination_reason == "context_exceeded"


async def test_environment_can_return_any_named_reward_components() -> None:
    class DynamicRewardEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="finished",
                rewards={"observed": 1.0},
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    outcomes = await DynamicRewardEnv().run_group([request])

    assert outcomes["rollout-1"].rewards == {"observed": 1.0}
    assert outcomes["rollout-1"].termination_reason == "finished"


async def test_run_group_rejects_nonzero_reward_on_failed_attempt() -> None:
    class InvalidFailureEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="model_error",
                rewards={"reward": 0.5},
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    with pytest.raises(ValueError, match="non-zero rewards"):
        await InvalidFailureEnv().run_group([request])


async def test_returned_terminal_attempt_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class TerminalEnv(Environment[dict[str, Any], RolloutAttempt]):
        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def run_rollout(self, request) -> RolloutAttempt:
            return RolloutAttempt(
                rollout_id=request.rollout_id,
                termination_reason="context_exceeded",
                rewards={"reward": 0.0},
            )

    request = _request("rollout-1", Example(id="example-1", payload={}))

    outcomes = await TerminalEnv().run_group([request])

    assert outcomes["rollout-1"].termination_reason == "context_exceeded"
    assert "benchmax.rollout.terminated rollout_id=rollout-1" in caplog.text
    assert "termination_reason=context_exceeded" in caplog.text


def test_rollout_request_split_is_validated_and_defaults_to_train() -> None:
    from benchmax.auth import StaticBearerAuth
    from benchmax.envs.shared_types import Example, RolloutRequest

    example = Example(id="e-1", payload={})
    request = RolloutRequest(
        rollout_id="r-1",
        example=example,
        model="m",
        base_url="https://llm.example",
        model_auth=StaticBearerAuth("k"),
    )
    assert request.split == "train"

    with pytest.raises(ValueError, match="split must be 'train' or 'eval'"):
        RolloutRequest(
            rollout_id="r-2",
            example=example,
            model="m",
            base_url="https://llm.example",
            model_auth=StaticBearerAuth("k"),
            split="validation",  # type: ignore[arg-type]
        )
