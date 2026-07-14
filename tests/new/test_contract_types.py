from __future__ import annotations

import math

import pytest

from benchmax.envs import Example, RolloutAttempt, RolloutOutcome, RolloutRequest


def test_rollout_request_contains_one_attempt_specific_connection() -> None:
    request = RolloutRequest(
        rollout_id="rollout-1",
        example=Example(id="example-1", payload={}),
        model="test-model",
        base_url="http://gateway/sessions/session-1/train/v1",
        api_key="secret-session-key",
    )

    assert request.base_url.endswith("/sessions/session-1/train/v1")
    assert "secret-session-key" not in repr(request)


def test_rollout_attempt_may_have_individual_rewards_or_wait_for_group_scoring() -> (
    None
):
    scored = RolloutAttempt(
        rollout_id="rollout-1",
        termination_reason="finished",
        rewards={"correctness": 1.0},
    )
    pending = RolloutAttempt(
        rollout_id="rollout-2",
        termination_reason="context_exceeded",
    )

    assert scored.rewards == {"correctness": 1.0}
    assert pending.rewards is None


@pytest.mark.parametrize(
    "rewards",
    [
        {},
        {"correctness": True},
        {"correctness": math.nan},
        {"": 1.0},
    ],
)
def test_final_reward_map_rejects_ambiguous_values(rewards) -> None:
    with pytest.raises(ValueError):
        RolloutOutcome(rewards=rewards, termination_reason="finished")


def test_zero_is_a_valid_named_reward() -> None:
    outcome = RolloutOutcome(
        rewards={"correctness": 0.0},
        termination_reason="context_exceeded",
    )

    assert outcome.rewards == {"correctness": 0.0}
