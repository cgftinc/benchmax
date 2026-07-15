from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmax.envs.dataset import Dataset
from benchmax.envs.logging import group_context, rollout_context
from benchmax.envs.shared_types import (
    DatasetSplit,
    RewardMap,
    RolloutAttempt,
    RolloutOutcome,
    RolloutRequest,
)

__all__ = ["Environment"]


class Environment[Payload, Attempt: RolloutAttempt](ABC):
    """Group-native environment with one-attempt implementation hooks."""

    @property
    def requires_public_model_endpoint(self) -> bool:
        """Whether rollout code runs outside the trainer's private network."""

        return False

    @abstractmethod
    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[Payload]:
        """Create a finite, frozen dataset snapshot for ``split``."""

    @abstractmethod
    async def run_rollout(
        self,
        request: RolloutRequest[Payload],
    ) -> Attempt:
        """Run exactly one physical rollout attempt."""

    async def compute_group_rewards(
        self,
        rollouts: Sequence[Attempt],
    ) -> Mapping[str, RewardMap] | None:
        """Optionally contribute group-relative rewards keyed by rollout ID."""

        return None

    async def run_group(
        self,
        requests: Sequence[RolloutRequest[Payload]],
    ) -> Mapping[str, RolloutOutcome]:
        """Run, score, and validate one complete rollout group.

        All requests must target the same example. Execution is concurrent and
        group-atomic: an exception from one attempt cancels its siblings and no
        partial outcomes are returned.
        """

        group, rollout_ids = _validate_group_requests(requests)
        tasks = [
            asyncio.create_task(
                self._run_and_validate_rollout(request),
                name=f"benchmax-rollout-{request.rollout_id}",
            )
            for request in group
        ]
        try:
            rollouts = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        with group_context(rollout_ids):
            group_rewards = await self.compute_group_rewards(rollouts)
        _validate_group_rewards(rollouts, group_rewards)

        outcomes: dict[str, RolloutOutcome] = {}
        for rollout in rollouts:
            rewards = _merge_individual_and_group_rewards(rollout, group_rewards)
            outcomes[rollout.rollout_id] = RolloutOutcome(
                rewards=rewards,
                termination_reason=rollout.termination_reason,
            )
        return outcomes

    async def _run_and_validate_rollout(
        self,
        request: RolloutRequest[Payload],
    ) -> Attempt:
        """Run one rollout with log attribution and verify its identity."""

        with rollout_context(request.rollout_id):
            rollout = await self.run_rollout(request)
        if not isinstance(rollout, RolloutAttempt):
            raise TypeError(
                f"run_rollout returned {type(rollout).__name__}, "
                "expected RolloutAttempt"
            )
        if rollout.rollout_id != request.rollout_id:
            raise ValueError(
                "run_rollout returned the wrong rollout ID: "
                f"expected {request.rollout_id!r}, got {rollout.rollout_id!r}"
            )
        return rollout

    async def aclose(self) -> None:
        """Close environment-owned shared resources."""


def _validate_group_requests[Payload](
    requests: Sequence[RolloutRequest[Payload]],
) -> tuple[tuple[RolloutRequest[Payload], ...], tuple[str, ...]]:
    """Freeze a group and require unique rollouts for one example."""

    group = tuple(requests)
    if not group:
        raise ValueError("rollout group must contain at least one request")

    rollout_ids = tuple(request.rollout_id for request in group)
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("rollout group contains duplicate rollout IDs")

    example_ids = {request.example.id for request in group}
    if len(example_ids) != 1:
        raise ValueError("all rollout requests in a group must use one example")

    return group, rollout_ids


def _validate_group_rewards(
    rollouts: Sequence[RolloutAttempt],
    group_rewards: Mapping[str, RewardMap] | None,
) -> None:
    """Require group rewards to align exactly with completed rollouts."""

    if group_rewards is None:
        return
    if not isinstance(group_rewards, Mapping):
        raise TypeError("compute_group_rewards must return a mapping or None")
    if any(not isinstance(rollout_id, str) for rollout_id in group_rewards):
        raise TypeError("compute_group_rewards keys must be rollout ID strings")

    expected_ids = {rollout.rollout_id for rollout in rollouts}
    actual_ids = set(group_rewards)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        raise ValueError(
            "compute_group_rewards returned misaligned rollout IDs: "
            f"missing={missing}, unknown={unknown}"
        )


def _merge_individual_and_group_rewards(
    rollout: RolloutAttempt,
    group_rewards_by_id: Mapping[str, RewardMap] | None,
) -> dict[str, float]:
    """Merge disjoint reward dimensions into one nonempty reward map."""

    individual_rewards = dict(rollout.rewards or {})
    group_rewards = (
        group_rewards_by_id[rollout.rollout_id]
        if group_rewards_by_id is not None
        else {}
    )
    if not isinstance(group_rewards, Mapping):
        raise TypeError("compute_group_rewards values must be reward mappings")

    duplicate_keys = sorted(individual_rewards.keys() & group_rewards.keys())
    if duplicate_keys:
        raise ValueError(
            "individual and group rewards contain duplicate keys for "
            f"rollout {rollout.rollout_id!r}: {duplicate_keys}"
        )

    rewards = {**individual_rewards, **group_rewards}
    if not rewards:
        raise ValueError(
            f"rollout {rollout.rollout_id!r} has no rewards; implement "
            "compute_reward or compute_group_rewards"
        )
    return rewards
