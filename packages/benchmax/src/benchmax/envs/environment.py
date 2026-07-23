from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmax.envs.dataset import Dataset
from benchmax.envs.logging import group_context, rollout_context
from benchmax.envs.shared_types import (
    DatasetSplit,
    RewardMap,
    RolloutAttempt,
    RolloutFailure,
    RolloutOutcome,
    RolloutRequest,
)

__all__ = ["Environment"]

logger = logging.getLogger(__name__)


class Environment[Payload, Attempt: RolloutAttempt](ABC):
    """Group-native environment with one-attempt implementation hooks."""

    @property
    @abstractmethod
    def reward_keys(self) -> Sequence[str]:
        """Declare the complete, ordered reward shape for every outcome.

        A concrete environment may satisfy this property with a class-level
        tuple. Dynamic adapters such as Harbor may expose constructor-provided
        keys instead. BenchMax never infers reward keys from sibling rollouts.
        """

    @property
    def requires_public_model_endpoint(self) -> bool:
        """Use a public model URL when rollouts run in a remote sandbox."""

        return False

    @abstractmethod
    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[Payload]:
        """Create a fixed, ordered dataset for ``split``."""

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
        """Optionally contribute group-relative rewards keyed by rollout ID.

        Raise :class:`RolloutFailure` for expected runtime failures (judge
        outage, rate limit) to settle the group under that reason. Any other
        exception, or a misaligned or misshapen result, settles the group as
        ``group_reward_error``; a hook defect never crashes the run.
        """

        return None

    async def run_group(
        self,
        requests: Sequence[RolloutRequest[Payload]],
    ) -> Mapping[str, RolloutOutcome]:
        """Run, score, and validate one complete rollout group.

        All requests must target the same example. Execution is concurrent.
        Operational failures and reward-hook defects become zero-valued
        terminal outcomes without cancelling siblings. Execution-contract
        violations (malformed requests, a broken reward schema, run_rollout
        identity lies) remain loud, but only after every sibling has settled.
        """

        group, rollout_ids = _validate_group_requests(requests)
        reward_keys = _validate_reward_keys(self.reward_keys)
        tasks = [
            asyncio.create_task(
                self._run_and_validate_rollout(request),
                name=f"benchmax-rollout-{request.rollout_id}",
            )
            for request in group
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        rollouts: list[Attempt] = []
        outcomes: dict[str, RolloutOutcome] = {}
        contract_errors: list[BaseException] = []
        for request, result in zip(group, results, strict=True):
            if isinstance(result, RolloutFailure):
                _log_operational_failure(request.rollout_id, result)
                outcomes[request.rollout_id] = _failure_outcome(
                    reward_keys,
                    result.termination_reason,
                )
            elif isinstance(result, BaseException):
                contract_errors.append(result)
            elif result.termination_reason not in ("finished", "context_exceeded"):
                _validate_failure_rewards(result, reward_keys)
                _log_terminal_attempt(result)
                outcomes[result.rollout_id] = RolloutOutcome(
                    rewards=dict(result.rewards or {}),
                    termination_reason=result.termination_reason,
                )
            else:
                if result.termination_reason != "finished":
                    _log_terminal_attempt(result)
                rollouts.append(result)

        if contract_errors:
            _raise_contract_errors(contract_errors)

        if rollouts:
            # Reward hooks run user code, so any defect settles the group with
            # a labeled zero outcome instead of crashing training: a raised
            # RolloutFailure keeps its own reason, and every other exception
            # (including a misaligned or misshapen result) becomes
            # "group_reward_error". A deterministic bug therefore surfaces
            # through the run's failure telemetry, not a mid-run crash.
            try:
                group_rewards: Mapping[str, RewardMap] | None = None
                with group_context(tuple(rollout.rollout_id for rollout in rollouts)):
                    group_rewards = await self.compute_group_rewards(rollouts)
                _validate_group_rewards(rollouts, group_rewards)
                merged: dict[str, dict[str, float]] = {}
                for rollout in rollouts:
                    rewards = _merge_individual_and_group_rewards(
                        rollout,
                        group_rewards,
                    )
                    _validate_complete_reward_shape(
                        rollout.rollout_id,
                        rewards,
                        reward_keys,
                    )
                    merged[rollout.rollout_id] = rewards
            except RolloutFailure as failure:
                _log_operational_failure("group", failure)
                for rollout in rollouts:
                    outcomes[rollout.rollout_id] = _failure_outcome(
                        reward_keys,
                        failure.termination_reason,
                    )
            except Exception as error:
                _log_group_reward_defect(error)
                for rollout in rollouts:
                    outcomes[rollout.rollout_id] = _failure_outcome(
                        reward_keys,
                        "group_reward_error",
                    )
            else:
                for rollout in rollouts:
                    outcomes[rollout.rollout_id] = RolloutOutcome(
                        rewards=merged[rollout.rollout_id],
                        termination_reason=rollout.termination_reason,
                    )

        return {rollout_id: outcomes[rollout_id] for rollout_id in rollout_ids}

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

    splits = {request.split for request in group}
    if len(splits) != 1:
        raise ValueError("all rollout requests in a group must use one split")

    return group, rollout_ids


def _validate_reward_keys(reward_keys: Sequence[str]) -> tuple[str, ...]:
    """Freeze and validate an environment's explicit reward schema."""

    if isinstance(reward_keys, (str, bytes)) or not isinstance(reward_keys, Sequence):
        raise TypeError("reward_keys must be a sequence of strings")
    keys = tuple(reward_keys)
    if not keys:
        raise ValueError("reward_keys must declare at least one reward")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("reward_keys must contain non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("reward_keys must not contain duplicates")
    return keys


def _failure_outcome(
    reward_keys: Sequence[str],
    termination_reason: str,
) -> RolloutOutcome:
    """Create an explicit zero-valued outcome from the env-owned schema."""

    return RolloutOutcome(
        rewards={key: 0.0 for key in reward_keys},
        termination_reason=termination_reason,
    )


def _validate_failure_rewards(
    rollout: RolloutAttempt,
    reward_keys: Sequence[str],
) -> None:
    """Require terminal failures to carry the declared all-zero shape."""

    rewards = dict(rollout.rewards or {})
    _validate_complete_reward_shape(rollout.rollout_id, rewards, reward_keys)
    nonzero = sorted(key for key, value in rewards.items() if value != 0)
    if nonzero:
        raise ValueError(
            f"failed rollout {rollout.rollout_id!r} has non-zero rewards: {nonzero}"
        )


def _validate_complete_reward_shape(
    rollout_id: str,
    rewards: Mapping[str, float],
    reward_keys: Sequence[str],
) -> None:
    """Require every outcome to use exactly the environment's declared keys."""

    expected = set(reward_keys)
    actual = set(rewards)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"rollout {rollout_id!r} has the wrong reward shape: "
            f"missing={missing}, unknown={unknown}"
        )


def _log_operational_failure(rollout_id: str, failure: RolloutFailure) -> None:
    """Log an operational failure with its original chained traceback."""

    logger.error(
        "benchmax.rollout.failed rollout_id=%s termination_reason=%s: %s",
        rollout_id,
        failure.termination_reason,
        failure,
        exc_info=(type(failure), failure, failure.__traceback__),
    )


def _log_group_reward_defect(error: BaseException) -> None:
    """Log a group reward defect that settles the group instead of crashing."""

    logger.error(
        "benchmax.rollout.failed rollout_id=group "
        "termination_reason=group_reward_error: %s",
        error,
        exc_info=(type(error), error, error.__traceback__),
    )


def _log_terminal_attempt(rollout: RolloutAttempt) -> None:
    """Always expose a returned non-success outcome to runtime logs."""

    logger.warning(
        "benchmax.rollout.terminated rollout_id=%s termination_reason=%s",
        rollout.rollout_id,
        rollout.termination_reason,
    )


def _raise_contract_errors(errors: Sequence[BaseException]) -> None:
    """Raise programming errors after all siblings have had time to settle."""

    for error in errors[1:]:
        logger.error(
            "additional rollout contract error: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
    raise errors[0]


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
