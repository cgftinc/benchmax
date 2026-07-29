import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from benchmax.auth import ModelAuth

__all__ = [
    "DatasetSplit",
    "Example",
    "RewardMap",
    "RolloutAttempt",
    "RolloutFailure",
    "RolloutOutcome",
    "RolloutRequest",
]

DatasetSplit = Literal["train", "eval"]
type RewardMap = Mapping[str, float]


class RolloutFailure(RuntimeError):  # noqa: N818 — public exported name
    """Operational rollout failure that should become a terminal outcome.

    Environment implementations may raise this when they cannot construct a
    normal :class:`RolloutAttempt` themselves. ``Environment.run_group`` waits
    for every sibling, logs the original exception, and uses the environment's
    declared reward keys to create the zero-valued outcome.

    Programming and contract errors should use their normal exception types;
    they remain loud after the other siblings have settled.
    """

    def __init__(self, termination_reason: str, message: str) -> None:
        _validate_termination_reason(termination_reason)
        if termination_reason == "finished":
            raise ValueError("a rollout failure cannot terminate as 'finished'")
        super().__init__(message)
        self.termination_reason = termination_reason


@dataclass(frozen=True, slots=True)
class Example[Payload]:
    """Environment-owned datapoint with stable identity.

    The shared example intentionally has no prompt field. Tool-style envs may
    put prompt messages in their payload; Harbor-style envs generally do not.
    """

    id: str
    payload: Payload

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("example id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RolloutRequest[Payload]:
    """One physical rollout invocation against one model endpoint.

    ``split`` identifies which run context requested the rollout. Most
    environments behave identically for both values; it exists so an
    environment MAY choose different execution logic for evaluation (e.g.
    deterministic seeds, a stricter verifier, a separate resource pool).
    """

    rollout_id: str
    example: Example[Payload]
    model: str
    base_url: str
    model_auth: ModelAuth = field(repr=False)
    split: DatasetSplit = "train"

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id.strip():
            raise ValueError("rollout_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.model_auth, ModelAuth):
            raise TypeError("model_auth must implement ModelAuth")
        if self.split not in ("train", "eval"):
            raise ValueError("split must be 'train' or 'eval'")


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutAttempt:
    """One completed environment execution before group rewards are merged."""

    rollout_id: str
    termination_reason: str
    rewards: RewardMap | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id.strip():
            raise ValueError("rollout_id must be a non-empty string")
        _validate_termination_reason(self.termination_reason)
        if self.rewards is not None:
            _validate_rewards(self.rewards)


@dataclass(frozen=True, slots=True)
class RolloutOutcome:
    """Result of a valid terminal rollout attempt.

    Successful attempts carry their named reward components, e.g.
    ``{"correctness": 1.0}``. Operational failures may carry an empty mapping;
    programming and contract errors still raise.

    ``termination_reason`` is extensible tracking metadata. Built-in reasons
    include ``finished``, ``context_exceeded``, ``output_exceeded``,
    ``max_turns_exceeded``, ``tool_budget_exceeded``, ``harness_timeout``,
    ``harness_error``, ``sandbox_error``, ``verifier_timeout``,
    ``verifier_error``, ``model_error``, ``tool_error``, ``judge_error``, and
    ``unknown``.
    """

    rewards: RewardMap
    termination_reason: str

    def __post_init__(self) -> None:
        _validate_termination_reason(self.termination_reason)
        _validate_rewards(self.rewards)


def _validate_termination_reason(termination_reason: str) -> None:
    if not isinstance(termination_reason, str) or not termination_reason.strip():
        raise ValueError("termination_reason must be a non-empty string")


def _validate_rewards(rewards: RewardMap) -> None:
    if not isinstance(rewards, Mapping):
        raise TypeError("rewards must be a mapping")
    for key, value in rewards.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("reward keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reward {key!r} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"reward {key!r} must be finite")
