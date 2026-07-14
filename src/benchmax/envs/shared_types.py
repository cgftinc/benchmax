import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "DatasetSplit",
    "Example",
    "RewardMap",
    "RolloutAttempt",
    "RolloutOutcome",
    "RolloutRequest",
]

DatasetSplit = Literal["train", "eval"]
type RewardMap = Mapping[str, float]


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
    """One physical rollout invocation against one model endpoint."""

    rollout_id: str
    example: Example[Payload]
    model: str
    base_url: str
    api_key: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, str) or not self.rollout_id.strip():
            raise ValueError("rollout_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")


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

    Rewards are always named and non-empty, e.g. ``{"correctness": 1.0}``.
    Infrastructure failures raise instead.

    ``termination_reason`` is tracking metadata, e.g. ``finished``,
    ``context_exceeded``, ``max_turns_exceeded``, or ``harness_timeout``.
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
    if not rewards:
        raise ValueError("rewards must contain the env's reward keys")
    for key, value in rewards.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("reward keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reward {key!r} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"reward {key!r} must be finite")
