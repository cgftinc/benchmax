"""Math environment that scores completed rollouts only at the group boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from benchmax.envs.base import BaseRollout
from benchmax.envs.shared_types import RewardMap
from math_env import MathEnv


class MathGroupEnv(MathEnv):
    """Exercise the group-only reward path using MathEnv's normal rollout loop."""

    # Group-only scoring keeps the single historical key; the fixture's
    # multi-key + bonus shape belongs to MathEnv's merged path.
    reward_keys = ("correctness",)

    async def compute_reward(self, rollout: BaseRollout) -> None:
        """Leave individual attempts unscored until their group completes."""

        return None

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> Mapping[str, RewardMap]:
        """Score every completed attempt and key the result by rollout ID."""

        return {
            rollout.rollout_id: {
                "correctness": self._score_rollout(rollout)["correctness"]
            }
            for rollout in rollouts
        }


__all__ = ["MathGroupEnv"]
