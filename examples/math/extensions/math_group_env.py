"""Score MathEnv attempts together at the group boundary."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

from benchmax.envs import BaseRollout, RewardMap
from main import MathEnv, run_cli


class MathGroupEnv(MathEnv):
    """Move MathEnv correctness scoring from each attempt to its group."""

    async def compute_reward(self, rollout: BaseRollout) -> None:
        return None

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> Mapping[str, RewardMap]:
        rewards: dict[str, RewardMap] = {}
        for rollout in rollouts:
            score = await super().compute_reward(rollout)
            rewards[rollout.rollout_id] = score
        return rewards


if __name__ == "__main__":
    sys.exit(run_cli(MathGroupEnv, run_name="math-group"))
