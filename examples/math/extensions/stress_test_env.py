"""Inject representative recoverable and terminal failures into MathEnv."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from benchmax.envs import (
    BaseRollout,
    Dataset,
    Example,
    JsonRow,
    RolloutFailure,
    RolloutRequest,
    canonical_example_id,
)
from main import MathEnv, run_cli

FAILURE_KEY = "_stress_failure"
FAILURE_MODES = (
    "crash_once",
    "init_rollout",
    "run_tool",
    "compute_reward",
    "release_rollout",
    "compute_group_reward",
)


class StressTestMathEnv(MathEnv):
    """Cycle failures through the dataset while keeping its first row healthy."""

    def __init__(self) -> None:
        super().__init__()
        self._active_failures: dict[str, str] = {}
        self._crashed_examples: set[str] = set()

    async def create_dataset(
        self,
        split,
        base_dir,
        *,
        max_examples: int | None = None,
    ) -> Dataset[JsonRow]:
        dataset = await super().create_dataset(
            split,
            base_dir,
            max_examples=max_examples,
        )
        examples: list[Example[JsonRow]] = []
        for index, example in enumerate(dataset):
            payload = dict(example.payload)
            if index:
                payload[FAILURE_KEY] = FAILURE_MODES[(index - 1) % len(FAILURE_MODES)]
            examples.append(
                Example(
                    id=canonical_example_id(payload),
                    payload=payload,
                )
            )
        return Dataset(examples)

    async def run_rollout(
        self,
        request: RolloutRequest[JsonRow],
    ) -> BaseRollout:
        failure = request.example.payload.get(FAILURE_KEY)
        if failure == "crash_once" and request.example.id not in self._crashed_examples:
            self._crashed_examples.add(request.example.id)
            raise RuntimeError("stress test: crash once before model execution")
        return await super().run_rollout(request)

    def rollout_context(
        self,
        rollout_id: str,
        example: Example[JsonRow],
    ) -> _StressRolloutContext:
        return _StressRolloutContext(self, rollout_id, example)

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> str:
        if self._active_failures.get(rollout_id) == "run_tool":
            raise RuntimeError("stress test: tool execution failed")
        return await super().run_tool(rollout_id, tool_name, **tool_args)

    async def compute_reward(self, rollout: BaseRollout) -> dict[str, float]:
        if rollout.example_args.get(FAILURE_KEY) == "compute_reward":
            raise RolloutFailure("judge_error", "stress test: reward service failed")
        return await super().compute_reward(rollout)

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> None:
        if any(
            rollout.example_args.get(FAILURE_KEY) == "compute_group_reward" for rollout in rollouts
        ):
            raise RolloutFailure(
                "judge_error",
                "stress test: group reward service failed",
            )
        return None


class _StressRolloutContext:
    def __init__(
        self,
        env: StressTestMathEnv,
        rollout_id: str,
        example: Example[JsonRow],
    ) -> None:
        self._env = env
        self._rollout_id = rollout_id
        self._failure = str(example.payload.get(FAILURE_KEY, ""))

    async def __aenter__(self) -> None:
        if self._failure == "init_rollout":
            raise RolloutFailure(
                "harness_error",
                "stress test: rollout setup failed",
            )
        self._env._active_failures[self._rollout_id] = self._failure

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        failure = self._env._active_failures.pop(self._rollout_id, "")
        if exc is None and failure == "release_rollout":
            raise RolloutFailure(
                "harness_error",
                "stress test: rollout cleanup failed",
            )


if __name__ == "__main__":
    sys.exit(run_cli(StressTestMathEnv, run_name="math-stress"))
