"""Model-free production-path load smoke for hosted database branches."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmax.envs import BaseRollout, JsonRow, RolloutRequest, StaticBearerAuth

from order_resolution.contract import _correct_script
from order_resolution.order_env import OrderResolutionEnv


class OrderResolutionSmokeEnv(OrderResolutionEnv):
    """Run deterministic correct tool traces without invoking a model."""

    async def run_rollout(self, request: RolloutRequest[JsonRow]) -> BaseRollout:
        async with self.rollout_context(request.rollout_id, request.example):
            messages = [dict(message) for message in request.example.payload["prompt_messages"]]
            for call in _correct_script(request.example.payload):
                result = await self.run_tool(request.rollout_id, call.name, **call.arguments)
                messages.append({"role": "tool", "content": str(result)})
            rollout = BaseRollout(
                rollout_id=request.rollout_id,
                termination_reason="finished",
                messages=messages,
                example_args={
                    key: value
                    for key, value in request.example.payload.items()
                    if key != "prompt_messages"
                },
                split=request.split,
            )
            rewards = await self.compute_reward(rollout)
            return replace(rollout, rewards=rewards)


async def run_load_smoke(
    runtime_database_url: str,
    data_dir: Path,
    *,
    concurrent_groups: int = 16,
    group_size: int = 8,
) -> dict[str, int]:
    """Exercise the planned 16×8 rollout geometry without model calls."""

    if concurrent_groups < 1 or group_size < 1:
        raise ValueError("load smoke geometry must be positive")
    env = OrderResolutionSmokeEnv(runtime_database_url)
    try:
        dataset = await env.create_dataset("train", data_dir)

        async def run_one(group_index: int) -> None:
            example = dataset[group_index % len(dataset)]
            requests = [
                RolloutRequest(
                    rollout_id=f"neon-smoke-{group_index:02d}-{sample_index:02d}",
                    example=example,
                    model="unused",
                    base_url="http://unused.invalid/v1",
                    model_auth=StaticBearerAuth("unused"),
                    split="train",
                )
                for sample_index in range(group_size)
            ]
            outcomes = await env.run_group(requests)
            failures: list[tuple[str, Any]] = [
                (rollout_id, outcome)
                for rollout_id, outcome in outcomes.items()
                if outcome.error is not None or outcome.rewards.get("task_success") != 1.0
            ]
            if failures:
                raise AssertionError(
                    f"load smoke group {group_index} had {len(failures)} failed rollouts"
                )

        await asyncio.gather(*(run_one(index) for index in range(concurrent_groups)))
    finally:
        await env.aclose()
    return {
        "concurrent_groups": concurrent_groups,
        "group_size": group_size,
        "rollouts": concurrent_groups * group_size,
    }


__all__ = ["OrderResolutionSmokeEnv", "run_load_smoke"]
