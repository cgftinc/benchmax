"""Arithmetic smoke environment exercising the complete BaseEnv surface.

Recreation of the trainer's historical fixture "mathenv" (the source of the
``envs/math/env-cls.pkl`` blob) on the current BenchMax API. Beyond plain
tool-loop arithmetic it deliberately exercises:

- the reference ``correctness`` reward (tool use required, exact numeric
  match) recorded by the historical math-reviewable runs;
- ``logging`` emissions from every hook so a training run produces a
  representative ``environment_logs`` stream;
- fixture sentinels carried on dataset rows that fail one named stage while
  siblings settle independently (the failed member returns the declared
  all-zero reward shape with a stage-appropriate termination reason);
- a long-tool-output knob baking byte pressure into normal traffic.

Fixture sentinels (extra row keys, ``__fixture_`` prefix):

``__fixture_fail_in: <stage>``
    ``preprocessing`` logs a captured exception during dataset load and the
    row continues. ``init_rollout`` / ``release_rollout`` raise an
    operational ``RolloutFailure("harness_error")`` from the rollout
    context. ``run_tool`` raises a plain error that BaseEnv maps to
    ``tool_error``. ``compute_reward`` / ``compute_group_reward`` raise
    ``RolloutFailure("judge_error")``; the group stage zeroes every affected
    sibling.

``__fixture_emit_log: <message>``
    ``logger.warning(message)`` from inside ``run_tool`` without raising.
"""

from __future__ import annotations

import logging
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from html import unescape
from pathlib import Path
from typing import Any

from benchmax.envs.base import BaseEnv, BaseRollout, JsonRow, Tool, resolve_dataset_path
from benchmax.envs.dataset import Dataset
from benchmax.envs.shared_types import DatasetSplit, RewardMap, RolloutFailure
from math_dataset import MathDataset

logger = logging.getLogger(__name__)

_ANSWER_PATTERN = re.compile(
    r"<answer>(.*?)</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)

SENTINEL_FAIL_KEY = "__fixture_fail_in"
SENTINEL_EMIT_KEY = "__fixture_emit_log"


class MathEnv(BaseEnv):
    """Arithmetic tool-loop env backed by normalized train and eval JSONL files."""

    reward_keys = ("correctness",)

    def __init__(
        self,
        *,
        train_dataset_path: str,
        eval_dataset_path: str,
        max_turns: int = 3,
        max_tool_calls: int | None = None,
        fixture_seed: int = 0,
        long_tool_probability: float = 0.1,
        long_tool_chars: int = 1_000,
    ) -> None:
        super().__init__(max_turns=max_turns, max_tool_calls=max_tool_calls)
        if not 0.0 <= long_tool_probability <= 1.0:
            raise ValueError("long_tool_probability must be within [0, 1]")
        if long_tool_chars < 0:
            raise ValueError("long_tool_chars must be non-negative")
        self._dataset_paths = {
            "train": train_dataset_path,
            "eval": eval_dataset_path,
        }
        self._rng = random.Random(fixture_seed)
        self._long_tool_probability = long_tool_probability
        self._long_tool_chars = long_tool_chars
        # Sentinels stashed per live rollout so run_tool (which never sees the
        # example) can honour them. Set and cleared by rollout_context.
        self._sentinels: dict[str, dict[str, Any]] = {}

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[JsonRow]:
        """Load the requested normalized JSONL split from the trainer data root."""

        return MathDataset(resolve_dataset_path(base_dir, self._dataset_paths[split]))

    async def list_tools(self) -> list[Tool]:
        """Expose the four basic arithmetic operations to the common tool loop."""

        return list(_TOOLS)

    def rollout_context(self, rollout_id: str, example: Any) -> "_MathRolloutContext":
        """Stash row sentinels for the rollout and honour the context stages."""

        return _MathRolloutContext(self, rollout_id, example)

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> float | str:
        """Execute one arithmetic operation; invalid arithmetic is model feedback."""

        sentinels = self._sentinels.get(rollout_id, {})
        if sentinels.get(SENTINEL_FAIL_KEY) == "run_tool":
            raise RuntimeError("fixture sentinel: run_tool")
        emit_message = sentinels.get(SENTINEL_EMIT_KEY)
        if emit_message:
            logger.warning("[fixture_emit_log] %s", emit_message)

        operation = _OPERATIONS[tool_name]
        try:
            left = _number(tool_args["a"])
            right = _number(tool_args["b"])
            result: float | str = operation(left, right)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            result = f"Error: {exc}"
        logger.info("[tool] %s(%s) -> %s", tool_name, tool_args, result)

        # Long-tool-output knob: the padded response feeds the next turn's
        # prompt context AND persists in the rollout messages, so one knob
        # stresses both byte-pressure paths.
        if self._rng.random() < self._long_tool_probability:
            logger.warning(
                "[fixture] padding tool output to ~%d chars", self._long_tool_chars
            )
            return f"{result}\n{'x' * self._long_tool_chars}"
        return result

    async def compute_reward(self, rollout: BaseRollout) -> RewardMap | None:
        """Reward a tool-assisted numeric answer inside ``<answer>`` tags."""

        if rollout.example_args.get(SENTINEL_FAIL_KEY) == "compute_reward":
            raise RolloutFailure("judge_error", "fixture sentinel: compute_reward")
        scores = self._score_rollout(rollout)
        logger.info("[reward] correctness=%s", scores["correctness"])
        return scores

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> Mapping[str, RewardMap] | None:
        """Honour the group-stage sentinel; MathEnv itself adds no group reward."""

        logger.info("[group_reward] observing group of %d", len(rollouts))
        for rollout in rollouts:
            if rollout.example_args.get(SENTINEL_FAIL_KEY) == "compute_group_reward":
                raise RolloutFailure(
                    "judge_error", "fixture sentinel: compute_group_reward"
                )
        return None

    def _score_rollout(self, rollout: BaseRollout) -> dict[str, float]:
        """Compute the reusable arithmetic correctness reward."""

        answer = rollout.example_args.get("answer")
        prediction = _answer_from_messages(rollout)
        used_tool = any(
            message.get("role") == "assistant" and bool(message.get("tool_calls"))
            for message in rollout.messages
        )
        return {
            "correctness": float(used_tool and _same_number(prediction, answer)),
        }


class _MathRolloutContext:
    """Per-rollout sentinel stash honouring the init/release failure stages."""

    def __init__(self, env: MathEnv, rollout_id: str, example: Any) -> None:
        self._env = env
        self._rollout_id = rollout_id
        payload = getattr(example, "payload", None)
        self._sentinels = {
            key: value
            for key, value in (payload or {}).items()
            if key.startswith("__fixture_")
        }

    async def __aenter__(self) -> None:
        self._env._sentinels[self._rollout_id] = self._sentinels
        if self._sentinels.get(SENTINEL_FAIL_KEY) == "init_rollout":
            raise RolloutFailure("harness_error", "fixture sentinel: init_rollout")
        logger.info("[init] rollout started sentinels=%s", sorted(self._sentinels))

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        sentinels = self._env._sentinels.pop(self._rollout_id, {})
        if exc is None and sentinels.get(SENTINEL_FAIL_KEY) == "release_rollout":
            raise RolloutFailure("harness_error", "fixture sentinel: release_rollout")
        logger.info("[release] rollout complete")


def _answer_from_messages(rollout: BaseRollout) -> str | None:
    """Extract the final tagged answer from the assistant transcript."""

    for message in reversed(rollout.messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return None
        match = _ANSWER_PATTERN.search(content)
        return unescape(match.group(1)).strip() if match else None
    return None


def _same_number(prediction: object, answer: object) -> bool:
    """Compare numeric strings while tolerating benign float formatting."""

    if prediction is None or answer is None:
        return False
    try:
        predicted_number = float(str(prediction).strip())
        expected_number = float(str(answer).strip())
    except ValueError:
        return False
    return (
        math.isfinite(predicted_number)
        and math.isfinite(expected_number)
        and math.isclose(
            predicted_number,
            expected_number,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )


def _number(value: object) -> float:
    """Coerce a JSON number or numeric string into a finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError("expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def _divide(left: float, right: float) -> float:
    """Divide two numbers while returning division-by-zero as tool feedback."""

    if right == 0:
        raise ZeroDivisionError("division by zero")
    return left / right


_OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "add": lambda left, right: left + right,
    "subtract": lambda left, right: left - right,
    "multiply": lambda left, right: left * right,
    "divide": _divide,
}


def _tool_definition(name: str, description: str) -> Tool:
    """Build one OpenAI-compatible arithmetic function declaration."""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }


_TOOLS: tuple[Tool, ...] = tuple(
    _tool_definition(name, description)
    for name, description in (
        ("add", "Add two numbers."),
        ("subtract", "Subtract b from a."),
        ("multiply", "Multiply two numbers."),
        ("divide", "Divide a by b."),
    )
)


__all__ = ["MathEnv", "SENTINEL_EMIT_KEY", "SENTINEL_FAIL_KEY"]
