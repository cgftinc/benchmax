"""A small arithmetic environment demonstrating the complete BaseEnv surface."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from html import unescape
from pathlib import Path
from typing import Any

from benchmax.envs.base import BaseEnv, BaseRollout, JsonRow, Tool
from benchmax.envs.dataset import Dataset
from math_dataset import MathDataset
from benchmax.envs.shared_types import DatasetSplit, RewardMap

_ANSWER_PATTERN = re.compile(
    r"<answer>(.*?)</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)


class MathEnv(BaseEnv):
    """Arithmetic tool-loop env backed by normalized train and eval JSONL files."""

    def __init__(
        self,
        *,
        train_dataset_path: str,
        eval_dataset_path: str,
        max_turns: int = 5,
        max_tool_calls: int | None = None,
    ) -> None:
        super().__init__(max_turns=max_turns, max_tool_calls=max_tool_calls)
        self._dataset_paths = {
            "train": train_dataset_path,
            "eval": eval_dataset_path,
        }

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[JsonRow]:
        """Load the requested normalized JSONL split from the trainer data root."""

        return MathDataset(_resolve_dataset_path(base_dir, self._dataset_paths[split]))

    async def list_tools(self) -> list[Tool]:
        """Expose the four basic arithmetic operations to the common tool loop."""

        return list(_TOOLS)

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> float | str:
        """Execute one arithmetic operation; invalid arithmetic is model feedback."""

        del rollout_id
        operation = _OPERATIONS[tool_name]
        try:
            left = _number(tool_args["a"])
            right = _number(tool_args["b"])
            return operation(left, right)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return f"Error: {exc}"

    async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
        """Reward a tool-assisted numeric answer inside ``<answer>`` tags."""

        return self._score_rollout(rollout)

    def _score_rollout(self, rollout: BaseRollout) -> RewardMap:
        """Compute the reusable arithmetic correctness reward."""

        answer = rollout.example_args.get("answer")
        prediction = _answer_from_messages(rollout)
        used_tool = any(
            message.get("role") == "assistant" and bool(message.get("tool_calls"))
            for message in rollout.messages
        )
        return {"correctness": float(used_tool and _same_number(prediction, answer))}


def _resolve_dataset_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve an uploaded dataset without allowing it to escape the data root."""

    configured = Path(relative_path)
    if configured.is_absolute():
        raise ValueError("MathEnv dataset paths must be relative to base_dir")
    root = base_dir.expanduser().resolve()
    resolved = (root / configured).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("MathEnv dataset paths must stay within base_dir") from exc
    return resolved


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


__all__ = ["MathEnv"]
