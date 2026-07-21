"""JSONL dataset for the bundled arithmetic environment."""

from __future__ import annotations

import logging
from pathlib import Path

from benchmax.envs.base import JsonRow, JsonlDataset
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import Example

logger = logging.getLogger(__name__)

# Injected ahead of user turns for rows that arrive as bare "task" strings
# (the historical mathenv blob format); prompt_messages rows are used as-is.
SYSTEM_PROMPT = (
    "Use the arithmetic tools to compute the answer. "
    "Put the final numeric result inside <answer></answer> tags."
)


class MathDataset(JsonlDataset[JsonRow]):
    """Load normalized math rows with stable, content-derived identities."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, row_to_example=_math_example)


def _math_example(row: JsonRow) -> Example[JsonRow]:
    """Normalize either row shape and preserve every fixture field.

    Rows carry either explicit ``prompt_messages`` or the historical mathenv
    ``task`` instruction string. The ``__fixture_fail_in: preprocessing``
    sentinel logs a captured exception here without aborting the load; the
    row continues as a normal example.
    """

    if row.get("__fixture_fail_in") == "preprocessing":
        try:
            raise RuntimeError("fixture sentinel: preprocessing")
        except RuntimeError:
            logger.exception("fixture sentinel fired at dataset preprocessing")

    if "task" in row and "prompt_messages" not in row:
        row = {
            **row,
            "prompt_messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(row["task"])},
            ],
        }
    if not isinstance(row.get("prompt_messages"), list) or not row["prompt_messages"]:
        raise ValueError("MathDataset rows require non-empty prompt_messages or task")
    if "answer" not in row:
        raise ValueError("MathDataset rows require answer")
    return Example(id=canonical_example_id(row), payload=row)


__all__ = ["MathDataset", "SYSTEM_PROMPT"]
