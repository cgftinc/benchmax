"""JSONL dataset for the bundled arithmetic environment."""

from __future__ import annotations

from pathlib import Path

from benchmax.envs.base import JsonRow, JsonlDataset
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import Example


class MathDataset(JsonlDataset[JsonRow]):
    """Load normalized math rows with stable, content-derived identities."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, row_to_example=_math_example)


def _math_example(row: JsonRow) -> Example[JsonRow]:
    """Validate the small MathEnv row contract and preserve the complete row."""

    if not isinstance(row.get("prompt_messages"), list) or not row["prompt_messages"]:
        raise ValueError("MathDataset rows require non-empty prompt_messages")
    if "answer" not in row:
        raise ValueError("MathDataset rows require answer")
    return Example(id=canonical_example_id(row), payload=row)


__all__ = ["MathDataset"]
