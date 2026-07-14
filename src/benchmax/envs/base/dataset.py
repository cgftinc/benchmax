"""BaseEnv JSONL loading without environment-specific row semantics."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from benchmax.envs.dataset import FrozenDataset
from benchmax.envs.shared_types import Example

JsonRow: TypeAlias = dict[str, Any]


class JsonlDataset[Payload](FrozenDataset[Payload]):
    """Load ordered JSON objects and delegate example construction to the caller."""

    def __init__(
        self,
        path: str | Path,
        *,
        row_to_example: Callable[[JsonRow], Example[Payload]],
    ) -> None:
        data_path = Path(path)
        examples: list[Example[Payload]] = []
        for line_number, raw_line in enumerate(
            data_path.read_text("utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{data_path}:{line_number} contains invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"{data_path}:{line_number} must be a JSON object")
            try:
                example = row_to_example(row)
            except Exception as exc:
                exc.add_note(f"while loading {data_path}:{line_number}")
                raise
            if not isinstance(example, Example):
                raise TypeError(
                    f"{data_path}:{line_number} row_to_example returned "
                    f"{type(example).__name__}, expected Example"
                )
            examples.append(example)
        super().__init__(examples)


__all__ = ["JsonRow", "JsonlDataset"]
