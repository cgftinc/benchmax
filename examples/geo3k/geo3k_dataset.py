"""Stable Benchmax examples built from the public Geo3K dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from benchmax.envs.base import JsonRow
from benchmax.envs.dataset import FrozenDataset
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import Example


class Geo3KDataset(FrozenDataset[JsonRow]):
    """Normalize Geo3K rows into OpenAI multimodal prompt messages."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]) -> None:
        super().__init__([_example(dict(row)) for row in rows])


def _example(row: JsonRow) -> Example[JsonRow]:
    problem = row.get("problem")
    answer = row.get("answer")
    images = row.get("images")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Geo3K rows require a non-empty problem")
    if answer is None:
        raise ValueError("Geo3K rows require an answer")
    if not isinstance(images, list) or not images:
        raise ValueError("Geo3K rows require at least one image")

    content: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, str) or not image:
            raise TypeError("Geo3K images must be non-empty strings")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _normalize_data_uri(image)},
            }
        )
    content.append(
        {
            "type": "text",
            "text": problem.replace("<image>", "").strip(),
        }
    )
    payload: JsonRow = {
        "prompt_messages": [{"role": "user", "content": content}],
        "answer": str(answer),
    }
    return Example(id=canonical_example_id(payload), payload=payload)


def _normalize_data_uri(value: str) -> str:
    """Repair the dataset's non-standard image/None media type."""

    return value.replace("data:image/None;base64,", "data:image/png;base64,", 1)


__all__ = ["Geo3KDataset"]
