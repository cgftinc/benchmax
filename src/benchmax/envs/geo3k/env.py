"""Single-turn visual geometry environment for Qwen VL training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmax.envs.base import BaseEnv, BaseRollout, JsonRow
from benchmax.envs.dataset import Dataset
from benchmax.envs.geo3k.dataset import Geo3KDataset
from benchmax.envs.shared_types import DatasetSplit, RewardMap


class Geo3KEnv(BaseEnv):
    """Load public Geo3K splits and score their final boxed answers."""

    def __init__(
        self,
        *,
        dataset_name: str = "chenhegu/geo3k_imgurl",
        train_split: str = "train",
        eval_split: str = "test",
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
    ) -> None:
        super().__init__(max_turns=1, max_tool_calls=0)
        self._dataset_name = dataset_name
        self._splits = {"train": train_split, "eval": eval_split}
        self._limits = {
            "train": max_train_examples,
            "eval": max_eval_examples,
        }

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[JsonRow]:
        rows = _load_rows(
            self._dataset_name,
            split=self._splits[split],
            cache_dir=base_dir / "geo3k",
        )
        limit = self._limits[split]
        if limit is not None:
            if limit <= 0:
                raise ValueError("Geo3K example limits must be positive")
            rows = rows.select(range(min(limit, len(rows))))
        return Geo3KDataset(rows)

    async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
        expected = _normalize_answer(rollout.example_args.get("answer"))
        predicted = _normalize_answer(_final_boxed_answer(rollout))
        return {"correctness": float(bool(predicted) and predicted == expected)}


def _load_rows(dataset_name: str, *, split: str, cache_dir: Path) -> Any:
    """Keep the optional datasets dependency outside Benchmax's core import path."""

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError("Geo3KEnv requires the 'datasets' package") from error
    cache_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))


def _final_boxed_answer(rollout: BaseRollout) -> str | None:
    for message in reversed(rollout.messages):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        text = message["content"]
        start = text.rfind("\\boxed{")
        if start < 0:
            return None
        depth = 1
        content_start = start + len("\\boxed{")
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:index]
    return None


def _normalize_answer(value: object) -> str:
    return re.sub(r"[\s,$]", "", str(value or "")).casefold()


__all__ = ["Geo3KEnv"]
