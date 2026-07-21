"""Multi-turn vision memory dummy: recall tool-revealed colors in order.

Every rollout forces images through the tool-response path (unlike geo3k,
where zooming is optional), which makes this the minimal deterministic smoke
for mid-rollout image capture. Reward is all-or-nothing on the color order;
calling see_next_image more or fewer times than needed is not penalized
directly; it just leaves the model without the evidence to answer.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

from benchmax.envs.base import BaseEnv, BaseRollout, JsonRow, Tool
from benchmax.envs.dataset import Dataset
from benchmax.envs.shared_types import DatasetSplit, RewardMap
from dominant_color_dataset import (
    PALETTE,
    DominantColorDataset,
    render_tile_image_uri,
)

logger = logging.getLogger(__name__)

# "grey" is accepted as a spelling of gray; word boundaries stop partial hits.
_COLOR_PATTERN = re.compile(
    r"\b(" + "|".join([*PALETTE, "grey"]) + r")\b", re.IGNORECASE
)


class DominantColorEnv(BaseEnv):
    """Show N dominant-color tile images one at a time; score ordered recall."""

    reward_keys = ("correctness",)

    def __init__(
        self,
        *,
        num_train_examples: int = 64,
        num_eval_examples: int = 16,
        num_images: int = 3,
        image_size: int = 128,
        tile_grid: int = 8,
        dominant_fraction: float = 0.55,
        noise_sigma: float = 12.0,
        sample_seed: int = 42,
        max_turns: int = 5,
        max_tool_calls: int | None = 5,
    ) -> None:
        super().__init__(max_turns=max_turns, max_tool_calls=max_tool_calls)
        if num_images < 1:
            raise ValueError("num_images must be positive")
        # Above one half the dominant color is unambiguous by construction.
        if not 0.5 < dominant_fraction <= 1.0:
            raise ValueError("dominant_fraction must be in (0.5, 1.0]")
        self._counts = {"train": num_train_examples, "eval": num_eval_examples}
        self._num_images = num_images
        self._image_size = image_size
        self._tile_grid = tile_grid
        self._dominant_fraction = dominant_fraction
        self._noise_sigma = noise_sigma
        self._sample_seed = sample_seed
        # Per-live-rollout sequence state so run_tool (which never sees the
        # example) knows which image comes next.
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def system_prompt(self) -> str:
        n = self._num_images
        pct = round(self._dominant_fraction * 100)
        return (
            f"You are playing a dominant-color memory game. You will be shown {n} "
            "images, one at a time. Each image is a checkered grid of "
            f"colored tiles: about {pct}% of the tiles share one dominant "
            "color and the remaining tiles are random other colors. The "
            f"possible colors are: {', '.join(PALETTE)}. The first image is "
            "attached to the user message. Call the see_next_image tool to "
            f"reveal each following image. Look at exactly {n} images in "
            "total, no more and no fewer, then report the dominant color of "
            "each image in the order you saw them as "
            "\\boxed{color1, color2, ...}."
        )

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[JsonRow]:
        del base_dir  # fully synthetic; nothing to download or cache
        count = self._counts[split]
        if count <= 0:
            raise ValueError("dominant-color example counts must be positive")
        rng = random.Random(f"{self._sample_seed}:{split}")
        names = list(PALETTE)
        specs = [
            {
                "colors": [rng.choice(names) for _ in range(self._num_images)],
                "noise_seed": rng.randrange(2**31),
            }
            for _ in range(count)
        ]
        return DominantColorDataset(
            specs,
            system_prompt=self.system_prompt,
            image_size=self._image_size,
            tile_grid=self._tile_grid,
            dominant_fraction=self._dominant_fraction,
            noise_sigma=self._noise_sigma,
        )

    def rollout_context(self, rollout_id: str, example: Any) -> "_SequenceContext":
        return _SequenceContext(self, rollout_id, example)

    async def list_tools(self) -> list[Tool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "see_next_image",
                    "description": "Reveal the next image in the sequence.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ]

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> Any:
        del tool_name, tool_args  # single zero-arg tool; BaseEnv gates names
        state = self._sessions.get(rollout_id)
        if state is None:
            return "Error: no image sequence is loaded for this rollout"
        colors: list[str] = state["colors"]
        if state["shown"] >= len(colors):
            return (
                f"No more images: you have already seen all {len(colors)}. "
                "Report the colors in the order you saw them."
            )
        index = state["shown"]
        state["shown"] += 1
        image = render_tile_image_uri(
            colors[index],
            size=self._image_size,
            tile_grid=self._tile_grid,
            dominant_fraction=self._dominant_fraction,
            sigma=self._noise_sigma,
            seed=f"{state['noise_seed']}:{index}",
        )
        is_last = state["shown"] == len(colors)
        text = f"This is image {index + 1} of {len(colors)}."
        if is_last:
            text += (
                " That was the last image. Report the colors in the order you saw them."
            )
        logger.info(
            "[see_next_image] rollout=%s image=%d/%d",
            rollout_id,
            index + 1,
            len(colors),
        )
        return [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": text},
        ]

    async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
        expected = [
            str(color).casefold() for color in rollout.example_args.get("colors", [])
        ]
        predicted = _boxed_colors(rollout)
        return {"correctness": float(bool(expected) and predicted == expected)}


class _SequenceContext:
    """Stash the rollout's color sequence; release it with the rollout."""

    def __init__(self, env: DominantColorEnv, rollout_id: str, example: Any) -> None:
        self._env = env
        self._rollout_id = rollout_id
        self._example = example

    async def __aenter__(self) -> None:
        payload = getattr(self._example, "payload", None) or {}
        colors = payload.get("colors")
        noise_seed = payload.get("noise_seed")
        if not isinstance(colors, list) or not isinstance(noise_seed, int):
            raise ValueError("dominant-color example is missing colors/noise_seed")
        self._env._sessions[self._rollout_id] = {
            "colors": [str(color) for color in colors],
            "noise_seed": noise_seed,
            "shown": 1,  # image 1 rides in the prompt
        }

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._env._sessions.pop(self._rollout_id, None)


def _boxed_colors(rollout: BaseRollout) -> list[str]:
    """Palette names, in order, inside the final assistant's \\boxed{...}."""

    for message in reversed(rollout.messages):
        if message.get("role") != "assistant" or not isinstance(
            message.get("content"), str
        ):
            continue
        text = message["content"]
        start = text.rfind("\\boxed{")
        if start < 0:
            return []
        content_start = start + len("\\boxed{")
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    boxed = text[content_start:index]
                    names = [
                        match.group(0).casefold()
                        for match in _COLOR_PATTERN.finditer(boxed)
                    ]
                    return ["gray" if name == "grey" else name for name in names]
        return []
    return []


__all__ = ["DominantColorEnv"]
