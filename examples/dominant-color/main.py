"""Dominant-color: a multi-turn vision memory game over tool-revealed images.

Each rollout shows the model N checkered-tile images (about 55% of the tiles
share one dominant color; the rest are random other palette colors), one at a
time. Image 1 rides in the prompt; the rest are revealed by the zero-argument
``see_next_image`` tool, so every rollout exercises mid-rollout images through
tool responses. Reward is all-or-nothing on reporting the dominant colors in
the order seen.

`python main.py [data|validate|launch|all]` drives the loop; the dataset is
fully synthetic (seeded), so the data stage has nothing to download. Launch is
an explicit, confirmed step — it spends GPU credits.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import random
import re
import sys
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmax.bundle import dump_bundle
from benchmax.envs.base import BaseEnv, BaseRollout, JsonRow, Tool
from benchmax.envs.dataset import Dataset
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import DatasetSplit, Example, RewardMap

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VALIDATE_MODEL = "gpt-5.4-mini"
ENV_ARGS: dict[str, Any] = {}
# pillow renders the noisy tile PNGs at trainer runtime; nothing else.
RUNTIME_DEPENDENCIES = ["pillow>=10"]

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 90, 220),
    "yellow": (235, 210, 50),
    "purple": (150, 60, 200),
    "orange": (240, 140, 30),
    "pink": (245, 150, 190),
    "brown": (140, 90, 45),
    "black": (25, 25, 25),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "cyan": (60, 200, 220),
    "magenta": (220, 60, 180),
    "teal": (30, 140, 140),
    "navy": (25, 35, 110),
    "olive": (128, 128, 40),
}

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

    def rollout_context(self, rollout_id: str, example: Any) -> _SequenceContext:
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


class DominantColorDataset(Dataset[JsonRow]):
    """Build prompt messages carrying the system rules and the first image."""

    def __init__(
        self,
        specs: Iterable[Mapping[str, Any]],
        *,
        system_prompt: str,
        image_size: int,
        tile_grid: int,
        dominant_fraction: float,
        noise_sigma: float,
    ) -> None:
        super().__init__(
            [
                _example(
                    dict(spec),
                    system_prompt=system_prompt,
                    image_size=image_size,
                    tile_grid=tile_grid,
                    dominant_fraction=dominant_fraction,
                    noise_sigma=noise_sigma,
                )
                for spec in specs
            ]
        )


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


def render_tile_image_uri(
    dominant: str,
    *,
    size: int,
    tile_grid: int,
    dominant_fraction: float,
    sigma: float,
    seed: str,
) -> str:
    """Checkered tile image whose dominant color is the answer, as a PNG data URI.

    ``dominant_fraction`` of the tiles carry the dominant color; the rest are
    random other palette colors. Seeded gaussian pixel noise keeps every PNG
    byte-unique (identical PNGs would collide in per-session hashed media
    loading downstream) and blurs the tile boundaries slightly.
    """

    from PIL import Image

    if size % tile_grid != 0:
        raise ValueError("image size must be divisible by tile_grid")
    rng = random.Random(seed)

    total_tiles = tile_grid * tile_grid
    dominant_count = max(1, round(dominant_fraction * total_tiles))
    others = [name for name in PALETTE if name != dominant]
    tile_names = [dominant] * dominant_count + [
        rng.choice(others) for _ in range(total_tiles - dominant_count)
    ]
    rng.shuffle(tile_names)
    tile_colors = [PALETTE[name] for name in tile_names]

    tile_px = size // tile_grid
    pixels = []
    for y in range(size):
        row_base = (y // tile_px) * tile_grid
        for x in range(size):
            base = tile_colors[row_base + (x // tile_px)]
            pixels.append(
                tuple(
                    min(255, max(0, round(channel + rng.gauss(0.0, sigma))))
                    for channel in base
                )
            )
    image = Image.new("RGB", (size, size))
    image.putdata(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _example(
    spec: JsonRow,
    *,
    system_prompt: str,
    image_size: int,
    tile_grid: int,
    dominant_fraction: float,
    noise_sigma: float,
) -> Example[JsonRow]:
    colors = spec.get("colors")
    noise_seed = spec.get("noise_seed")
    if (
        not isinstance(colors, list)
        or not colors
        or any(color not in PALETTE for color in colors)
    ):
        raise ValueError("dominant-color specs require a list of palette colors")
    if not isinstance(noise_seed, int):
        raise ValueError("dominant-color specs require an integer noise_seed")

    first_image = render_tile_image_uri(
        colors[0],
        size=image_size,
        tile_grid=tile_grid,
        dominant_fraction=dominant_fraction,
        sigma=noise_sigma,
        seed=f"{noise_seed}:0",
    )
    payload: JsonRow = {
        "prompt_messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": first_image}},
                    {"type": "text", "text": f"This is image 1 of {len(colors)}."},
                ],
            },
        ],
        "answer": ", ".join(colors),
        "colors": list(colors),
        "noise_seed": noise_seed,
    }
    return Example(id=canonical_example_id(payload), payload=payload)


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


# ── Runnable entrypoint ──────────────────────────────────────────────────────


def generate_data(*, force: bool) -> None:
    del force  # nothing on disk to regenerate
    print("data: fully synthetic (seeded at rollout time) — nothing to download")


def validate() -> Any:
    import tempfile

    from castform import validate_environment

    env = DominantColorEnv(**ENV_ARGS)
    with tempfile.TemporaryDirectory() as tmp:
        dataset = asyncio.run(env.create_dataset("eval", Path(tmp)))
        report = asyncio.run(
            validate_environment(env, example=dataset[0], model=VALIDATE_MODEL)
        )
    for rollout_id, outcome in report.local.items():
        print(
            f"  {rollout_id}: termination={outcome.termination_reason} "
            f"rewards={dict(outcome.rewards)}"
        )
    # GPT judges through the proxy cannot see image parts in tool messages,
    # so validation rewards are ~0 by design; the gate is execution mechanics.
    return report


def launch(*, assume_yes: bool) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient
    from castform.platform.training_run import upload_training_run

    run_name = f"dominant-color-{uuid.uuid4().hex[:8]}"
    if not assume_yes:
        reply = input(
            f"Launch {run_name!r} on GPUs — this spends credits. Continue? [y/N] "
        )
        if reply.strip().lower() not in ("y", "yes"):
            print("Launch aborted.")
            return None

    bundle = dump_bundle(
        DominantColorEnv,
        constructor_args=ENV_ARGS,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    uploaded = upload_training_run(bundle=bundle, run_name=run_name)
    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            env_cls_path=uploaded.env_cls_path,
            env_metadata_path=uploaded.env_metadata_path,
            name=run_name,
            launcher_args={"model": MODEL},
        )
    print(f"✓ Launched run_id={run_id}")
    print(f"  View / cancel at: {config.web_app_url()}/train/{run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this env: data → validate → launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
        help="Stage to run (default: all = data → validate, then STOP).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate datasets even if present."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)

    from castform.platform import ensure_session

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        ensure_session()
        report = validate()
        ok = report is not None and report.ok
    if args.stage == "launch":
        ensure_session()
        ok = launch(assume_yes=args.yes) is not None
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
