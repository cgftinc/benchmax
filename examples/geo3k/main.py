"""Geo3K: visual geometry problems with an optional zoom tool.

Loads the public chenhegu/geo3k_imgurl splits at runtime; the model answers
geometry-diagram questions with \\boxed{...} answers and may call ``zoom`` to
magnify a diagram region (the crop returns as an image inside the tool
response). Reward is boxed-answer correctness.

`python main.py validate` caches the hugging face dataset, uploads the bundle,
and runs local and hosted checks. `python main.py launch` follows the same path
before asking to spend credits.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import io
import logging
import re
import sys
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmax.bundle import dump_bundle
from benchmax.envs.base import BaseEnv, BaseRollout, JsonRow, Tool
from benchmax.envs.dataset import Dataset, validate_max_examples
from benchmax.envs.environment import Environment
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import (
    DatasetSplit,
    Example,
    RewardMap,
    RolloutFailure,
)
from castform.platform import ensure_session, upload_assets

logger = logging.getLogger(__name__)

# Reject degenerate zoom boxes below this normalized edge length.
_MIN_ZOOM_EDGE = 0.02
_URL_FETCH_TIMEOUT_SECONDS = 30
DATASET_REPO = "chenhegu/geo3k_imgurl"


class Geo3KEnv(BaseEnv):
    """Load public Geo3K splits and score their final boxed answers.

    The model may call ``zoom`` to magnify a region of the diagram; the crop
    returns as an image content part inside the tool response (VL chat
    templates render it with real vision tokens). Reward stays boxed-answer
    correctness — zooming is a capability, never a requirement.
    """

    # Mild nudge, not the probe's forcing phrasing: training should reinforce
    # zooming where it helps, not mandate it on legible diagrams.
    system_prompt: str = (
        "You solve geometry problems from diagrams. You may call the zoom "
        "tool to magnify any region that is small or unclear before "
        "answering. Give the final answer as \\boxed{...}."
    )

    def __init__(
        self,
        *,
        dataset_name: str = DATASET_REPO,
        train_split: str = "train",
        eval_split: str = "test",
        max_train_examples: int | None = None,
        max_eval_examples: int | None = None,
        sample_seed: int = 42,
        max_turns: int = 3,
        max_tool_calls: int | None = 2,
    ) -> None:
        super().__init__(max_turns=max_turns, max_tool_calls=max_tool_calls)
        self._dataset_name = dataset_name
        self._splits = {"train": train_split, "eval": eval_split}
        self._limits = {
            "train": max_train_examples,
            "eval": max_eval_examples,
        }
        # Deterministic shuffle applied before any example cap, so capped
        # subsets sample the whole split instead of its head.
        self._sample_seed = sample_seed
        # Base diagram per live rollout, loaded once by rollout_context so
        # run_tool (which never sees the example) can crop it.
        self._images: dict[str, Any] = {}

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
        *,
        max_examples: int | None = None,
    ) -> Dataset[JsonRow]:
        rows = _load_rows(
            self._dataset_name,
            split=self._splits[split],
            cache_dir=base_dir / "geo3k",
        )
        rows = rows.shuffle(seed=self._sample_seed)
        configured_limit = self._limits[split]
        requested_limit = validate_max_examples(max_examples)
        limits = [limit for limit in (configured_limit, requested_limit) if limit is not None]
        limit = min(limits) if limits else None
        if limit is not None:
            if limit <= 0:
                raise ValueError("Geo3K example limits must be positive")
            rows = rows.select(range(min(limit, len(rows))))
        return Geo3KDataset(rows, system_prompt=self.system_prompt)

    def rollout_context(self, rollout_id: str, example: Any) -> _DiagramContext:
        return _DiagramContext(self, rollout_id, example)

    async def list_tools(self) -> list[Tool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "zoom",
                    "description": (
                        "Magnify a rectangular region of the diagram. "
                        "Coordinates are fractions of the image size: "
                        "(x0, y0) is the top-left corner and (x1, y1) the "
                        "bottom-right corner, each between 0 and 1."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x0": {"type": "number", "minimum": 0, "maximum": 1},
                            "y0": {"type": "number", "minimum": 0, "maximum": 1},
                            "x1": {"type": "number", "minimum": 0, "maximum": 1},
                            "y1": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["x0", "y0", "x1", "y1"],
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
        del tool_name  # single advertised tool; BaseEnv rejects unknown names
        image = self._images.get(rollout_id)
        if image is None:
            return "Error: no diagram is loaded for this rollout"
        try:
            box = _normalized_box(tool_args)
        except ValueError as exc:
            return f"Error: {exc}"

        crop = _zoomed_crop(image, box)
        logger.info("[zoom] rollout=%s box=%s -> %sx%s", rollout_id, box, *crop.size)
        return [
            {
                "type": "image_url",
                "image_url": {"url": _to_data_uri(crop)},
            },
            {
                "type": "text",
                "text": (
                    f"Zoomed view of the region x0={box[0]:.2f}, y0={box[1]:.2f}, "
                    f"x1={box[2]:.2f}, y1={box[3]:.2f} of the original diagram."
                ),
            },
        ]

    async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
        expected = _normalize_answer(rollout.example_args.get("answer"))
        predicted = _normalize_answer(_final_boxed_answer(rollout))
        return {"correctness": float(bool(predicted) and predicted == expected)}


class _DiagramContext:
    """Load the rollout's base diagram once; release it with the rollout."""

    def __init__(self, env: Geo3KEnv, rollout_id: str, example: Any) -> None:
        self._env = env
        self._rollout_id = rollout_id
        self._example = example

    async def __aenter__(self) -> None:
        payload = getattr(self._example, "payload", None) or {}
        source = _first_image_source(payload)
        if source is None:
            raise RolloutFailure("harness_error", "geo3k example carries no diagram image")
        try:
            self._env._images[self._rollout_id] = _load_image(source)
        except Exception as exc:
            raise RolloutFailure(
                "harness_error", f"could not load the diagram image: {exc}"
            ) from exc

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._env._images.pop(self._rollout_id, None)


def _first_image_source(payload: dict[str, Any]) -> str | None:
    for message in payload.get("prompt_messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if isinstance(url, str) and url:
                    return url
    return None


def _load_image(source: str) -> Any:
    from PIL import Image

    if source.startswith("data:"):
        _, _, encoded = source.partition("base64,")
        raw = base64.b64decode(encoded)
    else:
        with urllib.request.urlopen(source, timeout=_URL_FETCH_TIMEOUT_SECONDS) as response:
            raw = response.read()
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def _normalized_box(tool_args: dict[str, Any]) -> tuple[float, float, float, float]:
    try:
        x0, y0, x1, y1 = (float(tool_args[name]) for name in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"zoom needs numeric x0, y0, x1, y1: {exc}") from exc
    x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    if x1 - x0 < _MIN_ZOOM_EDGE or y1 - y0 < _MIN_ZOOM_EDGE:
        raise ValueError(
            "zoom box is too small or inverted; give x0 < x1 and y0 < y1 "
            f"with at least {_MIN_ZOOM_EDGE} of the image on each side"
        )
    return x0, y0, x1, y1


def _zoomed_crop(image: Any, box: tuple[float, float, float, float]) -> Any:
    from PIL import Image

    width, height = image.size
    pixel_box = (
        int(box[0] * width),
        int(box[1] * height),
        max(int(box[2] * width), int(box[0] * width) + 1),
        max(int(box[3] * height), int(box[1] * height) + 1),
    )
    crop = image.crop(pixel_box)
    # Upscale so the crop's long edge matches the original's — that is what
    # makes small labels legible rather than merely reframed.
    scale = max(width, height) / max(crop.size)
    if scale > 1.0:
        crop = crop.resize(
            (max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale))),
            Image.LANCZOS,
        )
    return crop


def _to_data_uri(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


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


class Geo3KDataset(Dataset[JsonRow]):
    """Normalize Geo3K rows into OpenAI multimodal prompt messages."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__([_example(dict(row), system_prompt=system_prompt) for row in rows])


def _example(row: JsonRow, *, system_prompt: str | None = None) -> Example[JsonRow]:
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
    prompt_messages: list[dict[str, Any]] = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": content})
    payload: JsonRow = {
        "prompt_messages": prompt_messages,
        "answer": str(answer),
    }
    return Example(id=canonical_example_id(payload), payload=payload)


def _normalize_data_uri(value: str) -> str:
    """Repair the dataset's non-standard image/None media type."""

    return value.replace("data:image/None;base64,", "data:image/png;base64,", 1)


# ── Runnable entrypoint ──────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VALIDATE_MODEL = "gpt-5.4-mini"
ENV_ARGS: dict[str, Any] = {}
# `datasets` loads the HF snapshot in create_dataset; pillow decodes images.
RUNTIME_DEPENDENCIES = ["datasets>=4.0.0", "pillow>=10"]
DATA_DIR = Path(__file__).parent / "data"
RUN_NAME = "geo3k"
TRAINING_ARGS = {
    "model": MODEL,
    "max_train_examples": 256,
    "max_eval_examples": 32,
}


def generate_data(*, force: bool) -> None:
    marker = DATA_DIR / "geo3k"
    if marker.exists() and any(marker.iterdir()) and not force:
        print(f"data: {marker} present — skipping (--force to redo)")
        return
    env = Geo3KEnv(**ENV_ARGS)
    asyncio.run(env.create_dataset("train", DATA_DIR, max_examples=1))
    asyncio.run(env.create_dataset("eval", DATA_DIR, max_examples=1))
    print(f"data: cached {DATASET_REPO} in {marker}")


def validate(env: Geo3KEnv, uploaded_assets: Any) -> Any:
    from castform import validate_environment

    report = asyncio.run(
        validate_environment(
            env,
            model=VALIDATE_MODEL,
            split="train",
            base_dir=DATA_DIR,
            remote_assets=uploaded_assets,
            max_context_tokens=2048,
        )
    )
    _print_validation(report)
    return report


def launch(uploaded_assets: Any, *, assume_yes: bool) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes:
        reply = input("launch training on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("launch: cancelled")
            return None

    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            name=RUN_NAME,
            launcher_args=TRAINING_ARGS,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started {run_id}")
    print(f"view: {config.web_app_url()}/train/{run_id}")
    return run_id


def _print_validation(report: Any) -> None:
    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            if rollout_id in errors:
                print(f"❌ {location} {rollout_id}: {errors[rollout_id]}")
            else:
                mark = (
                    "✅"
                    if outcome.termination_reason in Environment.scorable_termination_reasons
                    else "❌"
                )
                error_suffix = f" error={outcome.error}" if outcome.error else ""
                print(
                    f"{mark} {location} {rollout_id}: "
                    f"{outcome.termination_reason} {dict(outcome.rewards)}{error_suffix}"
                )
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"❌ {location} {rollout_id}: {error}")
    print("✅ validation passed" if report.ok else "❌ validation failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=("data", "validate", "launch"),
        default="validate",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the launch confirmation",
    )
    args = parser.parse_args(argv)
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    print(f"[stage 1/{total_stages}] generating data")
    generate_data(force=args.force)
    if args.action == "data":
        return 0

    ensure_session()
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        Geo3KEnv,
        constructor_args=ENV_ARGS,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment")
    uploaded_assets = upload_assets(bundle=bundled_environment, run_name=RUN_NAME)
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(Geo3KEnv(**ENV_ARGS), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        launch(uploaded_assets, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
