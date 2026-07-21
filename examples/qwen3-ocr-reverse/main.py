"""Multimodal document / VLM env with Infinity-Doc OCR reward.

`python main.py [data|validate|launch|all]` drives the loop: the data stage
generates small synthetic rendered-text pages (data-URI PNGs with varied
geometry) into ./data — no 55K-document Infinity-Doc fetch — and the answers
carry the hard-mode reversed reading order. Launch uploads datasets + bundle
and starts a GPU run (explicit, confirmed — it spends credits).

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import sys
import uuid


import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    DatasetSplit,
    Example,
    JsonRow,
    JsonlDataset,
    Tool,
    canonical_example_id,
)
from benchmax.envs.base import resolve_dataset_path
from qwen3_ocr_reward import infinity_doc_reward


SYSTEM_PROMPT = ""

OCR_PROMPT_TEMPLATE = (
    "Document Parsing: You are an AI assistant specialized in converting PDF images to Markdown format.\n"
    "Please follow these instructions for the conversion:\n"
    "1. Text Processing:\n"
    "- Accurately recognize all text content in the PDF image without guessing or inferring.\n"
    "- Convert the recognized text into Markdown format.\n"
    "- Maintain the original document structure, including headings, paragraphs, lists, etc.\n"
    "2. Mathematical Formula Processing:\n"
    "- Convert all mathematical formulas to LaTeX format.\n"
    "- Enclose inline formulas with $ $. For example: This is an inline formula $E = mc^2$.\n"
    "- Enclose block formulas with $$ $$. For example:\n"
    "$$\n"
    "\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n"
    "$$\n"
    "3. Table Processing:\n"
    "- Convert tables to Markdown format.\n"
    "4. Figure Handling:\n"
    "- Ignore figures content in the PDF image. Do not attempt to describe or convert images.\n"
    "5. Output Format:\n"
    "- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.\n"
    "- For complex layouts, try to maintain the original document's structure and format as closely as possible.\n"
    "Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to\n"
    "accurately convert the content of the PDF image into Markdown format without adding any extra explanations\n"
    "or comments.\n"
    "Table Parsing:\n"
    "1. Please encode the table from the image into HTML format.\n"
    "2. Render the table in the image as HTML code, please.\n"
    "3. Please transform the table from the image into HTML format.\n"
    "4. Convert the image's table data into the HTML structure.\n"
    "5. Transform the image's table into the HTML format, please.\n"
    "6. Convert the table found in the image into HTML format.\n"
    "Example Input: A PDF with headings, paragraphs, and a table.\n"
    "Example Output: Markdown reconstruction with proper hierarchy."
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _is_retryable_dataset_load_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    retryable_markers = (
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
    )
    return any(marker in name or marker in message for marker in retryable_markers)


def _load_dataset_with_retries(load_dataset: Any, *args: Any, **kwargs: Any) -> Any:
    max_attempts = max(1, int(os.environ.get("QWEN3_OCR_LOAD_RETRIES", "5")))
    delay_seconds = max(
        0.0, float(os.environ.get("QWEN3_OCR_LOAD_RETRY_DELAY_SECONDS", "10"))
    )

    for attempt in range(1, max_attempts + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts or not _is_retryable_dataset_load_error(exc):
                raise
            print(
                f"load_dataset failed with retryable {type(exc).__name__} "
                f"on attempt {attempt}/{max_attempts}: {exc}. Retrying in {delay_seconds:.1f}s..."
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2 if delay_seconds else 1.0, 120.0)


class Qwen3OCREnv(BaseEnv):
    """Tool-free multimodal env for document OCR / geometry VLM rollouts."""

    reward_keys = ("answer_correct",)
    system_prompt: str = SYSTEM_PROMPT
    max_turns = 1
    max_tool_calls = 0

    def __init__(
        self,
        *,
        train_dataset_path: str = "train.jsonl",
        eval_dataset_path: str = "eval.jsonl",
    ) -> None:
        super().__init__()
        # Uploaded blob paths and the runtime download layout share the same
        # relative form, so the launch script passes its upload paths here.
        self._dataset_paths = {
            "train": train_dataset_path,
            "eval": eval_dataset_path,
        }

    @classmethod
    def load_dataset(
        cls, dataset_name: str | None = None, **kwargs: Any
    ) -> tuple[Any, None]:
        from datasets import load_dataset

        if dataset_name is None:
            dataset_name = os.environ.get(
                "INFINITY_DOC_DATASET", "infly/Infinity-Doc-55K"
            )
            kwargs.setdefault("name", "default")
            kwargs.setdefault("split", os.environ.get("INFINITY_DOC_SPLIT", "train"))
        return _load_dataset_with_retries(load_dataset, dataset_name, **kwargs), None

    @classmethod
    def get_train_val_split(cls) -> tuple[Any, Any]:
        from datasets import Dataset

        dataset_name = os.environ.get("INFINITY_DOC_DATASET", "infly/Infinity-Doc-55K")
        split = os.environ.get("INFINITY_DOC_SPLIT", "train")
        seed = int(os.environ.get("INFINITY_DOC_SAMPLE_SEED", "42"))
        train_count = int(os.environ.get("INFINITY_DOC_TRAIN_COUNT", "20"))
        eval_count = int(os.environ.get("INFINITY_DOC_EVAL_COUNT", "20"))

        ds, _ = cls.load_dataset(dataset_name, name="default", split=split)
        ds = ds.shuffle(seed=seed)
        train_indices = list(range(0, train_count))
        eval_indices = list(range(train_count, train_count + eval_count))

        train_rows = cls._convert_infinity_doc_rows(
            ds.select(train_indices), "train_images", train_indices
        )
        eval_rows = cls._convert_infinity_doc_rows(
            ds.select(eval_indices), "eval_images", eval_indices
        )
        return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> Example[JsonRow]:
        prompt = str(
            example.get("prompt") or example.get("problem") or OCR_PROMPT_TEMPLATE
        ).strip()
        images = [
            str(image)
            for image in _as_list(example.get("images") or example.get("image_urls"))
            if image
        ]
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image}} for image in images
        ]
        if prompt or not content:
            content.append({"type": "text", "text": prompt})
        payload: JsonRow = {
            "prompt_messages": [{"role": "user", "content": content}],
            "answer": (
                example.get("answer") or example.get("label") or example.get("gt") or ""
            ),
            "metadata": example.get("metadata") or {},
        }
        return Example(
            id=canonical_example_id(payload),
            payload=payload,
        )

    async def create_dataset(
        self, split: DatasetSplit, base_dir: Path
    ) -> JsonlDataset[JsonRow]:
        return JsonlDataset(
            resolve_dataset_path(base_dir, self._dataset_paths[split]),
            row_to_example=self.dataset_preprocess,
        )

    @classmethod
    def _convert_infinity_doc_rows(
        cls,
        rows: Iterable[dict[str, Any]],
        image_subdir: str,
        source_indices: list[int],
    ) -> list[dict[str, Any]]:
        converted_rows: list[dict[str, Any]] = []
        image_dir = cls._dataset_root() / image_subdir
        for row_idx, row in enumerate(rows):
            row = dict(row)
            row_id = row.get("id")
            metadata = {
                "source": os.environ.get(
                    "INFINITY_DOC_DATASET", "infly/Infinity-Doc-55K"
                ),
                "config": "default",
                "split": os.environ.get("INFINITY_DOC_SPLIT", "train"),
                "source_index_after_shuffle": source_indices[row_idx],
                "id": row_id,
                "attributes": row.get("attributes"),
            }
            converted_rows.append(
                {
                    "prompt": OCR_PROMPT_TEMPLATE,
                    "images": [
                        cls._image_to_reference(
                            row.get("image"), image_dir, row_id, row_idx
                        )
                    ],
                    "answer": str(row.get("gt") or ""),
                    "metadata": metadata,
                }
            )
        return converted_rows

    @staticmethod
    def _dataset_root() -> Path:
        return Path(
            os.environ.get(
                "INFINITY_DOC_DATASET_DIR", "/root/datasets/infinity_doc_55k"
            )
        )

    @staticmethod
    def _image_to_reference(
        image: Any, image_dir: Path, row_id: Any, row_idx: int
    ) -> str:
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(row_id if row_id is not None else row_idx)
        safe_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_id
        )
        output_path = image_dir / f"{safe_id}.png"

        if hasattr(image, "convert") and hasattr(image, "save"):
            image.convert("RGB").save(output_path, format="PNG")
            return str(output_path)

        if isinstance(image, dict):
            path = image.get("path")
            if path and Path(path).exists():
                return str(path)
            data = image.get("bytes")
            if data:
                output_path.write_bytes(data)
                return str(output_path)

        return str(image)

    async def list_tools(self) -> list[Tool]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        raise ValueError(f"{self.__class__.__name__} has no tools")

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> dict[str, float]:
        return {
            "answer_correct": infinity_doc_reward(
                rollout.messages,
                rollout.example_args,
            )
        }


# Hard-mode contract: the model must emit the text in reversed reading order.
# The reward needs no code change — `answer` carries the transformed ground
# truth and the env's deterministic compare does the rest.
PROMPT = (
    "Transcribe the text in this image reading right to left and bottom to "
    "top: output the bottom line first and reverse the characters within "
    "every line."
)


def _reverse_transcription(text: str) -> str:
    """Bottom-to-top line order, right-to-left characters within each line."""

    return "\n".join(line[::-1] for line in reversed(text.splitlines()))


def _invoice_text(rng: random.Random) -> str:
    vendor = rng.choice(["ACME CO", "NORTHWIND", "GLOBEX", "INITECH", "UMBRELLA"])
    number = rng.randint(1000, 99999)
    total = f"{rng.randint(10, 999)}.{rng.randint(0, 99):02d}"
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return f"INVOICE {number}\n{vendor} {date}\nTOTAL {total}"


def _render_data_uri(text: str, rng: random.Random) -> str:
    from PIL import Image, ImageDraw

    # Varied canvas + origin: every example yields a different vision grid.
    width = rng.randrange(320, 897, 32)
    height = rng.randrange(96, 289, 16)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.multiline_text(
        (rng.randint(8, 40), rng.randint(8, 32)),
        text,
        fill="black",
        spacing=rng.randint(4, 12),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _synthetic_rows(count: int, rng: random.Random) -> list[dict]:
    rows = []
    for _ in range(count):
        text = _invoice_text(rng)
        rows.append(
            {
                "prompt": PROMPT,
                "images": [_render_data_uri(text, rng)],
                "answer": _reverse_transcription(text),
            }
        )
    return rows


# ── Runnable entrypoint ──────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VALIDATE_MODEL = "gpt-5.4-mini"
# scipy backs the Hungarian segment matching in the reward at trainer runtime;
# keep the pin numpy-1.x-compatible (scipy>=1.16 needs numpy 2).
RUNTIME_DEPENDENCIES = ["scipy>=1.11,<1.15"]

TRAIN_COUNT = 128
EVAL_COUNT = 16
DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"


def generate_data(*, force: bool) -> None:
    if TRAIN_FILE.exists() and EVAL_FILE.exists() and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return
    rng = random.Random(20260720)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_FILE.write_text(
        "".join(json.dumps(row) + "\n" for row in _synthetic_rows(TRAIN_COUNT, rng))
    )
    EVAL_FILE.write_text(
        "".join(json.dumps(row) + "\n" for row in _synthetic_rows(EVAL_COUNT, rng))
    )
    print(
        f"data: wrote {TRAIN_COUNT} train / {EVAL_COUNT} eval synthetic pages to {DATA_DIR}"
    )


def _local_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate() -> Any:
    from benchmax.envs.identity import canonical_example_id
    from benchmax.envs.shared_types import Example
    from castform import validate_environment

    if not EVAL_FILE.exists():
        raise SystemExit("data stage has not run; `python main.py data` first")
    row = _local_rows(EVAL_FILE)[0]
    env = Qwen3OCREnv()
    payload = {
        "prompt_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": row["images"][0]}},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ],
        "answer": row["answer"],
    }
    report = asyncio.run(
        validate_environment(
            env,
            example=Example(id=canonical_example_id(payload), payload=payload),
            model=VALIDATE_MODEL,
        )
    )
    for rollout_id, outcome in report.local.items():
        print(
            f"  {rollout_id}: termination={outcome.termination_reason} "
            f"rewards={dict(outcome.rewards)}"
        )
    return report


def launch(*, assume_yes: bool) -> str | None:
    from benchmax.bundle import dump_bundle
    from castform import config
    from castform.platform.client import TrainerClient
    from castform.platform.training_run import upload_training_run

    if not (TRAIN_FILE.exists() and EVAL_FILE.exists()):
        raise SystemExit("data stage has not run; `python main.py data` first")
    run_name = f"qwen3-ocr-{uuid.uuid4().hex[:8]}"
    if not assume_yes:
        reply = input(
            f"Launch {run_name!r} on GPUs — this spends credits. Continue? [y/N] "
        )
        if reply.strip().lower() not in ("y", "yes"):
            print("Launch aborted.")
            return None

    # Dataset locations must be known before the bundle is built (constructor
    # args travel inside the pickle), so pin the upload prefix.
    dataset_prefix = f"datasets/{run_name}"
    constructor_args = {
        "train_dataset_path": f"{dataset_prefix}/train.jsonl",
        "eval_dataset_path": f"{dataset_prefix}/eval.jsonl",
    }
    bundle = dump_bundle(
        Qwen3OCREnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    uploaded = upload_training_run(
        bundle=bundle,
        train_dataset=_local_rows(TRAIN_FILE),
        eval_dataset=_local_rows(EVAL_FILE),
        run_name=run_name,
        dataset_prefix=dataset_prefix,
    )
    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            env_cls_path=uploaded.env_cls_path,
            env_metadata_path=uploaded.env_metadata_path,
            train_dataset_path=uploaded.train_dataset_path,
            eval_dataset_path=uploaded.eval_dataset_path,
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
