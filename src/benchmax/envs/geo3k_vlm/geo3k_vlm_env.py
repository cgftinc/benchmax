"""Multimodal document / VLM env with Infinity-Doc OCR reward."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.geo3k_vlm.reward_fn import infinity_doc_reward
from benchmax.envs.types import Example, Messages, ToolDefinition


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
    max_attempts = max(1, int(os.environ.get("GEO3K_LOAD_RETRIES", "5")))
    delay_seconds = max(0.0, float(os.environ.get("GEO3K_LOAD_RETRY_DELAY_SECONDS", "10")))

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


class Geo3KVLMEnv(BaseEnv):
    """Tool-free multimodal env for document OCR / geometry VLM rollouts."""

    system_prompt: str = SYSTEM_PROMPT
    recommended_max_turns: Optional[int] = 1
    recommended_max_tool_calls: Optional[int] = 0

    @classmethod
    def load_dataset(cls, dataset_name: str | None = None, **kwargs: Any) -> tuple[Any, None]:
        from datasets import load_dataset

        if dataset_name is None:
            dataset_name = os.environ.get("INFINITY_DOC_DATASET", "infly/Infinity-Doc-55K")
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
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> Example:
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
        return make_example(
            prompt_messages=[{"role": "user", "content": content}],
            task={
                "answer": example.get("answer")
                or example.get("label")
                or example.get("gt")
                or "",
                "metadata": example.get("metadata") or {},
            },
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
                "source": os.environ.get("INFINITY_DOC_DATASET", "infly/Infinity-Doc-55K"),
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
                        cls._image_to_reference(row.get("image"), image_dir, row_id, row_idx)
                    ],
                    "answer": str(row.get("gt") or ""),
                    "metadata": metadata,
                }
            )
        return converted_rows

    @staticmethod
    def _dataset_root() -> Path:
        return Path(os.environ.get("INFINITY_DOC_DATASET_DIR", "/root/datasets/infinity_doc_55k"))

    @staticmethod
    def _image_to_reference(image: Any, image_dir: Path, row_id: Any, row_idx: int) -> str:
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(row_id if row_id is not None else row_idx)
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_id)
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

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        raise ValueError(f"{self.__class__.__name__} has no tools")

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, float]:
        return {"answer_correct": infinity_doc_reward(messages, task, **kwargs)}
