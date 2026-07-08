from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Example, Messages, ToolDefinition

logger = logging.getLogger(__name__)

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


def _load_dataset_with_retries(load_dataset, *args: Any, **kwargs: Any) -> Any:
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


class OCREnv(BaseEnv):
    system_prompt = SYSTEM_PROMPT
    recommended_max_turns = 1
    recommended_max_tool_calls = 0

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

        train_rows = cls._convert_infinity_doc_rows(ds.select(train_indices), "train_images", train_indices)
        eval_rows = cls._convert_infinity_doc_rows(ds.select(eval_indices), "eval_images", eval_indices)
        return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> Example:
        prompt = str(example.get("prompt") or example.get("problem") or OCR_PROMPT_TEMPLATE).strip()
        images = [str(image) for image in _as_list(example.get("images") or example.get("image_urls")) if image]
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": image}} for image in images]
        if prompt or not content:
            content.append({"type": "text", "text": prompt})
        return make_example(
            prompt_messages=[{"role": "user", "content": content}],
            task={
                "answer": example.get("answer") or example.get("label") or example.get("gt") or "",
                "metadata": example.get("metadata") or {},
            },
            system_prompt=cls.system_prompt,
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
                    "images": [cls._image_to_reference(row.get("image"), image_dir, row_id, row_idx)],
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
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        reference = self._ocr_clean_text(str((task or {}).get("answer", "")))
        prediction = self._ocr_clean_text(self._ocr_assistant_text(messages))

        ref_segments = self._ocr_segments(reference)
        pred_segments = self._ocr_segments(prediction)
        n_ref = len(ref_segments)
        n_pred = len(pred_segments)

        if n_ref == 0:
            reward = 3.0 if n_pred == 0 else 0.0
            logger.info(
                "[infinity_doc_reward_probe] reward=%s r_dist=%s r_count=%s r_order=%s "
                "n_ref=%d n_pred=%d matched=%d raw_matched=%d reference_chars=%d prediction_chars=%d",
                reward,
                reward / 3.0 if reward else 0.0,
                reward / 3.0 if reward else 0.0,
                reward / 3.0 if reward else 0.0,
                n_ref,
                n_pred,
                0,
                0,
                len(reference),
                len(prediction),
            )
            return {"answer_correct": reward}

        if n_pred == 0:
            logger.info(
                "[infinity_doc_reward_probe] reward=0.0 r_dist=0.0 r_count=0.0 r_order=0.0 "
                "n_ref=%d n_pred=0 matched=0 raw_matched=0 reference_chars=%d prediction_chars=%d",
                n_ref,
                len(reference),
                len(prediction),
            )
            return {"answer_correct": 0.0}

        sim_matrix = [[self._ocr_similarity(ref, pred) for pred in pred_segments] for ref in ref_segments]
        raw_matches = self._ocr_hungarian_maximize(sim_matrix)
        good_matches = [(i, j, sim_matrix[i][j]) for i, j in raw_matches if sim_matrix[i][j] >= 0.12]

        best_per_ref = [max(row) if row else 0.0 for row in sim_matrix]
        r_dist = sum(best_per_ref) / max(n_ref, n_pred)
        r_dist = max(0.0, min(1.0, r_dist))

        r_count = min(n_ref, n_pred) / max(n_ref, n_pred)
        r_count = max(0.0, min(1.0, r_count))

        if len(good_matches) < 2:
            r_order = 0.0
        else:
            pred_order = [j for _i, j, _score in sorted(good_matches)]
            max_inv = len(pred_order) * (len(pred_order) - 1) / 2
            inversions = self._ocr_count_inversions(pred_order)
            r_order = 1.0 - inversions / max_inv if max_inv else 0.0
            r_order = max(0.0, min(1.0, r_order))

        reward = r_dist + r_count + r_order
        logger.info(
            "[infinity_doc_reward_probe] reward=%s r_dist=%s r_count=%s r_order=%s "
            "n_ref=%d n_pred=%d matched=%d raw_matched=%d reference_chars=%d prediction_chars=%d",
            reward,
            r_dist,
            r_count,
            r_order,
            n_ref,
            n_pred,
            len(good_matches),
            len(raw_matches),
            len(reference),
            len(prediction),
        )
        return {"answer_correct": reward}

    @staticmethod
    def _ocr_assistant_text(messages: Messages) -> str:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            parts.append(str(text))
                    elif item:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(content)
        return ""

    def _ocr_clean_text(self, value: str) -> str:
        text = str(value or "").strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[0].strip().lower() in ("```", "```markdown", "```md", "```html"):
                text = "\n".join(lines[1:-1]).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        return unescape(text).strip()

    def _ocr_segments(self, value: str) -> list[str]:
        text = self._ocr_clean_text(value)
        if not text:
            return []

        table_rows = self._ocr_html_table_rows(text)
        table_rows.extend(self._ocr_markdown_table_rows(text))
        if table_rows:
            return table_rows

        html_blocks = self._ocr_html_blocks(text)
        if len(html_blocks) > 1:
            return html_blocks

        blocks = [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]
        if len(blocks) > 1:
            return blocks

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines

        return [text.strip()]

    def _ocr_html_table_rows(self, value: str) -> list[str]:
        rows: list[str] = []
        for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", value, flags=re.IGNORECASE | re.DOTALL):
            cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
            normalized_cells = [self._ocr_strip_markup(cell) for cell in cells]
            normalized_cells = [cell for cell in normalized_cells if cell]
            if normalized_cells:
                rows.append(" | ".join(normalized_cells))
        return rows

    def _ocr_markdown_table_rows(self, value: str) -> list[str]:
        rows: list[str] = []
        for line in value.splitlines():
            stripped = line.strip()
            if "|" not in stripped:
                continue
            compact = stripped.strip("|").strip()
            if not compact:
                continue
            if re.fullmatch(r":?-{3,}:?(\s*\|\s*:?-{3,}:?)+", compact):
                continue
            cells = [self._ocr_strip_markup(cell) for cell in compact.split("|")]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        return rows

    def _ocr_html_blocks(self, value: str) -> list[str]:
        if not re.search(r"<[a-zA-Z][^>]*>", value):
            return []
        text = re.sub(r"<\s*br\s*/?\s*>", "\n", value, flags=re.IGNORECASE)
        text = re.sub(
            r"</\s*(p|div|section|article|h[1-6]|li|ul|ol|tr|table|thead|tbody)\s*>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        stripped = self._ocr_strip_markup(text)
        return [line.strip() for line in stripped.splitlines() if line.strip()]

    def _ocr_strip_markup(self, value: str) -> str:
        text = str(value or "")
        text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        return self._ocr_collapse_text(text)

    @staticmethod
    def _ocr_collapse_text(value: str) -> str:
        text = str(value or "")
        text = text.replace("\u00a0", " ")
        text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:%)])", r"\1", text)
        text = re.sub(r"([$([{])\s+", r"\1", text)
        return text.strip()

    def _ocr_normalize_for_similarity(self, value: str) -> str:
        text = self._ocr_strip_markup(value).lower()
        text = re.sub(r"[*_`#]+", " ", text)
        text = re.sub(r"\s*\|\s*", " ", text)
        text = re.sub(r"\$\s+", "$", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _ocr_similarity(self, reference: str, prediction: str) -> float:
        ref = self._ocr_normalize_for_similarity(reference)
        pred = self._ocr_normalize_for_similarity(prediction)
        if not ref and not pred:
            return 1.0
        if not ref or not pred:
            return 0.0

        char_score = SequenceMatcher(None, ref, pred, autojunk=False).ratio()
        ref_tokens = re.findall(r"[a-z0-9]+|[$%()/.+-]", ref)
        pred_tokens = re.findall(r"[a-z0-9]+|[$%()/.+-]", pred)
        if not ref_tokens or not pred_tokens:
            return char_score

        ref_counts = Counter(ref_tokens)
        pred_counts = Counter(pred_tokens)
        overlap = sum((ref_counts & pred_counts).values())
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return max(char_score, token_f1)

    @staticmethod
    def _ocr_hungarian_maximize(scores: list[list[float]]) -> list[tuple[int, int]]:
        if not scores or not scores[0]:
            return []
        try:
            from scipy.optimize import linear_sum_assignment

            cost = [[-score for score in row] for row in scores]
            row_indices, col_indices = linear_sum_assignment(cost)
            return [(int(i), int(j)) for i, j in zip(row_indices, col_indices)]
        except Exception as exc:
            logger.warning("[infinity_doc_reward_probe] scipy_hungarian_failed=%r using_greedy_fallback", exc)
            candidates = sorted(
                ((score, i, j) for i, row in enumerate(scores) for j, score in enumerate(row)),
                reverse=True,
            )
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            matches: list[tuple[int, int]] = []
            for _score, i, j in candidates:
                if i in used_rows or j in used_cols:
                    continue
                used_rows.add(i)
                used_cols.add(j)
                matches.append((i, j))
            return matches

    @staticmethod
    def _ocr_count_inversions(values: list[int]) -> int:
        inversions = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i] > values[j]:
                    inversions += 1
        return inversions


Geo3KVLMEnv = OCREnv
