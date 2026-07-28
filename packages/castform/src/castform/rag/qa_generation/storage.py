"""Storage utilities for persisting and loading QA datasets."""

import json
from pathlib import Path
from typing import Any

from castform.rag.qa_generation.models import QADataPoint
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


def _to_literal_scalars(data: list[QADataPoint]) -> list[dict[str, Any]]:
    """Convert multiline strings to LiteralScalarString for block style output."""
    result = []
    for item in data:
        new_item: dict[str, Any] = dict(item)
        if isinstance(new_item.get("answer"), str) and "\n" in new_item["answer"]:
            new_item["answer"] = LiteralScalarString(new_item["answer"])
        if "reference_chunks" in new_item:
            new_chunks = []
            for chunk in new_item["reference_chunks"]:
                new_chunk: dict[str, Any] = dict(chunk)
                if isinstance(new_chunk.get("content"), str) and "\n" in new_chunk["content"]:
                    new_chunk["content"] = LiteralScalarString(new_chunk["content"])
                new_chunks.append(new_chunk)
            new_item["reference_chunks"] = new_chunks
        result.append(new_item)
    return result


# === YAML ===


def save_qa_dataset(dataset: list[QADataPoint], path: str | Path) -> None:
    """Save QA dataset to a YAML file (human-readable format).

    Args:
        dataset: List of QADataPoints to save
        path: File path to save to (will be created if doesn't exist)

    Example:
        >>> save_qa_dataset([QADataPoint(...)], "qa_pairs.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 1000

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_to_literal_scalars(dataset), f)


def load_qa_dataset(path: str | Path) -> list[QADataPoint]:
    """Load QA dataset from a YAML file.

    Args:
        path: File path to load from

    Returns:
        List of QADataPoints loaded from file

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> dataset = load_qa_dataset("qa_pairs.yaml")
    """
    path = Path(path)

    yaml = YAML()
    with open(path, encoding="utf-8") as f:
        return yaml.load(f)


# === JSONL ===


def save_qa_dataset_jsonl(dataset: list[QADataPoint], path: str | Path) -> None:
    """Save QA dataset to a JSONL file (one JSON object per line).

    This format is compatible with HuggingFace ``load_dataset("json", ...)``.

    Args:
        dataset: List of QADataPoints to save
        path: File path to save to (will be created if doesn't exist)

    Example:
        >>> save_qa_dataset_jsonl([QADataPoint(...)], "qa_pairs.jsonl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for dp in dataset:
            f.write(json.dumps(dp, ensure_ascii=False) + "\n")


def save_jsonl_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Save generic dict rows to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qa_dataset_jsonl(path: str | Path) -> list[QADataPoint]:
    """Load QA dataset from a JSONL file.

    Args:
        path: File path to load from

    Returns:
        List of QADataPoints loaded from file

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> dataset = load_qa_dataset_jsonl("qa_pairs.jsonl")
    """
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def qa_dataset_to_jsonl_bytes(dataset: list[QADataPoint]) -> bytes:
    """Serialize QA dataset to JSONL bytes (for upload).

    Args:
        dataset: List of QADataPoints to serialize

    Returns:
        UTF-8 encoded JSONL bytes
    """
    lines = [json.dumps(dp, ensure_ascii=False) for dp in dataset]
    return ("\n".join(lines) + "\n").encode("utf-8")
