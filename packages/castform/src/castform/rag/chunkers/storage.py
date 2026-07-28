"""Storage utilities for persisting and loading chunk collections."""

from pathlib import Path
from typing import Any

from castform.rag.chunkers.models import Chunk, ChunkCollection
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


def _to_literal_scalars(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert multiline strings to LiteralScalarString for block style output."""
    result = []
    for item in data:
        new_item = dict(item)
        if isinstance(new_item.get("content"), str) and "\n" in new_item["content"]:
            new_item["content"] = LiteralScalarString(new_item["content"])
        result.append(new_item)
    return result


def save_chunks(collection: ChunkCollection, path: str | Path) -> None:
    """Save chunk collection to a YAML file.

    Args:
        collection: ChunkCollection to save
        path: File path to save to (will be created if doesn't exist)

    Example:
        >>> collection = chunker.chunk_folder("./docs")
        >>> save_chunks(collection, "chunks.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to flat list for YAML
    chunks_list = list(collection)
    data = _to_literal_scalars([chunk.to_dict() for chunk in chunks_list])

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 1000

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def load_chunks(path: str | Path) -> ChunkCollection:
    """Load chunks from a YAML file.

    Args:
        path: File path to load from

    Returns:
        ChunkCollection created from loaded chunks

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> collection = load_chunks("chunks.yaml")
        >>> inspector = ChunkInspector(collection)
    """
    path = Path(path)

    yaml = YAML()
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = yaml.load(f)

    chunks = []
    for item in data:
        metadata = item.get("metadata", {})
        # Convert dict metadata to tuple format if needed
        if isinstance(metadata, dict):
            metadata = tuple(metadata.items())
        chunks.append(Chunk(content=item["content"], metadata=metadata))
    return ChunkCollection(chunks)
