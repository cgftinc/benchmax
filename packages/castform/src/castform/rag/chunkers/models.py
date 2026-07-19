"""Data models for chunking."""

import hashlib
import json
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel, frozen=True):
    """A chunk of text with associated metadata.

    Attributes:
        content: The text content of the chunk.
        metadata: Dictionary containing metadata about the chunk,
            such as header hierarchy (h1, h2, h3), source file, and index.

    Note:
        This model is frozen (immutable). All fields including metadata
        must be set at creation time and cannot be modified afterwards.
    """

    model_config = {"frozen": True}

    content: str = Field(description="The text content of the chunk")
    metadata: tuple[tuple[str, Any], ...] = Field(
        default=(),
        description="Metadata about the chunk (headers, file, index, etc.) as frozen tuples",
    )
    hash: str = Field(default="", description="The hash of the chunk metadata + content")

    @model_validator(mode="after")
    def _compute_hash(self) -> "Chunk":
        """Automatically compute hash from metadata + content."""
        if not self.hash:
            combined = self.metadata_str() + "\n" + self.content
            # Use object.__setattr__ to bypass frozen restriction during validation
            object.__setattr__(self, "hash", hashlib.sha256(combined.encode()).hexdigest())
        return self

    @property
    def metadata_dict(self) -> dict[str, Any]:
        """Return metadata as a dictionary."""
        return dict(self.metadata)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key.

        Args:
            key: The metadata key to look up.
            default: Value to return if key is not found.

        Returns:
            The metadata value or the default.
        """
        return self.metadata_dict.get(key, default)

    def __len__(self) -> int:
        """Return the length of the chunk content."""
        return len(self.content)

    def __str__(self) -> str:
        """Return string representation with metadata and content."""
        return self.chunk_str()

    def chunk_str(self, max_chars: int | None = None, truncate: str = "trailing") -> str:
        """Return string representation with metadata and optionally truncated content.

        Args:
            max_chars: Maximum characters to show. If None, shows full content.
            truncate: Where to truncate if max_chars exceeded.
                - "trailing": Show beginning, cut end (content...)
                - "leading": Show end, cut beginning (...content)

        Returns:
            Formatted string with metadata and content.
        """
        content = self.content
        if max_chars is not None and len(content) > max_chars:
            if truncate == "leading":
                content = "..." + content[-(max_chars - 3) :]
            else:  # trailing
                content = content[: max_chars - 3] + "..."
        return f"{self.metadata_str()}\n{content}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dict with 'content' and 'metadata' keys.
        """
        return {"content": self.content, "metadata": self.metadata_dict}

    def metadata_str(self) -> str:
        """Return a string representation of the metadata for debugging.

        Returns:
            Stringified metadata.
        """
        return json.dumps(self.metadata_dict, indent=2, sort_keys=True)


class ChunkCollection:
    """Container for chunks with file structure awareness.

    List-compatible: can iterate, index, and get length.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Initialize with a list of chunks.

        Args:
            chunks: List of Chunk objects
        """
        self.chunks = chunks
        self._by_file: dict[str, list[Chunk]] = defaultdict(list)
        self._chunk_index: dict[str, tuple[str, int]] = {}

        for chunk in chunks:
            file = chunk.get_metadata("file", "unknown")
            position = len(self._by_file[file])
            self._by_file[file].append(chunk)
            self._chunk_index[chunk.hash] = (file, position)

    def __iter__(self):
        """Iterate over all chunks."""
        return iter(self.chunks)

    def __len__(self):
        """Return total number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx):
        """Get chunk by index."""
        return self.chunks[idx]

    @property
    def files(self) -> list[str]:
        """List of unique file paths in the collection."""
        return list(self._by_file.keys())

    def get_file_chunks(self, file: str) -> list[Chunk]:
        """Get all chunks from a specific file.

        Args:
            file: File path (as stored in chunk metadata)

        Returns:
            List of chunks from that file
        """
        return self._by_file.get(file, [])

    def get_neighboring_chunks(
        self, chunk: Chunk, before: int = 1, after: int = 1
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Get chunks before/after the given chunk in the same file.

        Args:
            chunk: The reference chunk
            before: Number of chunks before to retrieve
            after: Number of chunks after to retrieve

        Returns:
            Tuple of (chunks_before, chunks_after)
        """
        if chunk.hash not in self._chunk_index:
            return ([], [])

        file, position = self._chunk_index[chunk.hash]
        file_chunks = self._by_file[file]

        before_chunks = file_chunks[max(0, position - before) : position]
        after_chunks = file_chunks[position + 1 : position + 1 + after]

        return (before_chunks, after_chunks)

    def get_chunk_by_hash(self, chunk_hash: str) -> Chunk | None:
        """Get a chunk by its hash.

        Args:
            chunk_hash: The hash of the chunk to retrieve

        Returns:
            The chunk if found, None otherwise
        """
        index_entry = self._chunk_index.get(chunk_hash)
        if index_entry is None:
            return None
        file, position = index_entry
        return self._by_file[file][position]

    def get_top_level_chunks(self) -> list[Chunk]:
        """Get chunks from files at the top level (shallowest directory depth).

        Returns:
            List of chunks from files with the minimum directory depth
        """
        if not self._by_file:
            return []

        # Calculate depth for each file (number of path separators)
        file_depths = {f: f.count("/") for f in self._by_file.keys()}
        min_depth = min(file_depths.values())

        top_level_files = [f for f, depth in file_depths.items() if depth == min_depth]
        return [chunk for f in top_level_files for chunk in self._by_file[f]]

    def get_chunk_with_context(
        self,
        chunk: Chunk,
        context_max_chars: int = 200,
        include_before: bool = True,
        include_after: bool = True,
    ) -> dict[str, str]:
        """Get a chunk with its neighboring context as formatted strings.

        Args:
            chunk: The chunk to get context for
            context_max_chars: Max chars for context previews
            include_before: Whether to include the previous chunk
            include_after: Whether to include the next chunk

        Returns:
            Dict with keys:
                - chunk_content: The main chunk as a string
                - prev_chunk_preview: Previous chunk preview (or placeholder)
                - next_chunk_preview: Next chunk preview (or placeholder)
        """
        before_count = 1 if include_before else 0
        after_count = 1 if include_after else 0

        chunks_before, chunks_after = self.get_neighboring_chunks(
            chunk, before=before_count, after=after_count
        )

        prev_preview = "(No previous chunk)"
        if include_before and chunks_before:
            prev_preview = chunks_before[0].chunk_str(
                max_chars=context_max_chars, truncate="leading"
            )

        next_preview = "(No next chunk)"
        if include_after and chunks_after:
            next_preview = chunks_after[0].chunk_str(
                max_chars=context_max_chars, truncate="trailing"
            )

        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": prev_preview,
            "next_chunk_preview": next_preview,
        }
