"""File-structure awareness helpers for Pinecone indexes.

Provides neighboring-chunk context, top-level file discovery, and
adjacent-chunk detection — backed by ``file_path`` / ``chunk_index``
metadata fields.  Gracefully degrades when those fields are absent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from benchmax.rag.chunkers.models import Chunk

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .index_client import PineconeIndexClient


class FileAwareness:
    """Lazy-probing, cached file-structure helpers.

    All public methods return sensible defaults (empty lists, ``False``,
    etc.) when the index lacks ``file_path`` / ``chunk_index`` metadata,
    so callers never need to guard against missing metadata.
    """

    def __init__(self, client: PineconeIndexClient) -> None:
        self._client = client
        self._file_cache: dict[str, list[Chunk]] = {}
        self._all_file_paths: list[str] | None = None
        self._aware: bool | None = None  # None = not yet probed

    # ------------------------------------------------------------------
    # Static chunk-metadata extractors
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_file_path(chunk: Chunk) -> str | None:
        """Return the file path from a chunk's metadata."""
        return chunk.get_metadata("file_path") or chunk.get_metadata("file") or None

    @staticmethod
    def chunk_index(chunk: Chunk) -> int | None:
        """Return the integer chunk index from metadata."""
        val = chunk.get_metadata("chunk_index")
        if val is None:
            val = chunk.get_metadata("index")
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Namespace probing
    # ------------------------------------------------------------------

    def check(self) -> bool:
        """Probe the index for ``file_path`` + ``chunk_index`` metadata.

        Fetches a single vector to inspect its metadata keys.
        Result is cached after the first call.
        """
        if self._aware is not None:
            return self._aware
        try:
            ids = self._client.list_all_ids()
            if not ids:
                self._aware = False
                return self._aware

            chunks = self._client.fetch_by_ids(ids[:1])
            if chunks:
                chunk = chunks[0]
                has_fp = chunk.get_metadata("file_path") is not None
                has_ci = chunk.get_metadata("chunk_index") is not None
                self._aware = has_fp and has_ci
            else:
                self._aware = False
        except Exception:
            self._aware = False
        return self._aware

    # ------------------------------------------------------------------
    # File-level queries (cached)
    # ------------------------------------------------------------------

    def get_file_chunks(self, file_path: str) -> list[Chunk]:
        """Return all chunks for *file_path*, sorted by ``chunk_index``.

        Results are cached per file path until :meth:`invalidate` is
        called.
        """
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        fp_key = self._client._pc_field("file_path")
        filter_dict = {fp_key: {"$eq": file_path}}

        # Use a zero-vector with metadata filter to retrieve all chunks
        # for this file.  The vector value doesn't matter since we're
        # filtering by metadata — we just need the correct dimension.
        chunks: list[Chunk] = []
        # Pinecone max top_k is 10000; most files have far fewer chunks.
        result = self._client.query(
            vector=self._client.zero_vector(),
            top_k=10000,
            filter=filter_dict,
            include_metadata=True,
        )
        for match in result.matches or []:
            chunks.append(self._client.match_to_chunk(match))

        chunks.sort(key=lambda c: self.chunk_index(c) or 0)
        self._file_cache[file_path] = chunks
        return chunks

    def get_all_file_paths(self) -> list[str]:
        """Return all unique ``file_path`` values in the index.

        Paginates all vector IDs via ``list_paginated``, then batch-fetches
        metadata to extract file paths.  Slower than a single query but
        complete for indexes of any size.  Cached until :meth:`invalidate`
        is called.
        """
        if self._all_file_paths is not None:
            return self._all_file_paths

        all_ids = self._client.list_all_ids()
        paths: set[str] = set()
        fp_key = self._client._pc_field("file_path")
        batch_size = 100

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            chunks = self._client.fetch_by_ids(batch_ids)
            for chunk in chunks:
                fp = chunk.get_metadata(fp_key) or chunk.get_metadata("file_path")
                if fp:
                    paths.add(str(fp))

        self._all_file_paths = sorted(paths)
        return self._all_file_paths

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Clear all file-structure caches.

        Should be called after any populate / upsert operation.
        """
        self._file_cache.clear()
        self._all_file_paths = None
        self._aware = None
