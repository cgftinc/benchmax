"""File-structure awareness helpers for Chroma collections.

Provides neighboring-chunk context, top-level file discovery, and
adjacent-chunk detection — all backed by ``file_path`` / ``chunk_index``
metadata stored on each document.  Gracefully degrades when those fields
are absent.
"""

from __future__ import annotations

from typing import Any

from castform.rag.chunkers.models import Chunk


class FileAwareness:
    """Lazy-probing, cached file-structure helpers for Chroma.

    All public methods return sensible defaults (empty lists, ``False``,
    etc.) when the collection lacks ``file_path`` / ``chunk_index``
    metadata, so callers never need to guard against missing metadata.
    """

    def __init__(self, get_collection: Any) -> None:
        """Initialize with a callable that returns the Chroma collection.

        Args:
            get_collection: Callable returning a chromadb Collection object.
                We use a callable (rather than the collection directly) to
                support lazy client initialization for pickle safety.
        """
        self._get_collection = get_collection
        self._file_cache: dict[str, list[Chunk]] = {}
        self._all_file_paths: list[str] | None = None
        self._aware: bool | None = None

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
    # Collection probing
    # ------------------------------------------------------------------

    def check(self) -> bool:
        """Probe the collection for ``file_path`` + ``chunk_index`` metadata.

        Result is cached after the first call.
        """
        if self._aware is not None:
            return self._aware
        try:
            col = self._get_collection()
            result = col.get(limit=1, include=["metadatas"])
            if result["metadatas"]:
                meta = result["metadatas"][0]
                has_fp = "file_path" in meta
                has_ci = "chunk_index" in meta
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

        col = self._get_collection()
        result = col.get(
            where={"file_path": {"$eq": file_path}},
            include=["documents", "metadatas"],
        )

        chunks: list[Chunk] = []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            chunks.append(_meta_to_chunk(doc, meta))

        chunks.sort(key=lambda c: self.chunk_index(c) or 0)
        self._file_cache[file_path] = chunks
        return chunks

    def get_all_file_paths(self) -> list[str]:
        """Return all unique ``file_path`` values in the collection.

        Cached until :meth:`invalidate` is called.
        """
        if self._all_file_paths is not None:
            return self._all_file_paths

        col = self._get_collection()
        paths: set[str] = set()
        offset = 0
        page_size = 1000

        while True:
            result = col.get(
                include=["metadatas"],
                limit=page_size,
                offset=offset,
            )
            metas = result.get("metadatas") or []
            if not metas:
                break
            for meta in metas:
                fp = meta.get("file_path")
                if fp:
                    paths.add(fp)
            if len(metas) < page_size:
                break
            offset += page_size

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


def _meta_to_chunk(content: str, metadata: dict[str, Any]) -> Chunk:
    """Build a Chunk from a Chroma document + metadata dict."""
    return Chunk(content=content, metadata=tuple(metadata.items()))
