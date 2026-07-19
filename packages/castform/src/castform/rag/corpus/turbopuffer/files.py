"""File-structure awareness helpers for Turbopuffer namespaces.

Provides neighboring-chunk context, top-level file discovery, and
adjacent-chunk detection — all backed by ``file_path`` / ``chunk_index``
attributes in the namespace.  Gracefully degrades when those attributes
are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from castform.rag.chunkers.models import Chunk

if TYPE_CHECKING:
    from .namespace import TpufNamespace


class FileAwareness:
    """Lazy-probing, cached file-structure helpers.

    All public methods return sensible defaults (empty lists, ``False``,
    etc.) when the namespace lacks ``file_path`` / ``chunk_index``
    attributes, so callers never need to guard against missing metadata.
    """

    def __init__(self, client: TpufNamespace) -> None:
        self._client = client
        self._file_cache: dict[str, list[Chunk]] = {}
        self._all_file_paths: list[str] | None = None
        self._aware: bool | None = None  # None = not yet probed

    # ------------------------------------------------------------------
    # Static chunk-metadata extractors
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_file_path(chunk: Chunk) -> str | None:
        """Return the file path from a chunk's metadata.

        Checks both ``file_path`` (Turbopuffer attr written during
        upload) and ``file`` (original chunker metadata key).
        """
        return chunk.get_metadata("file_path") or chunk.get_metadata("file") or None

    @staticmethod
    def chunk_index(chunk: Chunk) -> int | None:
        """Return the integer chunk index from metadata.

        Checks ``chunk_index`` and ``index`` variants.
        """
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
        """Probe the namespace for ``file_path`` + ``chunk_index``.

        Result is cached after the first call.
        """
        if self._aware is not None:
            return self._aware
        try:
            result = self._client.ns.query(
                rank_by=["id", "asc"],
                top_k=1,
                include_attributes=["file_path", "chunk_index"],
            )
            if result.rows:
                row = result.rows[0]
                has_fp = getattr(row, "file_path", None) is not None
                has_ci = getattr(row, "chunk_index", None) is not None
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

        chunks: list[Chunk] = []
        last_id = 0
        page_size = 500

        while True:
            result = self._client.ns.query(
                rank_by=["id", "asc"],
                filters=[
                    "And",
                    [
                        ["file_path", "Eq", file_path],
                        ["id", "Gt", last_id],
                    ],
                ],
                top_k=page_size,
                include_attributes=True,
            )
            if not result.rows:
                break
            for row in result.rows:
                chunks.append(self._client.row_to_chunk(row))
            last_id = result.rows[-1].id
            if len(result.rows) < page_size:
                break

        chunks.sort(key=lambda c: self.chunk_index(c) or 0)
        self._file_cache[file_path] = chunks
        return chunks

    def get_all_file_paths(self) -> list[str]:
        """Return all unique ``file_path`` values in the namespace.

        Cached until :meth:`invalidate` is called.
        """
        if self._all_file_paths is not None:
            return self._all_file_paths

        paths: set[str] = set()
        last_id = 0
        page_size = 1000

        while True:
            result = self._client.ns.query(
                rank_by=["id", "asc"],
                filters=["id", "Gt", last_id],
                top_k=page_size,
                include_attributes=["file_path"],
            )
            if not result.rows:
                break
            for row in result.rows:
                fp = getattr(row, "file_path", None)
                if fp:
                    paths.add(fp)
            last_id = result.rows[-1].id
            if len(result.rows) < page_size:
                break

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
