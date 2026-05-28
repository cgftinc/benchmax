"""TpufNamespace — low-level Turbopuffer namespace wrapper.

Handles query building (BM25, vector, hybrid), row ↔ Chunk conversion,
batch upsert, and ID pagination.  Used by ``TpufChunkSource`` so the
source module stays thin.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm

from benchmax.rag.chunkers.models import Chunk, ChunkCollection
from benchmax.rag.corpus.search_schema.search_types import (
    HybridOptions,
)


class TpufNamespace:
    """Thin wrapper around a Turbopuffer namespace.

    Encapsulates query construction, row conversion, upsert logic, and
    pagination so that ``TpufChunkSource`` can focus on the ChunkSource
    protocol.
    """

    def __init__(
        self,
        api_key: str,
        namespace: str,
        region: str = "aws-us-east-1",
        content_attr: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        vector_attr: str = "vector",
        distance_metric: str = "cosine_distance",
    ) -> None:
        import turbopuffer

        self._ns = turbopuffer.Turbopuffer(api_key=api_key, region=region).namespace(namespace)
        self.fields: list[str] = content_attr if content_attr is not None else ["content"]
        self.vector_field = str(vector_attr or "vector").strip() or "vector"
        self.embed_fn = embed_fn
        self.distance_metric = distance_metric

    # ------------------------------------------------------------------
    # Namespace access
    # ------------------------------------------------------------------

    @property
    def ns(self) -> Any:
        """Direct access to the underlying Turbopuffer namespace resource."""
        return self._ns

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    _TPUF_BM25_MAX_QUERY_LEN = 1024

    def build_bm25_rank_by(self, query: str) -> Any:
        """Build the ``rank_by`` argument for a BM25 query."""
        # Turbopuffer caps BM25 queries at 1024 unicode code points.
        query = query[: self._TPUF_BM25_MAX_QUERY_LEN]
        if len(self.fields) == 1:
            return (self.fields[0], "BM25", query)
        return (
            "Sum",
            tuple(("Product", 1, (f, "BM25", query)) for f in self.fields),
        )

    def build_vector_rank_by(self, vector: list[float]) -> Any:
        """Build the ``rank_by`` argument for a vector ANN query."""
        return (self.vector_field, "ANN", vector)

    def build_native_hybrid_rank_by(
        self,
        query: str,
        vector: list[float],
        hybrid_opts: HybridOptions | None,
    ) -> Any:
        """Build the ``rank_by`` argument for a hybrid BM25+vector query."""
        opts = hybrid_opts or {}
        lexical_weight = opts.get("lexical_weight", 1.0)
        vector_weight = opts.get("vector_weight", 1.0)
        return [
            {
                "type": "bm25",
                "text": query,
                "fields": self.fields,
                "weight": lexical_weight,
            },
            {
                "type": "vector",
                "vector": vector,
                "field": self.vector_field,
                "weight": vector_weight,
            },
        ]

    # ------------------------------------------------------------------
    # Row ↔ Chunk conversion
    # ------------------------------------------------------------------

    def row_content(self, row: Any) -> str:
        """Return the text content for a row, drawn from ``self.fields``."""
        if len(self.fields) == 1:
            return str(getattr(row, self.fields[0], ""))
        return json.dumps({f: getattr(row, f, "") for f in self.fields}, default=str)

    def row_to_chunk(self, row: Any) -> Chunk:
        """Convert a Turbopuffer row to a :class:`Chunk`.

        Stores the Turbopuffer row ID in metadata as ``_tpuf_id`` so that
        ``search_related`` can reliably identify and skip the source chunk.
        """
        content = self.row_content(row)

        try:
            # Turbopuffer Row is a Pydantic model; dynamic attributes
            # (all user metadata) live in model_extra, not __dict__.
            raw = getattr(row, "model_extra", None) or vars(row)
            attrs: dict = {
                k: v
                for k, v in raw.items()
                if not k.startswith("$") and k not in ("id", self.vector_field)
            }
        except TypeError:
            attrs = {f: getattr(row, f, "") for f in self.fields}

        attrs["_tpuf_id"] = row.id
        metadata = tuple(attrs.items())
        return Chunk(content=content, metadata=metadata)

    # ------------------------------------------------------------------
    # Batch upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        collection: ChunkCollection,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
        id_offset: int = 0,
    ) -> int:
        """Upload a :class:`ChunkCollection` to the namespace.

        Args:
            id_offset: Starting offset for row IDs (default 0). The first
                chunk gets ID ``id_offset + 1``. Use this when uploading
                a slice of a larger collection to keep IDs consistent.

        Returns the total number of chunks written.
        """
        all_chunks = list(collection)

        if show_summary:
            print(f"\nUploading {len(all_chunks)} chunks to Turbopuffer namespace...")

        for batch_start in tqdm(
            range(0, len(all_chunks), batch_size),
            desc="Uploading batches",
            disable=not show_summary,
        ):
            batch = all_chunks[batch_start : batch_start + batch_size]

            _TPUF_ATTR_LIMIT = 4000  # Turbopuffer 4096 byte limit, leave margin

            upsert_rows = []
            for i, chunk in enumerate(batch):
                row: dict[str, Any] = {
                    "id": id_offset + batch_start + i + 1,
                    "content": chunk.content,
                }

                # Write all user metadata, truncating strings to the
                # Turbopuffer attribute size limit.
                for mk, mv in chunk.metadata:
                    if mk == "content":
                        continue
                    if isinstance(mv, (str, int, float, bool)):
                        row.setdefault(
                            mk,
                            mv[:_TPUF_ATTR_LIMIT] if isinstance(mv, str) else mv,
                        )
                    elif mv is not None:
                        row.setdefault(mk, str(mv)[:_TPUF_ATTR_LIMIT])

                # Derived fields — only fill if not already present.
                row.setdefault(
                    "file_path",
                    chunk.get_metadata("file", "")[:_TPUF_ATTR_LIMIT],
                )
                row.setdefault("chunk_index", chunk.get_metadata("index", 0))
                row["chunk_hash"] = chunk.hash
                row["char_count"] = len(chunk)

                upsert_rows.append(row)

            write_kwargs: dict = {
                "upsert_rows": upsert_rows,
                "schema": {"content": {"type": "string", "full_text_search": True}},
            }

            if embed_fn:
                vectors = embed_fn([chunk.content for chunk in batch])
                for row, v in zip(upsert_rows, vectors):
                    row[self.vector_field] = v
                write_kwargs["distance_metric"] = self.distance_metric

            self._ns.write(**write_kwargs)

        if show_summary:
            print(f"\nUpload complete! {len(all_chunks)} chunks written to namespace.")

        return len(all_chunks)

    # ------------------------------------------------------------------
    # Namespace metadata
    # ------------------------------------------------------------------

    def get_approx_row_count(self) -> int | None:
        """Return the approximate row count from namespace metadata.

        Uses the tpuf metadata endpoint which returns ``approx_row_count``.
        Unlike ``get_max_id()``, this reflects actual rows (accounting for
        deletions) rather than the highest assigned ID.
        """
        try:
            meta = self._ns.metadata()
            count = getattr(meta, "approx_row_count", None)
            if isinstance(count, int):
                return count
            # Fallback: some SDK versions return a dict
            if isinstance(meta, dict):
                return meta.get("approx_row_count")
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # ID pagination
    # ------------------------------------------------------------------

    def get_max_id(self) -> int | None:
        """Return the highest row ID in the namespace, or None if empty."""
        try:
            result = self._ns.query(
                rank_by=["id", "desc"],
                top_k=1,
            )
        except Exception:
            # Namespace doesn't exist yet — will be created on first write.
            return None
        rows = result.rows
        if not rows:
            return None
        return rows[0].id

    def scan_all_rows(self, limit: int | None = None, page_size: int = 10_000) -> list[Any]:
        """Sequentially scan all rows with attributes via cursor pagination.

        Much faster than random-ID sampling for large fetches — single pass,
        no retries, no ID collisions. Returns up to ``limit`` rows (all if
        None).
        """
        all_rows: list[Any] = []
        last_id = 0

        while True:
            result = self._ns.query(
                rank_by=["id", "asc"],
                filters=["id", "Gt", last_id],
                top_k=page_size,
                include_attributes=True,
            )
            rows = result.rows
            if not rows:
                break
            all_rows.extend(rows)
            last_id = rows[-1].id
            if limit is not None and len(all_rows) >= limit:
                return all_rows[:limit]
            if len(rows) < page_size:
                break

        return all_rows

    def paginate_all_ids(self, page_size: int = 1000) -> list[int]:
        """Return all row IDs in the namespace via cursor pagination."""
        all_ids: list[int] = []
        last_id = 0

        while True:
            result = self._ns.query(
                rank_by=["id", "asc"],
                filters=["id", "Gt", last_id],
                top_k=page_size,
            )
            rows = result.rows
            if not rows:
                break
            all_ids.extend(row.id for row in rows)
            last_id = rows[-1].id
            if len(rows) < page_size:
                break

        return all_ids

    def query_rows_by_ids(self, ids: list[int]) -> list[Any]:
        """Fetch rows for a list of IDs, with attributes."""
        if not ids:
            return []
        result = self._ns.query(
            rank_by=["id", "asc"],
            filters=["id", "In", ids],
            top_k=len(ids),
            include_attributes=True,
        )
        return list(result.rows)
