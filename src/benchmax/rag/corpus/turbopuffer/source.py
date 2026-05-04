"""TpufChunkSource — ChunkSource implementation backed by Turbopuffer."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

from benchmax.rag.chunkers.models import Chunk, ChunkCollection
from benchmax.rag.corpus.search_schema.search_exceptions import (
    InvalidSearchSpecError,
    UnsupportedSearchModeError,
)
from benchmax.rag.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchCapabilities,
    SearchMode,
    SearchSpec,
    validate_search_spec_shape,
)

from .files import FileAwareness
from .filter_mapper import to_turbopuffer_filters
from .namespace import TpufNamespace

_DEFAULT_RELATED_SEARCH_MODE: SearchMode = "lexical"
_HYBRID_FUSION_RRF_K = 60.0

logger = logging.getLogger(__name__)


class TpufChunkSource:
    """ChunkSource backed by a Turbopuffer namespace.

    Chunks are stored in Turbopuffer with vector embeddings and BM25-enabled
    full-text search.

    File-structure awareness (neighboring chunk context, top-level chunk
    retrieval, adjacent-chunk filtering in search_related) is supported when
    the namespace contains ``file_path`` and ``chunk_index`` attributes.
    These are written automatically by ``populate_from_folder`` /
    ``populate_from_chunks``.  For pre-existing namespaces that lack these
    fields, all file-aware methods gracefully degrade to their previous
    stub behavior (empty context, empty top-level list, no neighbor
    filtering).

    Args:
        api_key: Turbopuffer API key
        namespace: Turbopuffer namespace name
        region: Turbopuffer region (default "aws-us-east-1")
        content_attr: List of Turbopuffer attribute names to use as the chunk's
            searchable text content. Defaults to ["content"]. For pre-existing
            namespaces, supply the BM25-indexed field(s), e.g. ["description"]
            or ["title", "content"].
        vector_attr: Name of the vector attribute in the namespace. Defaults to
            "vector". Set this if your namespace stores embeddings under a
            different attribute name.

    Example:
        >>> # Managed docs — populate from folder
        >>> source = TpufChunkSource(api_key="tpuf_...", namespace="my-docs")
        >>> source.populate_from_folder("./docs", embed_fn=my_embed_fn)
        >>> chunks = source.sample_chunks(n=10, min_chars=400)

        >>> # Pre-existing namespace with known BM25-indexed fields
        >>> source = TpufChunkSource(
        ...     api_key="tpuf_...",
        ...     namespace="product-catalog",
        ...     content_attr=["description"],
        ... )
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
        self._client = TpufNamespace(
            api_key=api_key,
            namespace=namespace,
            region=region,
            content_attr=content_attr,
            embed_fn=embed_fn,
            vector_attr=vector_attr,
            distance_metric=distance_metric,
        )
        self._files = FileAwareness(self._client)

        modes: set[SearchMode] = {"lexical"}
        ranking: set[str] = {"bm25"}
        if embed_fn is not None:
            modes |= {"vector", "hybrid"}
            ranking |= {"cosine", "rrf"}

        self._search_capabilities: SearchCapabilities = {
            "backend": "turbopuffer",
            "modes": modes,
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": ranking,
            "constraints": {
                "max_top_k": 10000,
                "vector_dimensions": None,
                "vector_field": self._client.vector_field,
            },
            "graph_expansion": False,
        }

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def populate_from_folder(
        self,
        docs_path: str,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Chunk documents in a folder and upload them to Turbopuffer.

        Args:
            docs_path: Path to the folder containing documents to chunk
            embed_fn: Optional embedding function for vector/hybrid search
            min_chars: Minimum characters per chunk (default 1024)
            max_chars: Maximum characters per chunk (default 2048)
            overlap_chars: Character overlap between adjacent chunks
            file_extensions: File types to process (default [".md", ".mdx"])
            batch_size: Number of chunks per upload batch (default 100)
            show_summary: Print chunking summary and upload progress
        """
        from benchmax.rag.chunkers.inspector import ChunkInspector
        from benchmax.rag.chunkers.markdown import MarkdownChunker

        if file_extensions is None:
            file_extensions = [".md", ".mdx"]

        if show_summary:
            print(f"Chunking documents from {docs_path}...")

        chunker = MarkdownChunker(
            min_char=min_chars,
            max_char=max_chars,
            chunk_overlap=overlap_chars,
        )
        collection = chunker.chunk_folder(docs_path, file_extensions=file_extensions)

        if show_summary:
            inspector = ChunkInspector(collection)
            inspector.summary(max_depth=3, max_files_per_folder=4)

        self.populate_from_chunks(
            collection=collection,
            embed_fn=embed_fn,
            batch_size=batch_size,
            show_summary=show_summary,
        )
        self._files.invalidate()

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to Turbopuffer.

        Args:
            collection: ChunkCollection produced by any chunker.
            embed_fn: Optional embedding function for vector/hybrid search.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print upload progress (default True).
        """
        self._client.upsert_chunks(
            collection,
            embed_fn=embed_fn,
            batch_size=batch_size,
            show_summary=show_summary,
        )
        self._files.invalidate()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_chunk_count(self) -> int:
        """Return the total number of chunks in the namespace."""
        return self._client.get_max_id() or 0

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by minimum length.

        Uses random ID generation for dense namespaces (efficient in practice).
        Automatically falls back to full ID pagination when the namespace is sparse
        (high deletion rate or non-sequential IDs), detected by a low hit rate on
        the first sampling attempt.
        """
        max_id = self._client.get_max_id()
        if max_id is None:
            return []

        # Over-sample to account for gaps in ID space and min_chars filtering.
        # With min_chars, we need more candidates since some will be too short.
        oversample = 3 if min_chars > 0 else 1.2
        max_attempts = 5
        collected: list[Chunk] = []
        seen_ids: set[int] = set()

        for attempt in range(max_attempts):
            if len(collected) >= n:
                break

            remaining = n - len(collected)
            sample_size = min(int(remaining * oversample), max_id - len(seen_ids))
            candidate_ids = []
            while len(candidate_ids) < sample_size:
                rid = random.randint(1, max_id)
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    candidate_ids.append(rid)
                # Safety: if we've exhausted the full ID space, stop
                if len(seen_ids) >= max_id:
                    break

            if not candidate_ids:
                break

            hits_before = len(collected)

            # Fetch in batches of 500 to avoid oversized requests
            for batch_start in range(0, len(candidate_ids), 500):
                batch_ids = candidate_ids[batch_start : batch_start + 500]
                rows = self._client.query_rows_by_ids(batch_ids)
                for row in rows:
                    chunk = self._client.row_to_chunk(row)
                    if min_chars > 0 and len(chunk.content) < min_chars:
                        continue
                    collected.append(chunk)
                    if len(collected) >= n:
                        break
                if len(collected) >= n:
                    break

            # After the first attempt, measure hit rate. A low rate means the
            # namespace has sparse IDs (deletions or non-sequential assignment);
            # fall back to full pagination to avoid many wasted retries.
            if attempt == 0 and candidate_ids:
                hit_rate = (len(collected) - hits_before) / len(candidate_ids)
                if hit_rate < 0.2:
                    logger.debug(
                        "sample_chunks: low hit rate (%.2f) on random ID sampling — "
                        "falling back to full ID pagination for sparse namespace",
                        hit_rate,
                    )
                    return self._sample_chunks_via_pagination(n, min_chars=min_chars)

        return collected[:n]

    def _sample_chunks_via_pagination(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Fallback sampler for sparse namespaces: paginate all IDs then sample uniformly."""
        all_ids = self._client.paginate_all_ids()
        if not all_ids:
            return []
        oversample = 3 if min_chars > 0 else 1
        candidate_count = min(int(n * oversample), len(all_ids))
        sampled_ids = random.sample(all_ids, candidate_count)
        collected: list[Chunk] = []
        for batch_start in range(0, len(sampled_ids), 500):
            batch_ids = sampled_ids[batch_start : batch_start + 500]
            rows = self._client.query_rows_by_ids(batch_ids)
            for row in rows:
                chunk = self._client.row_to_chunk(row)
                if min_chars > 0 and len(chunk.content) < min_chars:
                    continue
                collected.append(chunk)
                if len(collected) >= n:
                    return collected[:n]
        return collected[:n]

    # ------------------------------------------------------------------
    # Context & structure
    # ------------------------------------------------------------------

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return chunk with neighboring context from the same file.

        When ``file_path`` and ``chunk_index`` attributes exist, previous
        and next chunk previews are fetched.  Otherwise returns empty
        strings.
        """
        fallback = {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": "",
            "next_chunk_preview": "",
        }

        if not self._files.check():
            return fallback

        file_path = self._files.chunk_file_path(chunk)
        idx = self._files.chunk_index(chunk)
        if not file_path or idx is None:
            return fallback

        file_chunks = self._files.get_file_chunks(file_path)
        if not file_chunks:
            return fallback

        position = None
        for i, fc in enumerate(file_chunks):
            if self._files.chunk_index(fc) == idx:
                position = i
                break

        if position is None:
            return fallback

        prev_preview = "(No previous chunk)"
        if position > 0:
            prev_preview = file_chunks[position - 1].chunk_str(
                max_chars=max_chars, truncate="leading"
            )

        next_preview = "(No next chunk)"
        if position < len(file_chunks) - 1:
            next_preview = file_chunks[position + 1].chunk_str(
                max_chars=max_chars, truncate="trailing"
            )

        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": prev_preview,
            "next_chunk_preview": next_preview,
        }

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return chunks from files at the shallowest directory depth.

        Returns an empty list when file-structure metadata is unavailable
        or the namespace is too large (>50K rows) to enumerate efficiently.
        """
        if not self._files.check():
            return []

        # Skip expensive full-namespace pagination for large namespaces.
        # Use actual row count (not max_id) to handle sparse ID spaces where
        # max_id >> row_count due to deletions or non-sequential assignment.
        all_ids = self._client.paginate_all_ids()
        if len(all_ids) > 50_000:
            return []

        all_paths = self._files.get_all_file_paths()
        if not all_paths:
            return []

        path_depths = {p: p.count("/") for p in all_paths}
        min_depth = min(path_depths.values())
        top_level_paths = [p for p, d in path_depths.items() if d == min_depth]

        chunks: list[Chunk] = []
        for path in top_level_paths:
            chunks.extend(self._files.get_file_chunks(path))
        return chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search for chunks related to *source*.

        Args:
            mode: Search mode override. Defaults to lexical. Set to "hybrid"
                or "vector" to use embeddings (requires embed_fn).
            hybrid: Optional hybrid weight overrides (lexical_weight,
                vector_weight).

        Deduplicates by row ID, skips the source chunk and adjacent
        neighbors when file-structure metadata is available.

        Returns:
            List of dicts sorted by relevance, each containing:
            chunk, queries, same_file, max_score.
        """
        source_tpuf_id = source.get_metadata("_tpuf_id")
        related_map: dict[int, dict] = {}

        file_aware = self._files.check()
        source_file = self._files.chunk_file_path(source) if file_aware else None
        source_idx = self._files.chunk_index(source) if file_aware else None

        effective_mode = mode or _DEFAULT_RELATED_SEARCH_MODE

        # Batch-embed all queries only when vector/hybrid is explicitly requested.
        vectors: list[list[float]] | None = None
        if effective_mode in {"vector", "hybrid"}:
            if self._client.embed_fn is None:
                raise UnsupportedSearchModeError(
                    backend="turbopuffer",
                    mode=effective_mode,
                    supported_modes={str(m) for m in self._search_capabilities["modes"]},
                )
            vectors = self._client.embed_fn(queries)

        for i, query in enumerate(queries):
            spec: SearchSpec = {
                "mode": effective_mode,
                "top_k": top_k,
            }
            if effective_mode in {"lexical", "hybrid"}:
                spec["text_query"] = query
            if effective_mode in {"vector", "hybrid"} and vectors is not None:
                spec["vector_query"] = vectors[i]
            if hybrid is not None and effective_mode == "hybrid":
                spec["hybrid"] = hybrid

            rows = self._query_rows(spec)

            for row in rows:
                # Skip source chunk
                if source_tpuf_id is not None:
                    if row.id == source_tpuf_id:
                        continue
                else:
                    if self._client.row_content(row) == source.content:
                        continue

                result_chunk = self._client.row_to_chunk(row)

                # File-awareness: detect same-file, skip adjacent
                is_same_file = False
                if source_file is not None:
                    result_file = self._files.chunk_file_path(result_chunk)
                    is_same_file = result_file == source_file

                    if is_same_file and source_idx is not None:
                        result_idx = self._files.chunk_index(result_chunk)
                        if result_idx is not None and abs(result_idx - source_idx) <= 1:
                            continue

                score = getattr(row, "$dist", 0.0) or 0.0

                if row.id not in related_map:
                    related_map[row.id] = {
                        "chunk": result_chunk,
                        "queries": [],
                        "same_file": is_same_file,
                        "max_score": score,
                    }
                else:
                    related_map[row.id]["max_score"] = max(related_map[row.id]["max_score"], score)

                related_map[row.id]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (
                len(x["queries"]),
                not x["same_file"],
                x["max_score"],
            ),
            reverse=True,
        )

    def probe_rank_by_payloads(self, query: str = "cgft hybrid probe") -> dict[str, dict[str, Any]]:
        """Probe which rank_by payload shapes are accepted by the live namespace.

        This is a diagnostic tool — it fires real queries against the namespace.
        It is *not* called automatically; use it to debug namespace compatibility.

        Returns a dict keyed by payload type ("lexical", "vector",
        "native_hybrid") with "accepted", "payload", and "error" for each.
        """
        results: dict[str, dict[str, Any]] = {}

        # Lexical probe
        bm25_payload = self._client.build_bm25_rank_by(query)
        results["lexical"] = self._probe_rank_payload(bm25_payload)

        # Vector / native hybrid probes require embeddings
        if self._client.embed_fn is None:
            msg = "embed_fn unavailable"
            results["vector"] = {"accepted": False, "payload": None, "error": msg}
            results["native_hybrid"] = {"accepted": False, "payload": None, "error": msg}
            return results

        try:
            vector_query = self._client.embed_fn([query])[0]
        except Exception as exc:
            msg = str(exc)
            results["vector"] = {"accepted": False, "payload": None, "error": msg}
            results["native_hybrid"] = {"accepted": False, "payload": None, "error": msg}
            return results

        results["vector"] = self._probe_rank_payload(
            self._client.build_vector_rank_by(vector_query)
        )
        results["native_hybrid"] = self._probe_rank_payload(
            self._client.build_native_hybrid_rank_by(query, vector_query, None)
        )
        return results

    def _probe_rank_payload(self, payload: Any) -> dict[str, Any]:
        try:
            self._client.ns.query(rank_by=payload, top_k=1, include_attributes=True)
            return {"accepted": True, "payload": payload, "error": None}
        except Exception as exc:
            return {"accepted": False, "payload": payload, "error": str(exc)}

    def _build_query_kwargs(self, spec: SearchSpec) -> dict[str, Any]:
        """Validate spec and build turbopuffer query kwargs.

        Handles lexical and vector modes. Hybrid is routed through
        ``_query_rows_hybrid_fused`` before reaching this method.
        """
        mode = spec.get("mode")
        supported_modes = self._search_capabilities["modes"]
        if mode not in supported_modes:
            raise UnsupportedSearchModeError(
                backend="turbopuffer",
                mode=str(mode),
                supported_modes={str(m) for m in supported_modes},
            )

        shape_errors = validate_search_spec_shape(spec)
        if shape_errors:
            raise InvalidSearchSpecError(
                backend="turbopuffer",
                message="; ".join(shape_errors),
                spec=spec,
            )

        text_query = str(spec.get("text_query") or "")
        vector_query: list[float] | None = spec.get("vector_query")

        if mode == "lexical":
            rank_by = self._client.build_bm25_rank_by(text_query)
        elif mode == "vector":
            rank_by = self._client.build_vector_rank_by(vector_query)  # type: ignore[arg-type]
        else:
            # Hybrid specs that reach here have already been validated
            # by _query_rows; build native hybrid rank_by.
            rank_by = self._client.build_native_hybrid_rank_by(
                text_query,
                vector_query,
                spec.get("hybrid"),  # type: ignore[arg-type]
            )

        query_kwargs: dict[str, Any] = {
            "rank_by": rank_by,
            "top_k": int(spec.get("top_k", 10)),
            "include_attributes": True,
        }
        translated_filters = to_turbopuffer_filters(spec.get("filter"), self._search_capabilities)
        if translated_filters is not None:
            query_kwargs["filters"] = translated_filters

        return query_kwargs

    def _query_rows(self, spec: SearchSpec) -> list[Any]:
        """Execute a search spec and return raw tpuf rows.

        Hybrid mode always uses client-side RRF fusion (two separate
        lexical + vector queries merged by reciprocal rank).
        """
        mode = spec.get("mode")
        if mode == "hybrid":
            return self._query_rows_hybrid_fused(spec)

        query_kwargs = self._build_query_kwargs(spec)
        result = self._client.ns.query(**query_kwargs)
        return list(result.rows)

    def _query_rows_hybrid_fused(self, spec: SearchSpec) -> list[Any]:
        """Execute hybrid search via client-side RRF fusion.

        Issues separate lexical and vector queries, then merges results
        using Reciprocal Rank Fusion.
        """
        text_query = str(spec.get("text_query") or "")
        vector_query: list[float] | None = spec.get("vector_query")
        hybrid_opts: HybridOptions | None = spec.get("hybrid")
        if vector_query is None:
            raise InvalidSearchSpecError(
                backend="turbopuffer",
                message="hybrid mode requires vector_query",
                spec=spec,
            )

        requested_top_k = int(spec.get("top_k", 10))
        max_top_k = int(self._search_capabilities["constraints"].get("max_top_k", 10000))
        oversampled_top_k = min(max_top_k, requested_top_k * 2)

        translated_filters = to_turbopuffer_filters(spec.get("filter"), self._search_capabilities)

        lexical_kwargs: dict[str, Any] = {
            "rank_by": self._client.build_bm25_rank_by(text_query),
            "top_k": oversampled_top_k,
            "include_attributes": True,
        }
        vector_kwargs: dict[str, Any] = {
            "rank_by": self._client.build_vector_rank_by(vector_query),
            "top_k": oversampled_top_k,
            "include_attributes": True,
        }
        if translated_filters is not None:
            lexical_kwargs["filters"] = translated_filters
            vector_kwargs["filters"] = translated_filters

        lexical_rows = list(self._client.ns.query(**lexical_kwargs).rows)
        vector_rows = list(self._client.ns.query(**vector_kwargs).rows)

        opts = hybrid_opts or {}
        lexical_weight = float(opts.get("lexical_weight", 1.0))
        vector_weight = float(opts.get("vector_weight", 1.0))

        fused: dict[int, dict[str, Any]] = {}
        for rank, row in enumerate(lexical_rows, start=1):
            entry = fused.setdefault(row.id, {"row": row, "score": 0.0})
            entry["score"] += lexical_weight / (_HYBRID_FUSION_RRF_K + rank)
        for rank, row in enumerate(vector_rows, start=1):
            entry = fused.setdefault(row.id, {"row": row, "score": 0.0})
            entry["score"] += vector_weight / (_HYBRID_FUSION_RRF_K + rank)

        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        return [item["row"] for item in ranked[:requested_top_k]]

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
        return [self._client.row_to_chunk(row) for row in self._query_rows(spec)]

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings without Chunk construction.

        Cloudpickle-safe alternative to ``search()`` for use in remote envs
        where Pydantic models don't survive pickle roundtripping.
        """
        return [self._client.row_content(row) for row in self._query_rows(spec)]

    def embed_query(self, text: str) -> list[float] | None:
        """Return an embedding vector for *text*, or ``None``."""
        if self._client.embed_fn is None:
            return None
        return self._client.embed_fn([text])[0]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query and optional filter."""
        return self.search(
            SearchSpec(
                mode="lexical",
                text_query=text_query,
                top_k=top_k,
                filter=filter,
            )
        )

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for Turbopuffer backend."""
        return self._search_capabilities
