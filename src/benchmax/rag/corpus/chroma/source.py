"""ChromaChunkSource -- ChunkSource implementation backed by ChromaDB.

Delegates all SDK operations to :class:`ChromaClient` (from ``.client``).
This module handles only Chunk-specific logic: converting raw dicts to
``Chunk`` objects, ``ChunkCollection`` handling, and the ``ChunkSource``
protocol surface.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, cast

from tqdm.auto import tqdm

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

from .client import ChromaClient
from .files import FileAwareness
from .filter_mapper import to_chroma_filters


class ChromaChunkSource:
    """ChunkSource backed by a ChromaDB collection.

    Supports three retrieval modes:

    * **vector** -- dense embedding similarity (always available).
    * **lexical** -- native BM25 via Chroma's sparse-vector index.
    * **hybrid** -- Reciprocal Rank Fusion of dense + sparse rankings.

    Lexical and hybrid modes require ``chromadb >= 1.0`` with the
    ``Search`` API and a client-server connection (sparse indexing is
    not available on local/in-memory Chroma).  When unavailable the
    source gracefully degrades to vector-only.

    The client is lazily initialized on first use so that the source
    survives ``pickle`` round-trips (only connection parameters are
    stored).  For remote training environments, only **client-server**
    mode (``host``/``port``) is safe -- local/in-memory collections
    cannot be transferred to a different process.

    Args:
        collection_name: Name of the Chroma collection.
        host: Chroma server hostname for client-server mode.
        port: Chroma server port (default 8000).
        path: Directory for persistent-local mode (mutually exclusive
            with ``host``).  When neither ``host`` nor ``path`` is given,
            an ephemeral in-memory client is created.
        embed_fn: Optional embedding callable ``list[str] -> list[list[float]]``.
            When provided, Chroma's default dense embedding function is
            replaced.
        content_attr: List of metadata field names to treat as searchable
            text content.  Defaults to ``["content"]``.  Useful for
            pre-existing collections where the text lives in a different
            field (analogous to ``content_attr`` on ``TpufChunkSource``).
        distance_metric: Distance function for the dense index.  One of
            ``"cosine"``, ``"l2"``, ``"ip"`` (default ``"cosine"``).
            Applied both with and without BM25 -- controls the HNSW index.
        enable_bm25: Create a sparse BM25 index on the collection for
            lexical / hybrid search (default ``True``).  Only takes
            effect when client-server mode and the Search API are both
            available.
        rrf_k: RRF smoothing constant for hybrid search (default 60).
            Higher values flatten rank differences.
        rrf_oversample: Multiplier for candidate pool in each RRF leg,
            capped at ``rrf_max_candidates`` (default 20x).
        rrf_max_candidates: Hard cap on per-leg candidate count
            (default 200).

    Example:
        >>> source = ChromaChunkSource(
        ...     collection_name="my-docs",
        ...     host="chroma.internal",
        ... )
        >>> source.populate_from_folder("./docs")
        >>> chunks = source.sample_chunks(n=10, min_chars=400)
    """

    def __init__(
        self,
        collection_name: str,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int = 8000,
        path: str | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        content_attr: list[str] | None = None,
        distance_metric: str = "cosine",
        enable_bm25: bool = True,
        rrf_k: int = 60,
        rrf_oversample: int = 20,
        rrf_max_candidates: int = 200,
    ) -> None:
        self._chroma = ChromaClient(
            collection_name=collection_name,
            api_key=api_key,
            tenant=tenant,
            database=database,
            host=host,
            port=port,
            path=path,
            embed_fn=embed_fn,
            content_attr=content_attr,
            distance_metric=distance_metric,
            enable_bm25=enable_bm25,
            rrf_k=rrf_k,
            rrf_oversample=rrf_oversample,
            rrf_max_candidates=rrf_max_candidates,
        )

        self._files = FileAwareness(self._chroma.get_collection)

        # Build SearchCapabilities from ChromaClient's modes/ranking
        self._search_capabilities: SearchCapabilities = {
            "backend": "chroma",
            "modes": cast(set[SearchMode], self._chroma.modes),
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": set(self._chroma.ranking),
            "constraints": {"max_top_k": 10000, "vector_dimensions": None},
            "graph_expansion": False,
        }

    @property
    def is_client_server(self) -> bool:
        """True when configured for HTTP client-server mode."""
        return self._chroma.is_client_server

    # ------------------------------------------------------------------
    # Pickle
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Strip live caches for pickle safety.

        Connection params, config, and search capabilities are preserved.
        The ChromaClient and file-awareness caches are rebuilt lazily
        after unpickling.
        """
        state = self.__dict__.copy()
        state["_files"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._files = FileAwareness(self._chroma.get_collection)
        # Sync capabilities in case ChromaClient downgraded modes during
        # collection creation before pickling.
        self._search_capabilities["modes"] = cast(set[SearchMode], self._chroma.modes)
        self._search_capabilities["ranking"] = set(self._chroma.ranking)

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
        """Chunk documents in a folder and upload them to Chroma.

        Args:
            docs_path: Path to the folder containing documents to chunk.
            embed_fn: Ignored (kept for API parity). Use the constructor
                ``embed_fn`` or Chroma's default embedding.
            min_chars: Minimum characters per chunk (default 1024).
            max_chars: Maximum characters per chunk (default 2048).
            overlap_chars: Character overlap between adjacent chunks.
            file_extensions: File types to process (default [".md", ".mdx"]).
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print chunking summary and upload progress.
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
            batch_size=batch_size,
            show_summary=show_summary,
        )

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to ChromaDB.

        Args:
            collection: ChunkCollection produced by any chunker.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print upload progress (default True).
        """
        col = self._chroma.get_collection()
        all_chunks = list(collection)
        collection_name = self._chroma.collection_name

        if show_summary:
            print(
                f"\nUploading {len(all_chunks)} chunks to Chroma collection '{collection_name}'..."
            )

        for batch_start in tqdm(
            range(0, len(all_chunks), batch_size),
            desc="Uploading batches",
            disable=not show_summary,
        ):
            batch = all_chunks[batch_start : batch_start + batch_size]

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for chunk in batch:
                ids.append(chunk.hash)
                documents.append(chunk.content)
                metadatas.append(
                    {
                        "file_path": chunk.get_metadata("file", ""),
                        "h1": chunk.get_metadata("h1", ""),
                        "h2": chunk.get_metadata("h2", ""),
                        "h3": chunk.get_metadata("h3", ""),
                        "chunk_index": chunk.get_metadata("index", 0),
                        "chunk_hash": chunk.hash,
                        "char_count": len(chunk),
                    }
                )

            col.upsert(ids=ids, documents=documents, metadatas=metadatas)

        self._chroma._total_count = len(all_chunks)
        self._files.invalidate()

        if show_summary:
            print(
                f"\nUpload complete! {len(all_chunks)} chunks written to"
                f" collection '{collection_name}'."
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_chunk_count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self._chroma.get_total_count()

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by length."""
        total = self._chroma.get_total_count()
        if total == 0:
            return []

        col = self._chroma.get_collection()

        if min_chars == 0:
            sample_size = min(n, total)
            if sample_size == total:
                result = col.get(include=["documents", "metadatas"], limit=total)
                return self._get_results_to_chunks(result)

            indices = sorted(random.sample(range(total), sample_size))
            chunks: list[Chunk] = []
            for idx in indices:
                result = col.get(
                    include=["documents", "metadatas"],
                    limit=1,
                    offset=idx,
                )
                chunks.extend(self._get_results_to_chunks(result))
            return chunks

        # With min_chars: fetch pages at random offsets, filter locally
        collected: list[Chunk] = []
        page_size = 200
        offsets = list(range(0, total, page_size))
        random.shuffle(offsets)

        for start in offsets:
            result = col.get(
                include=["documents", "metadatas"],
                limit=page_size,
                offset=start,
            )
            for chunk in self._get_results_to_chunks(result):
                if len(chunk.content) >= min_chars:
                    collected.append(chunk)
                    if len(collected) >= n:
                        return collected[:n]

        return collected

    def _get_results_to_chunks(self, result: dict[str, Any]) -> list[Chunk]:
        """Convert a Chroma ``get()`` result dict to Chunks via ChromaClient."""
        rows = ChromaClient._get_result_to_raw(result)
        return [
            Chunk(
                content=self._chroma.extract_content(r["content"], r["metadata"]),
                metadata=tuple(r["metadata"].items()),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Context & structure
    # ------------------------------------------------------------------

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return chunk with neighboring context from the same file."""
        fallback: dict[str, str] = {
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
        """Return chunks from files at the shallowest directory depth."""
        if not self._files.check():
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
    # Embedding
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float] | None:
        """Return a dense embedding vector for *text*, or ``None``.

        Uses the custom ``embed_fn`` if provided.  Otherwise returns
        ``None`` -- Chroma will auto-embed via ``query_texts``.
        """
        return self._chroma.embed(text)

    # ------------------------------------------------------------------
    # Search validation
    # ------------------------------------------------------------------

    def _validate_spec(self, spec: SearchSpec) -> None:
        """Validate mode and shape, raising on error.

        Chroma can auto-embed text queries in vector mode, so we relax
        the standard shape validation that would require ``vector_query``
        -- a ``text_query`` is sufficient.
        """
        mode = spec.get("mode")
        supported_modes = self._search_capabilities["modes"]
        if mode not in supported_modes:
            raise UnsupportedSearchModeError(
                backend="chroma",
                mode=mode or "None",
                supported_modes={m for m in supported_modes},
            )

        shape_errors = validate_search_spec_shape(spec)

        # Chroma auto-embeds text queries via its built-in embedding
        # function, so vector_query is never strictly required -- a
        # text_query is sufficient for vector and hybrid modes.
        if mode in ("vector", "hybrid") and shape_errors:
            text_query = spec.get("text_query")
            has_text = isinstance(text_query, str) and bool(text_query.strip())
            shape_errors = [e for e in shape_errors if not (has_text and "vector_query" in e)]

        if shape_errors:
            raise InvalidSearchSpecError(
                backend="chroma",
                message="; ".join(shape_errors),
                spec=spec,
            )

    # ------------------------------------------------------------------
    # Raw results -> Chunks (delegates to ChromaClient)
    # ------------------------------------------------------------------

    def _raw_rows_to_chunks(self, rows: list[dict[str, Any]]) -> list[Chunk]:
        """Convert raw dicts from ChromaClient to Chunks."""
        return [
            Chunk(
                content=self._chroma.extract_content(r["content"], r["metadata"]),
                metadata=tuple(r["metadata"].items()),
            )
            for r in rows
        ]

    def _raw_rows_to_scored_chunks(self, rows: list[dict[str, Any]]) -> list[tuple[Chunk, float]]:
        """Convert raw dicts from ChromaClient to (Chunk, score) pairs."""
        return [
            (
                Chunk(
                    content=self._chroma.extract_content(r["content"], r["metadata"]),
                    metadata=tuple(r["metadata"].items()),
                ),
                float(r.get("score", 0.0)),
            )
            for r in rows
        ]

    def _raw_rows_to_strings(self, rows: list[dict[str, Any]]) -> list[str]:
        """Convert raw dicts from ChromaClient to content strings."""
        return [self._chroma.extract_content(r["content"], r["metadata"]) for r in rows]

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
        self._validate_spec(spec)
        mode = spec.get("mode")

        where = to_chroma_filters(spec.get("filter"), self._search_capabilities)

        if self._chroma.search_api and mode in ("lexical", "hybrid"):
            rows = self._chroma.search_api_raw(
                text_query=spec.get("text_query") or "",
                vector_query=spec.get("vector_query"),
                mode=mode,
                top_k=spec.get("top_k", 10),
                where=where,
                hybrid_opts=cast(dict[str, Any] | None, spec.get("hybrid")),
            )
            return self._raw_rows_to_chunks(rows)

        # Vector mode (or fallback when Search API unavailable)
        rows = self._chroma.query_raw(
            text_query=spec.get("text_query"),
            vector_query=spec.get("vector_query"),
            top_k=spec.get("top_k", 10),
            where=where,
        )
        return self._raw_rows_to_chunks(rows)

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings without Chunk construction.

        Cloudpickle-safe alternative to ``search()`` for use in remote
        envs where Pydantic models don't survive pickle roundtripping.
        """
        self._validate_spec(spec)
        mode = spec.get("mode")

        where = to_chroma_filters(spec.get("filter"), self._search_capabilities)

        if self._chroma.search_api and mode in ("lexical", "hybrid"):
            rows = self._chroma.search_api_raw(
                text_query=spec.get("text_query") or "",
                vector_query=spec.get("vector_query"),
                mode=mode,
                top_k=spec.get("top_k", 10),
                where=where,
                hybrid_opts=cast(dict[str, Any] | None, spec.get("hybrid")),
            )
            return self._raw_rows_to_strings(rows)

        rows = self._chroma.query_raw(
            text_query=spec.get("text_query"),
            vector_query=spec.get("vector_query"),
            top_k=spec.get("top_k", 10),
            where=where,
        )
        return self._raw_rows_to_strings(rows)

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query and optional filter.

        Picks the best available mode: hybrid > lexical > vector.
        """
        modes = self._current_modes()
        if "hybrid" in modes:
            vector = self.embed_query(text_query)
            return self.search(
                SearchSpec(
                    mode="hybrid",
                    top_k=top_k,
                    text_query=text_query,
                    vector_query=vector,
                    filter=filter,
                )
            )
        if "lexical" in modes:
            return self.search(
                SearchSpec(
                    mode="lexical",
                    top_k=top_k,
                    text_query=text_query,
                    filter=filter,
                )
            )
        # Vector-only fallback
        vector = self.embed_query(text_query)
        return self.search(
            SearchSpec(
                mode="vector",
                top_k=top_k,
                text_query=text_query,
                vector_query=vector,
                filter=filter,
            )
        )

    def _current_modes(self) -> set[SearchMode]:
        """Return live search modes, refreshing capabilities after lazy init.

        ``get_collection()`` is cached/idempotent; calling it ensures any
        capability downgrade (e.g. a collection lacking a BM25 index) has been
        applied before we read modes. ``_search_capabilities`` is frozen at
        construction — before that lazy init — so re-sync it from the client.
        """
        self._chroma.get_collection()
        self._search_capabilities["modes"] = cast(set[SearchMode], self._chroma.modes)
        self._search_capabilities["ranking"] = set(self._chroma.ranking)
        return self._search_capabilities["modes"]

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: str | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search for chunks related to *source*.

        Uses the best available mode.  When the Search API is available,
        actual distance scores from the backend are used for ranking.
        Otherwise falls back to the legacy query API with Chroma distance
        scores (lower = closer, converted to higher = better).

        Deduplicates by document ID, skips the source chunk and adjacent
        neighbors when file-structure metadata is available.

        Returns:
            List of dicts sorted by relevance, each containing:
            chunk, queries, same_file, max_score.
        """
        source_hash = source.hash

        file_aware = self._files.check()
        source_file = self._files.chunk_file_path(source) if file_aware else None
        source_idx = self._files.chunk_index(source) if file_aware else None

        # Refresh capabilities before choosing a mode: the backing collection may
        # lack a BM25 index, in which case modes was downgraded to vector-only.
        modes = self._current_modes()

        # Pick mode. "hybrid"/None use the best available strategy and KEEP
        # lexical enabled as a fallback: hybrid = dense + sparse, and when we
        # can't produce dense query vectors (no embed_fn, the usual remote case)
        # the per-query loop below degrades to the sparse/lexical leg — which
        # needs no embedding.
        #
        # An explicit "vector" (e.g. the linker's "inference" reasoning mode)
        # normally disables lexical. But if the dense embed would be the
        # chromadb-local fallback (no embed_fn, no hosted EF) we keep lexical
        # instead: a remote default-EF collection would otherwise download
        # all-MiniLM to embed every query. Server-side dense (hosted EF) or a
        # client embed_fn stays on the vector path — semantic search preserved.
        if mode == "vector":
            use_hybrid = False
            use_lexical = (
                "lexical" in modes and self._chroma.dense_requires_local_model()
            )
        elif mode == "lexical":
            use_hybrid = False
            use_lexical = "lexical" in modes
        else:  # "hybrid", None, or unrecognized -> best available
            use_hybrid = "hybrid" in modes
            use_lexical = "lexical" in modes

        # Batch-embed all queries when embed_fn available and vectors needed
        vectors: list[list[float]] | None = None
        if self._chroma.embed_fn is not None and (use_hybrid or not use_lexical):
            vectors = self._chroma.embed_fn(queries)

        related_map: dict[str, dict] = {}

        for i, query in enumerate(queries):
            if use_hybrid and vectors is not None:
                spec = SearchSpec(
                    mode="hybrid",
                    top_k=top_k,
                    text_query=query,
                    vector_query=vectors[i],
                )
            elif use_lexical:
                spec = SearchSpec(
                    mode="lexical",
                    top_k=top_k,
                    text_query=query,
                )
            else:
                spec = SearchSpec(
                    mode="vector",
                    top_k=top_k,
                    text_query=query,
                    vector_query=vectors[i] if vectors else None,
                )

            # Get scored results for proper ranking
            scored = self._search_with_scores(spec)

            for result_chunk, score in scored:
                doc_id = result_chunk.hash
                if doc_id == source_hash:
                    continue
                if result_chunk.content == source.content:
                    continue

                is_same_file = False
                if source_file is not None:
                    result_file = self._files.chunk_file_path(result_chunk)
                    is_same_file = result_file == source_file

                    if is_same_file and source_idx is not None:
                        result_idx = self._files.chunk_index(result_chunk)
                        if result_idx is not None and abs(result_idx - source_idx) <= 1:
                            continue

                if doc_id not in related_map:
                    related_map[doc_id] = {
                        "chunk": result_chunk,
                        "queries": [],
                        "same_file": is_same_file,
                        "max_score": score,
                    }
                else:
                    related_map[doc_id]["max_score"] = max(related_map[doc_id]["max_score"], score)

                related_map[doc_id]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (
                len(x["queries"]),
                not x["same_file"],
                x["max_score"],
            ),
            reverse=True,
        )

    def _search_with_scores(self, spec: SearchSpec) -> list[tuple[Chunk, float]]:
        """Search and return (Chunk, score) pairs.

        Uses actual backend scores -- Search API scores for lexical/hybrid,
        Chroma distances (converted: higher = better) for vector/legacy.
        """
        self._validate_spec(spec)
        mode = spec.get("mode")

        where = to_chroma_filters(spec.get("filter"), self._search_capabilities)

        if self._chroma.search_api and mode in ("lexical", "hybrid"):
            rows = self._chroma.search_api_raw(
                text_query=spec.get("text_query") or "",
                vector_query=spec.get("vector_query"),
                mode=mode,
                top_k=spec.get("top_k", 10),
                where=where,
                hybrid_opts=cast(dict[str, Any] | None, spec.get("hybrid")),
            )
            return self._raw_rows_to_scored_chunks(rows)

        # Legacy query() API -- distances (lower = closer)
        rows = self._chroma.query_raw(
            text_query=spec.get("text_query"),
            vector_query=spec.get("vector_query"),
            top_k=spec.get("top_k", 10),
            where=where,
        )
        return self._raw_rows_to_scored_chunks(rows)

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for ChromaDB backend.

        When BM25 is requested on a client-server connection, triggers
        lazy collection initialization so that capabilities reflect what
        the server actually supports (sparse indexing may be unavailable
        on older servers, causing a downgrade to vector-only).
        """
        if (
            self._chroma._collection is None
            and self._chroma.enable_bm25
            and self.is_client_server
            and "lexical" in self._search_capabilities["modes"]
        ):
            # Force collection creation to detect sparse index support
            self._chroma.get_collection()
            # Sync capabilities after potential downgrade
            self._search_capabilities["modes"] = cast(set[SearchMode], self._chroma.modes)
            self._search_capabilities["ranking"] = set(self._chroma.ranking)
        return self._search_capabilities
