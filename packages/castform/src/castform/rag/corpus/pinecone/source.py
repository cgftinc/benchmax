"""PineconeChunkSource — ChunkSource implementation backed by Pinecone."""

from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Callable
from typing import Any

from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidSearchSpecError,
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import (
    FilterPredicate,
    SearchCapabilities,
    SearchSpec,
    validate_search_spec_shape,
)

from .files import FileAwareness
from .filter_mapper import to_pinecone_filters
from .index_client import PineconeIndexClient

logger = logging.getLogger(__name__)

#: Max IDs per vectors/fetch call — Pinecone caps fetch batches at 100.
_FETCH_BATCH_SIZE = 100


def _raw_to_chunk(raw: dict[str, Any]) -> Chunk:
    """Convert a raw dict from PineconeIndexClient to a Chunk."""
    return Chunk(
        content=raw["content"],
        metadata=tuple(raw["metadata"].items()),
    )


class PineconeChunkSource:
    """ChunkSource backed by a Pinecone index.

    Chunks are stored in Pinecone with vector embeddings.  Only vector
    search is supported (no BM25/lexical).

    Embedding is handled automatically via Pinecone's hosted Inference
    API (default model ``"multilingual-e5-large"``).  Pass a custom
    ``embed_fn`` to use an external embedding provider instead.

    File-structure awareness (neighboring chunk context, top-level chunk
    retrieval, adjacent-chunk filtering in search_related) is supported when
    the index contains ``file_path`` and ``chunk_index`` metadata fields.
    These are written automatically by ``populate_from_folder`` /
    ``populate_from_chunks``.  For pre-existing indexes that lack these
    fields, all file-aware methods gracefully degrade (empty context, empty
    top-level list, no neighbor filtering).

    Args:
        api_key: Pinecone API key.
        index_name: Name of the Pinecone index.
        index_host: Optional host URL (bypasses index name lookup).
        namespace: Pinecone namespace within the index (default ``""``).
        embed_fn: Custom embedding function ``list[str] → list[list[float]]``.
            When ``None``, Pinecone's Inference API is used with
            ``embed_model``.
        embed_model: Pinecone hosted embedding model name.  Ignored when
            ``embed_fn`` is provided.  Defaults to
            ``"multilingual-e5-large"``.
        field_mapping: Low-level escape hatch — maps Pinecone metadata field
            names to internal names when structural fields (``file_path``,
            ``chunk_index``, headers) are also relocated.  For the common
            case, prefer ``content_field``.
        content_field: Pinecone metadata key holding the chunk text — the
            canonical way to point at your text column for pre-existing
            indexes that don't use ``content``.

    Example:
        >>> # Using Pinecone's built-in embeddings (simplest)
        >>> source = PineconeChunkSource(
        ...     api_key="pcsk_...",
        ...     index_name="my-docs",
        ... )
        >>> source.populate_from_folder("./docs")

        >>> # Using a custom embedding function
        >>> source = PineconeChunkSource(
        ...     api_key="pcsk_...",
        ...     index_name="my-docs",
        ...     embed_fn=my_embed_fn,
        ... )

        >>> # Pre-existing index whose text lives under another key
        >>> source = PineconeChunkSource(
        ...     api_key="pcsk_...",
        ...     index_name="product-catalog",
        ...     embed_model="llama-text-embed-v2",
        ...     content_field="description",
        ... )
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        *,
        index_host: str | None = None,
        namespace: str = "",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_model: str = "multilingual-e5-large",
        field_mapping: dict[str, str] | None = None,
        content_field: str | None = None,
    ) -> None:
        self._client = PineconeIndexClient(
            api_key=api_key,
            index_name=index_name,
            index_host=index_host,
            namespace=namespace,
            embed_fn=embed_fn,
            embed_model=embed_model,
            field_mapping=field_mapping,
            content_field=content_field,
        )
        self._files = FileAwareness(self._client)

        self._search_capabilities: SearchCapabilities = {
            "backend": "pinecone",
            "modes": {"vector"},
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": {"cosine"},
            "constraints": {"max_top_k": 10000, "vector_dimensions": None},
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
        batch_size: int = 96,
        show_summary: bool = True,
    ) -> None:
        """Chunk documents in a folder and upload them to Pinecone.

        Args:
            docs_path: Path to the folder containing documents to chunk.
            embed_fn: Accepted for ChunkSource protocol compatibility but
                **ignored** — the instance-level ``embed_fn`` is always used
                since Pinecone requires vector embeddings.
            min_chars: Minimum characters per chunk (default 1024).
            max_chars: Maximum characters per chunk (default 2048).
            overlap_chars: Character overlap between adjacent chunks.
            file_extensions: File types to process (default [".md", ".mdx"]).
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print chunking summary and upload progress.
        """
        from castform.rag.chunkers.inspector import ChunkInspector
        from castform.rag.chunkers.markdown import MarkdownChunker

        if embed_fn is not None:
            warnings.warn(
                "PineconeChunkSource.populate_from_folder: embed_fn argument "
                "is ignored — the instance-level embed_fn is always used.",
                UserWarning,
                stacklevel=2,
            )

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
        self._files.invalidate()

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        batch_size: int = 96,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to Pinecone.

        Args:
            collection: ChunkCollection produced by any chunker.
            embed_fn: Accepted for ChunkSource protocol compatibility but
                **ignored** — the instance-level ``embed_fn`` is always used.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print upload progress (default True).
        """
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in collection:
            ids.append(chunk.hash or f"chunk-{len(ids) + 1}")
            documents.append(chunk.content)
            metadatas.append(
                {
                    self._client._pc_field("content"): chunk.content,
                    self._client._pc_field("file_path"): chunk.get_metadata("file", ""),
                    self._client._pc_field("h1"): chunk.get_metadata("h1", ""),
                    self._client._pc_field("h2"): chunk.get_metadata("h2", ""),
                    self._client._pc_field("h3"): chunk.get_metadata("h3", ""),
                    self._client._pc_field("chunk_index"): chunk.get_metadata("index", 0),
                    self._client._pc_field("chunk_hash"): chunk.hash,
                    self._client._pc_field("char_count"): len(chunk),
                }
            )
        self._client.upsert_raw(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            batch_size=batch_size,
            show_summary=show_summary,
        )
        self._files.invalidate()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_chunk_count(self) -> int:
        """Return the number of vectors in the configured namespace.

        Scoped to the namespace this source reads from — an index-wide
        total would disagree with what sampling/search can actually see.
        """
        return self._client.namespace_vector_count()

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by
        minimum length.

        Samples uniformly from the paginated ID listing and hydrates the
        sample via fetch — no query vector involved, so the draw is
        genuinely uniform (not nearest-to-a-random-point) and works for
        dense and sparse indexes alike.
        """
        # Oversample when a length filter will discard part of the draw
        fetch_n = min(n * 3, 10000) if min_chars > 0 else min(n, 10000)
        ids = self._client.sample_ids(fetch_n)
        if not ids:
            return []

        raws: list[dict[str, Any]] = []
        for batch_start in range(0, len(ids), _FETCH_BATCH_SIZE):
            raws.extend(
                self._client.fetch_by_ids_raw(
                    ids[batch_start : batch_start + _FETCH_BATCH_SIZE]
                )
            )
        chunks = [_raw_to_chunk(r) for r in raws]

        # Every fetched record decoding to empty content means the text key
        # is wrong (BYO index whose schema doesn't use the configured field),
        # not that the corpus is empty. Without this, the pipeline dies later
        # with an unactionable "No eligible chunks were found".
        if chunks and all(not c.content for c in chunks):
            content_key = self._client._pc_field("content")
            seen_keys = sorted(
                {k for r in raws for k in r.get("metadata", {}) if not k.startswith("_")}
            )
            raise ValueError(
                f"No text found under metadata field '{content_key}' in any "
                f"sampled record. This index's metadata fields are: "
                f"{seen_keys}. Set content_field to the one holding the "
                f"chunk text."
            )

        if min_chars > 0:
            chunks = [c for c in chunks if len(c.content) >= min_chars]

        random.shuffle(chunks)
        return chunks[:n]

    # ------------------------------------------------------------------
    # Context & structure
    # ------------------------------------------------------------------

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return chunk with neighboring context from the same file."""
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

    _TOP_LEVEL_MAX_VECTORS = 50_000

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return chunks from files at the shallowest directory depth.

        Returns an empty list when file-structure metadata is unavailable
        or the index is too large (>50K vectors) to enumerate efficiently.
        """
        if not self._files.check():
            return []

        # Skip expensive full-index pagination for large indexes.
        index = self._client._get_index()
        stats = index.describe_index_stats()
        if stats.total_vector_count > self._TOP_LEVEL_MAX_VECTORS:
            logger.warning(
                "Skipping get_top_level_chunks: index has %d vectors (limit %d)",
                stats.total_vector_count,
                self._TOP_LEVEL_MAX_VECTORS,
            )
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
        mode: str | None = None,
        hybrid: dict | None = None,
    ) -> list[dict]:
        """Search for chunks related to *source* using vector queries.

        Each query string is embedded and used for ANN search.
        Deduplicates by vector ID, skips the source chunk and adjacent
        neighbors when file-structure metadata is available.

        Args:
            mode: Accepted for protocol compatibility but ignored —
                Pinecone always uses vector search.
            hybrid: Accepted for protocol compatibility but ignored.

        Returns:
            List of dicts sorted by relevance, each containing:
            chunk, queries, same_file, max_score.
        """
        source_pc_id = source.get_metadata("_pinecone_id")
        related_map: dict[str, dict] = {}

        file_aware = self._files.check()
        source_file = self._files.chunk_file_path(source) if file_aware else None
        source_idx = self._files.chunk_index(source) if file_aware else None

        # Batch-embed all queries
        vectors = self._client.embed_fn(queries)

        for i, query in enumerate(queries):
            result = self._client.query(
                vector=vectors[i],
                top_k=top_k,
                include_metadata=True,
            )

            for match in result.matches or []:
                # Skip source chunk
                if source_pc_id is not None:
                    if match.id == source_pc_id:
                        continue
                else:
                    if self._client.match_content(match) == source.content:
                        continue

                result_chunk = _raw_to_chunk(self._client.match_to_raw(match))

                # File-awareness: detect same-file, skip adjacent
                is_same_file = False
                if source_file is not None:
                    result_file = self._files.chunk_file_path(result_chunk)
                    is_same_file = result_file == source_file

                    if is_same_file and source_idx is not None:
                        result_idx = self._files.chunk_index(result_chunk)
                        if result_idx is not None and abs(result_idx - source_idx) <= 1:
                            continue

                score = getattr(match, "score", 0.0) or 0.0

                if match.id not in related_map:
                    related_map[match.id] = {
                        "chunk": result_chunk,
                        "queries": [],
                        "same_file": is_same_file,
                        "max_score": score,
                    }
                else:
                    related_map[match.id]["max_score"] = max(
                        related_map[match.id]["max_score"], score
                    )

                related_map[match.id]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (
                len(x["queries"]),
                not x["same_file"],
                x["max_score"],
            ),
            reverse=True,
        )

    def _build_query_kwargs(self, spec: SearchSpec) -> dict[str, Any]:
        """Validate spec and build Pinecone query kwargs."""
        mode = spec.get("mode")
        supported_modes = set(self._search_capabilities.get("modes", set()))
        if mode not in supported_modes:
            raise UnsupportedSearchModeError(
                backend=str(self._search_capabilities.get("backend", "unknown")),
                mode=str(mode),
                supported_modes={str(m) for m in supported_modes},
            )

        shape_errors = validate_search_spec_shape(spec)
        if shape_errors:
            raise InvalidSearchSpecError(
                backend=str(self._search_capabilities.get("backend", "unknown")),
                message="; ".join(shape_errors),
                spec=spec,
            )

        vector_query: list[float] | None = spec.get("vector_query")
        if vector_query is None:
            raise InvalidSearchSpecError(
                backend="pinecone",
                message="vector_query is required for Pinecone",
                spec=spec,
            )

        query_kwargs: dict[str, Any] = {
            "vector": vector_query,
            "top_k": int(spec.get("top_k", 10)),
            "include_metadata": True,
        }
        translated_filters = to_pinecone_filters(spec.get("filter"), self._search_capabilities)
        if translated_filters is not None:
            query_kwargs["filter"] = translated_filters

        return query_kwargs

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
        query_kwargs = self._build_query_kwargs(spec)
        result = self._client.query(**query_kwargs)
        return [_raw_to_chunk(self._client.match_to_raw(m)) for m in (result.matches or [])]

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings without Chunk construction.

        Cloudpickle-safe alternative to ``search()`` for use in remote envs
        where Pydantic models don't survive pickle roundtripping.
        """
        query_kwargs = self._build_query_kwargs(spec)
        result = self._client.query(**query_kwargs)
        return [self._client.match_content(match) for match in (result.matches or [])]

    def embed_query(self, text: str) -> list[float] | None:
        """Return an embedding vector for *text*."""
        return self._client.embed_fn([text])[0]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query (auto-embedded) and optional filter."""
        vector = self.embed_query(text_query)
        return self.search(
            SearchSpec(
                mode="vector",
                text_query=text_query,
                vector_query=vector,
                top_k=top_k,
                filter=filter,
            )
        )

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for Pinecone backend."""
        return self._search_capabilities
