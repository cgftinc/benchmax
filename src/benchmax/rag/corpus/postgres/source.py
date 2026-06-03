"""PostgresChunkSource — ChunkSource implementation backed by the Corpora API."""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

from benchmax import config
from benchmax.platform.credentials import resolve_token_provider
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

from .client import CorpusClient
from .filter_mapper import to_corpora_filters

if TYPE_CHECKING:
    from .models import Corpus


class PostgresChunkSource:
    """ChunkSource backed by local ChunkCollection + Corpora API BM25 search.

    Chunks are stored locally in a ChunkCollection and uploaded to the Corpora
    API to enable BM25 search. The collection is exposed as a public attribute
    for advanced users who want to leverage file-structure awareness directly.

    Args:
        corpus_name: Name of the corpus to create or reuse
        api_key: Corpora API key. Optional — an empty string (the default)
            resolves the bearer per request via the credential seam (cached
            device-auth session / ACT_AS_TOKEN_PATH / PLATFORM_API_KEY).
        base_url: Corpora API base URL. Optional — defaults to the
            session-derived platform URL (``config.platform_url()``).

    Example:
        >>> source = PostgresChunkSource(corpus_name="my-docs")
        >>> source.populate_from_folder("./docs")
        >>> chunks = source.sample_chunks(n=10, min_chars=400)
    """

    def __init__(
        self,
        corpus_name: str,
        api_key: str = "",
        base_url: str | None = None,
    ) -> None:
        # Per-request bearer: an explicit key wins; an empty key resolves via the
        # platform credential seam (CorpusClient no longer bakes a static key).
        # base_url defaults to the session-derived platform URL.
        self._client = CorpusClient(
            base_url=base_url if base_url is not None else config.platform_url(),
            token_provider=resolve_token_provider(api_key),
        )
        self._corpus_name = corpus_name
        self._corpus: Corpus | None = None
        self.collection: ChunkCollection | None = (
            None  # exposed publicly for advanced users
        )
        self._search_capabilities: SearchCapabilities = {
            "backend": "corpora",
            "modes": {"lexical"},
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte", "contains_any", "contains_all"},
                "logical": {"and", "or", "not"},
            },
            "ranking": {"bm25"},
            "constraints": {"max_top_k": 1000},
            "graph_expansion": True,
        }

    def populate_from_folder(
        self,
        docs_path: str,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Chunk documents in a folder and upload them to the Corpora API.

        Chunks the documents, creates or reuses the named corpus, and uploads
        all chunks for BM25 search. Sets self.collection after chunking.

        Args:
            docs_path: Path to the folder containing documents to chunk
            min_chars: Minimum characters per chunk (default 1024)
            max_chars: Maximum characters per chunk (default 2048)
            overlap_chars: Character overlap between adjacent chunks (default 128)
            file_extensions: File types to process (default [".md", ".mdx"])
            batch_size: Number of chunks per upload batch (default 100)
            show_summary: Print chunking summary and upload progress (default True)
        """
        from benchmax.rag.chunkers.inspector import ChunkInspector
        from benchmax.rag.chunkers.markdown import MarkdownChunker

        if file_extensions is None:
            file_extensions = [".md", ".mdx"]

        if show_summary:
            print(f"Chunking documents from {docs_path}...")

        chunker = MarkdownChunker(
            min_char=min_chars, max_char=max_chars, chunk_overlap=overlap_chars
        )
        collection = chunker.chunk_folder(docs_path, file_extensions=file_extensions)

        if show_summary:
            inspector = ChunkInspector(collection)
            inspector.summary(max_depth=3, max_files_per_folder=4)

        self.populate_from_chunks(
            collection, batch_size=batch_size, show_summary=show_summary
        )

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to the Corpora API.
        Sets self.collection after upload.

        Args:
            collection: ChunkCollection produced by any chunker.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print corpus info and upload progress (default True).
        """
        self.collection = collection

        self._corpus = self._client.get_or_create_corpus(
            self._corpus_name, on_limit="prompt"
        )

        if show_summary:
            print(f"Using corpus: {self._corpus.name} (ID: {self._corpus.id})")
            print(f"Uploading {len(self.collection)} chunks to corpus...")

        upload_result = self._client.upload_chunks(
            corpus_id=self._corpus.id,
            collection=self.collection,
            batch_size=batch_size,
            show_progress=show_summary,
        )

        if show_summary:
            print(f"\nUpload complete! Inserted: {upload_result.inserted_count}")

    def populate_from_existing_corpus(
        self,
        corpus_id: str,
        page_size: int = 500,
        show_summary: bool = True,
    ) -> None:
        """Load an existing corpus by ID into a local ChunkCollection.

        Unlike populate_from_folder(), this does not upload anything.

        Args:
            corpus_id: Existing corpus ID to load
            page_size: Number of chunks to fetch per API page (default 500)
            show_summary: Print load summary (default True)
        """
        from benchmax.rag.chunkers.inspector import ChunkInspector

        self._corpus = self._client.get_corpus(corpus_id)
        self._corpus_name = self._corpus.name

        if show_summary:
            print(
                f"Loading chunks from corpus: {self._corpus.name} (ID: {self._corpus.id})"
            )

        chunks: list[Chunk] = []
        cursor: str | None = None
        while True:
            results, next_cursor = self._client.list_corpus_chunks(
                corpus_id=corpus_id,
                limit=page_size,
                cursor=cursor,
            )
            if not results:
                break

            for row in results:
                metadata = dict(row.metadata or {})
                metadata.pop("_local_hash", None)
                chunks.append(
                    Chunk(
                        content=row.content,
                        metadata=tuple(metadata.items()),
                        hash=row.id,
                    )
                )

            if next_cursor is None:
                break
            cursor = next_cursor

        self.collection = ChunkCollection(chunks)

        if show_summary:
            inspector = ChunkInspector(self.collection)
            inspector.summary(max_depth=3, max_files_per_folder=4)
            print(f"\nLoaded {len(self.collection)} chunks from existing corpus.")

    def populate_from_existing_corpus_name(
        self,
        corpus_name: str | None = None,
        page_size: int = 500,
        show_summary: bool = True,
    ) -> None:
        """Load an existing corpus by name into a local ChunkCollection.

        Args:
            corpus_name: Existing corpus name to load. Defaults to constructor value.
            page_size: Number of chunks to fetch per API page (default 500)
            show_summary: Print load summary (default True)
        """
        target_name = corpus_name or self._corpus_name
        matched = [c for c in self._client.list_corpora() if c.name == target_name]
        if not matched:
            raise ValueError(f"Could not find existing corpus named '{target_name}'.")

        # Keep behavior deterministic if duplicate names exist.
        selected = sorted(matched, key=lambda c: c.created_at)[-1]
        self.populate_from_existing_corpus(
            corpus_id=selected.id,
            page_size=page_size,
            show_summary=show_summary,
        )

    def _assert_ready(self) -> None:
        if self.collection is None or self._corpus is None:
            raise RuntimeError("Corpus is not ready — no data has been loaded.")

    @property
    def corpus_id(self) -> str:
        """Return corpus ID. Available after loading data into this source."""
        self._assert_ready()
        return self._corpus.id

    def get_chunk_count(self) -> int:
        """Return the total number of chunks in the corpus."""
        if self.collection is None:
            return 0
        return len(self.collection)

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by minimum length."""
        self._assert_ready()
        eligible = [c for c in self.collection.chunks if len(c) >= min_chars]
        return random.sample(eligible, min(n, len(eligible)))

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return a chunk with truncated previews of its neighbors in the same file.

        Returns:
            Dict with keys: chunk_content, prev_chunk_preview, next_chunk_preview
        """
        self._assert_ready()
        return self.collection.get_chunk_with_context(
            chunk, context_max_chars=max_chars
        )

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return chunks from files at the shallowest directory depth in the corpus."""
        self._assert_ready()
        return self.collection.get_top_level_chunks()

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search for chunks related to source using BM25 queries via the Corpora API.

        Runs each query, deduplicates results by chunk hash, skips the source chunk
        and its immediate neighbors in the same file, and aggregates scores across queries.

        Returns:
            List of dicts sorted by relevance (num matching queries DESC, cross-file first,
            max BM25 score DESC), each containing: chunk, queries, same_file, max_score
        """
        if hybrid is not None:
            warnings.warn(
                "PostgresChunkSource does not support hybrid search; 'hybrid' parameter is ignored.",
                stacklevel=2,
            )
        if mode is not None and mode != "lexical":
            warnings.warn(
                f"PostgresChunkSource only supports 'lexical' mode; '{mode}' will be ignored.",
                stacklevel=2,
            )
        self._assert_ready()
        related_map: dict[str, dict] = {}

        for query in queries:
            matched_chunks = self._client.search_with_chunks(
                corpus_id=self._corpus.id,
                query=query,
                collection=self.collection,
                limit=top_k,
            )

            for result_chunk, score in matched_chunks[:top_k]:
                if result_chunk.hash == source.hash:
                    continue

                is_same_file = result_chunk.get_metadata("file") == source.get_metadata(
                    "file"
                )
                if is_same_file:
                    index_diff = abs(
                        result_chunk.get_metadata("index", 0)
                        - source.get_metadata("index", 0)
                    )
                    if index_diff <= 1:
                        continue

                if result_chunk.hash not in related_map:
                    related_map[result_chunk.hash] = {
                        "chunk": result_chunk,
                        "queries": [],
                        "same_file": is_same_file,
                        "max_score": score,
                    }
                else:
                    related_map[result_chunk.hash]["max_score"] = max(
                        related_map[result_chunk.hash]["max_score"], score
                    )

                related_map[result_chunk.hash]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (len(x["queries"]), not x["same_file"], x["max_score"]),
            reverse=True,
        )

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
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

        self._assert_ready()
        filters = to_corpora_filters(spec.get("filter"), self._search_capabilities)
        matched = self._client.search_with_chunks(
            corpus_id=self._corpus.id,
            query=str(spec.get("text_query") or ""),
            collection=self.collection,
            limit=int(spec.get("top_k", 10)),
            filters=filters,
        )
        return [chunk for chunk, _score in matched]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query and optional filter."""
        return self.search(
            SearchSpec(
                mode="lexical", text_query=text_query, top_k=top_k, filter=filter
            )
        )

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings without Chunk construction."""
        chunks = self.search(spec)
        return [chunk.content for chunk in chunks]

    def embed_query(self, text: str) -> list[float] | None:
        """Return ``None`` — Corpora backend has no embedding support."""
        return None

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for Corpora backend."""
        return self._search_capabilities
