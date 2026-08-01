"""NeonChunkSource — ChunkSource implementation over a Neon corpus.

Contract-freeze artifact (Slice A). All methods are stubs; ingest and search
land in Slices 1/2/4. This module freezes two surface commitments:

- ``search_related`` obeys the shared ``ChunkSource`` contract (relevance
  descending, a ``max_score`` per result) using the surfaced reciprocal-rank
  score from ``search.py`` (Contract #4).
- ``scan_chunks`` is a *stable*, pageable full-corpus iterator ordered by the
  typed non-null columns ``(source_file, chunk_index, id)`` so qa-gen full-corpus
  materialization is deterministic (Contract #5). It is a Neon extension to the
  surface — promoting it onto the shared ``ChunkSource`` protocol is deferred to
  a later slice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from castform.platform.credentials import TokenProvider
from castform.rag.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchCapabilities,
    SearchMode,
    SearchSpec,
)

if TYPE_CHECKING:
    from castform.rag.chunkers.models import Chunk, ChunkCollection

# Total, pageable scan order (B6). These are TYPED, NOT NULL physical columns —
# ``source_file text`` and ``chunk_index integer`` (populated at ingest from
# metadata) plus ``id`` (the chunk hash) as the final tiebreak — backed by the
# ``scan`` btree. Using real columns (not JSONB extraction) avoids lexical
# ``chunk_index`` sorting (1,10,2), NULL-break keysets, and collation ambiguity;
# the keyset cursor is the row-tuple predicate ``(source_file, chunk_index, id) >
# (%s, %s, %s)``.
SCAN_ORDER_BY = ("source_file", "chunk_index", "id")


class NeonChunkSource:
    """Corpus backend backed by a versioned Neon (Postgres + pgvector) table.

    Args:
        logical_name: Stable logical corpus name (the active-version view).
        embed_fn: Embedding function for vector/hybrid modes.
        read_dsn_provider: Read-only DSN seam for search.
        write_dsn_provider: Read-write DSN seam for ingest. Separate from the
            read provider so a search-only handle cannot mutate the corpus.
    """

    def __init__(
        self,
        logical_name: str,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        read_dsn_provider: str | TokenProvider | None = None,
        write_dsn_provider: str | TokenProvider | None = None,
    ) -> None:
        self._logical_name = logical_name
        self._embed_fn = embed_fn
        self._read_dsn_provider = read_dsn_provider
        self._write_dsn_provider = write_dsn_provider

    # --- ingest --------------------------------------------------------------

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
        """Chunk a folder and ingest as a new corpus version. Built in Slice 2."""
        raise NotImplementedError("Neon ingest is built in Slice 2")

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Ingest a pre-built collection as a new corpus version. Built in Slice 2."""
        raise NotImplementedError("Neon ingest is built in Slice 2")

    # --- read / sample -------------------------------------------------------

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks. Built in Slice 1."""
        raise NotImplementedError("Neon read is built in Slice 1")

    def scan_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        """Yield every chunk in a stable order for deterministic materialization.

        Ordered by ``(file, index, id)`` (see ``SCAN_ORDER_BY``) via keyset
        pagination so qa-gen full-corpus reads are reproducible run to run.
        Design-lock stub: built in Slice 1.
        """
        raise NotImplementedError("scan_chunks is built in Slice 1")

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return a chunk with neighboring-context previews. Built in Slice 1."""
        raise NotImplementedError("Neon read is built in Slice 1")

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return top-level-file chunks (empty if unsupported). Built in Slice 1."""
        raise NotImplementedError("Neon read is built in Slice 1")

    # --- search --------------------------------------------------------------

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search related chunks; relevance-descending with a ``max_score``.

        Returns dicts keyed ``{chunk, queries, same_file, max_score, native_score}``
        sorted by ``(len(queries), not same_file, max_score)`` descending, using
        the surfaced reciprocal-rank score for ``max_score``; ``native_score``
        carries the raw backend score from the SAME winning hit that supplied
        ``max_score`` (NB1). Built in Slice 1.
        """
        raise NotImplementedError("Neon search is built in Slice 1")

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Structured search. Built in Slice 1."""
        raise NotImplementedError("Neon search is built in Slice 1")

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Cloudpickle-safe content-only search. Built in Slice 1."""
        raise NotImplementedError("Neon search is built in Slice 1")

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Text search with optional filter. Built in Slice 1."""
        raise NotImplementedError("Neon search is built in Slice 1")

    def embed_query(self, text: str) -> list[float] | None:
        """Embed a query, or ``None`` if no embed_fn. Built in Slice 1."""
        raise NotImplementedError("Neon search is built in Slice 1")

    # --- capabilities --------------------------------------------------------

    def get_chunk_count(self) -> int:
        """Total chunk count for the active version. Built in Slice 1."""
        raise NotImplementedError("Neon read is built in Slice 1")

    def get_search_capabilities(self) -> SearchCapabilities:
        """Report Neon backend capabilities. Built in Slice 1."""
        raise NotImplementedError("Neon capabilities are built in Slice 1")
