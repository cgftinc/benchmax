"""Async search surface for the Neon corpus — a SEPARATE protocol from ChunkSource.

``asearch_related`` lives here, on :class:`AsyncChunkSource`, NOT on the
runtime-checkable sync ``ChunkSource`` protocol. Widening ``ChunkSource`` with an
async method would break every sync backend (chroma/pinecone/tpuf) and the
conformance stubs, since a ``runtime_checkable`` protocol asserts method presence
— so the async twin is a distinct protocol a caller opts into explicitly (F6/B11).

Connection model (F6/B11): the async path uses a fresh async psycopg connection
**per query op**, opened and closed inside :meth:`_AsyncQueryRunner.query_rows`,
never a long-lived cached socket. That keeps async retrieval independent of the
sync ``NeonChunkSource``'s cached connection and safe under Neon autosuspend
(a per-op connect always resolves the DSN and reconnects). The hybrid-RRF fusion
and surfaced-score formula are still the query layer's (``query.py``) — reused
here, never re-implemented — and the SQL is composed by the frozen ``NeonClient``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from castform.platform.credentials import TokenProvider
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.credentials import resolve_read_dsn_provider
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA
from castform.rag.corpus.neon.query import (
    NeonQueryRequest,
    QueryRow,
    run_query_async,
)
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG
from castform.rag.corpus.neon.source import NeonChunkSource
from castform.rag.corpus.search_schema.search_exceptions import (
    UnsupportedSearchModeError,
)

if TYPE_CHECKING:
    from castform.rag.chunkers.models import Chunk
    from castform.rag.corpus.search_schema.search_types import HybridOptions, SearchMode

# ``mode=None`` resolves to the richest available mode, best-first.
_AUTO_MODE_PREFERENCE: tuple[str, ...] = ("hybrid", "vector", "lexical")


@runtime_checkable
class AsyncChunkSource(Protocol):
    """Protocol for corpus backends that offer async related-chunk search.

    Deliberately separate from the sync ``ChunkSource`` protocol: a backend may
    implement one, both, or neither. Keeping the async method off ``ChunkSource``
    is what lets the sync protocol stay ``runtime_checkable`` without forcing an
    ``asearch_related`` onto sync-only backends.
    """

    async def asearch_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Async twin of ``ChunkSource.search_related`` (same result contract)."""
        ...


class _AsyncQueryRunner:
    """Transport for the async query path: a fresh async connection per call (F6/B11).

    Owns ONLY the per-op connection lifecycle — open the async connection (never a
    long-lived cached socket, so it is safe under Neon autosuspend), register the
    pgvector adapters, hand the open connection to the shared
    :func:`castform.rag.corpus.neon.query.run_query_async` executor, and close on
    the way out. All ranking, version resolution, prefilter, and RRF fusion live in
    the query layer (``query.py``, the single owner), never here. The ``NeonClient``
    is a pure SQL composer for that executor — its own connection is never opened.
    """

    def __init__(
        self,
        dsn_provider: TokenProvider,
        *,
        logical_name: str,
        schema: str,
        text_search_config: str,
    ) -> None:
        self._dsn_provider = dsn_provider
        self._logical_name = logical_name
        self._schema = schema
        self._text_search_config = text_search_config
        self._client = NeonClient(dsn_provider)  # composer only; never connects here

    async def query_rows(self, request: NeonQueryRequest) -> list[QueryRow]:
        """Open a per-op async connection and run the request via the shared executor."""
        import psycopg

        conn = await psycopg.AsyncConnection.connect(
            self._dsn_provider(), prepare_threshold=None
        )
        try:
            await self._register_vector(conn)
            return await run_query_async(
                conn,
                self._client,
                request,
                logical_name=self._logical_name,
                schema=self._schema,
                text_search_config=self._text_search_config,
            )
        finally:
            await conn.close()

    @staticmethod
    async def _register_vector(conn: Any) -> None:
        """Register pgvector async adapters; the ``::vector`` cast covers binding if absent."""
        try:
            from pgvector.psycopg import register_vector_async

            await register_vector_async(conn)
        except Exception:
            pass  # embedding is projected out of reads; the SQL cast binds the param


class NeonAsyncChunkSource:
    """Async related-chunk search over a Neon corpus (structural AsyncChunkSource).

    Read-only: it shares the sync source's dedup/ordering/mapping (the pure static
    helpers on :class:`NeonChunkSource`) so the async result contract is identical,
    and runs queries through :class:`_AsyncQueryRunner` (a fresh async connection
    per query op). Ingest and the sync read/search surface stay on
    ``NeonChunkSource``.

    Args:
        logical_name: Stable logical corpus name (active-version reader view).
        embed_fn: Embedding function for vector/hybrid modes; lexical-only without.
        read_dsn_provider: Read-only DSN seam. ``None`` reads ``NEON_CORPUS_DSN_RO``
            from the environment per connection.
        schema: Schema qualifying the BM25 index regclass for the RO invoker.
        text_search_config: ``regconfig`` the corpus tsvector was built with.
    """

    def __init__(
        self,
        logical_name: str,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        read_dsn_provider: str | TokenProvider | None = None,
        schema: str = CORPUS_SCHEMA,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
    ) -> None:
        self._logical_name = logical_name
        self._embed_fn = embed_fn
        self._read_dsn_provider = read_dsn_provider
        self._schema = schema
        self._text_search_config = text_search_config
        self._search: Any = None

    def _search_runner(self) -> Any:
        """Return the lazily-built per-op async query runner."""
        if self._search is None:
            self._search = _AsyncQueryRunner(
                resolve_read_dsn_provider(self._read_dsn_provider),
                logical_name=self._logical_name,
                schema=self._schema,
                text_search_config=self._text_search_config,
            )
        return self._search

    async def asearch_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Async twin of ``NeonChunkSource.search_related`` — identical result contract.

        Runs each query on its own per-op async connection (F6/B11), then applies
        the same source/neighbor skip, max-surfaced-score dedup, winning-hit
        ``native_score``, and 3-tuple ordering as the sync path.
        """
        resolved = self._resolve_mode(mode)
        runner = self._search_runner()
        related: dict[str, dict] = {}
        for query in queries:
            request = self._build_request(resolved, query, top_k, hybrid=hybrid)
            rows = await runner.query_rows(request)
            NeonChunkSource._accumulate_related(related, source, query, rows, top_k)
        return NeonChunkSource._sorted_related(related)

    def _build_request(
        self,
        mode: str,
        text: str,
        top_k: int,
        *,
        hybrid: HybridOptions | None,
    ) -> NeonQueryRequest:
        vector = None
        if mode in ("vector", "hybrid"):
            if self._embed_fn is None:
                raise ValueError("vector/hybrid search requires an embed_fn")
            vector = tuple(self._embed_fn([text])[0])
        return NeonQueryRequest(
            mode=mode,  # type: ignore[arg-type]
            top_k=top_k,
            text=text,
            vector=vector,
            hybrid=hybrid,
        )

    def _resolve_mode(self, mode: SearchMode | None) -> str:
        modes = self._modes()
        if mode is None:
            return next(m for m in _AUTO_MODE_PREFERENCE if m in modes)
        if mode not in modes:
            raise UnsupportedSearchModeError(
                backend="neon", mode=str(mode), supported_modes=modes
            )
        return mode

    def _modes(self) -> set[str]:
        modes = {"lexical"}
        if self._embed_fn is not None:
            modes |= {"vector", "hybrid"}
        return modes
