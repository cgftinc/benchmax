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
    HYBRID_OVERSAMPLE_CAP,
    HYBRID_OVERSAMPLE_FACTOR,
    NeonQueryRequest,
    QueryRow,
    fuse_rrf,
    surfaced_score,
)
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG, NeonTableSpec
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
    """Runs one ``NeonQueryRequest`` on a fresh async connection per call (F6/B11).

    Mirrors ``query.run_query`` in async form: open a per-op async connection,
    take the shared per-logical advisory lock, optionally enable the BM25
    prefilter (F7), resolve the current version, run each retrieval leg, and fuse
    hybrid legs with the query layer's :func:`fuse_rrf`. The ``NeonClient`` is used
    only as a pure SQL composer here — its own connection is never opened.
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
        """Open a per-op async connection, run the request in one read snapshot."""
        import psycopg

        where, filter_params = self._resolve_where(request)
        conn = await psycopg.AsyncConnection.connect(
            self._dsn_provider(), prepare_threshold=None
        )
        try:
            await self._register_vector(conn)
            async with conn.transaction():
                await conn.execute(
                    self._client._advisory_lock_shared_stmt(),
                    {"logical": self._logical_name},
                )
                for statement in self._prefilter(request):
                    await conn.execute(statement)
                spec = await self._resolve_spec(conn)
                return await self._run_legs(conn, spec, request, where, filter_params)
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

    @staticmethod
    def _resolve_where(
        request: NeonQueryRequest,
    ) -> tuple[Any | None, dict[str, Any]]:
        if request.filter is None:
            return None, {}
        from castform.rag.corpus.neon.filter_mapper import to_neon_where

        return to_neon_where(request.filter)

    def _prefilter(self, request: NeonQueryRequest) -> list[Any]:
        """``SET LOCAL`` enabling the BM25 prefilter for a filtered lexical/hybrid query."""
        if request.filter is None or request.mode not in ("lexical", "hybrid"):
            return []
        from psycopg import sql

        return [sql.SQL("SET LOCAL lakebase_bm25.prefilter = on")]

    async def _resolve_spec(self, conn: Any) -> NeonTableSpec:
        cur = await conn.execute(
            self._client.read_ledger_sql(), {"logical": self._logical_name}
        )
        for version, _state, is_current in await cur.fetchall():
            if is_current:
                return NeonTableSpec(
                    self._logical_name,
                    version,
                    text_search_config=self._text_search_config,
                )
        raise LookupError(
            f"neon corpus {self._logical_name!r} has no current published version"
        )

    async def _run_legs(
        self,
        conn: Any,
        spec: NeonTableSpec,
        request: NeonQueryRequest,
        where: Any | None,
        filter_params: dict[str, Any],
    ) -> list[QueryRow]:
        if request.mode == "vector":
            rows = await self._vector_rows(
                conn, spec, request, where, filter_params, request.top_k
            )
            return _single_leg(rows)
        if request.mode == "lexical":
            rows = await self._bm25_rows(
                conn, spec, request, where, filter_params, request.top_k
            )
            return _single_leg(rows)
        if request.mode == "hybrid":
            depth = min(request.top_k * HYBRID_OVERSAMPLE_FACTOR, HYBRID_OVERSAMPLE_CAP)
            vector_rows = await self._vector_rows(
                conn, spec, request, where, filter_params, depth
            )
            bm25_rows = await self._bm25_rows(
                conn, spec, request, where, filter_params, depth
            )
            return _fuse(vector_rows, bm25_rows, request.top_k)
        raise ValueError(f"unknown search mode {request.mode!r}")

    async def _vector_rows(
        self,
        conn: Any,
        spec: NeonTableSpec,
        request: NeonQueryRequest,
        where: Any | None,
        filter_params: dict[str, Any],
        depth: int,
    ) -> list[tuple[Any, ...]]:
        if request.vector is None:
            raise ValueError("vector/hybrid search requires a query embedding")
        query, params = self._client.vector_candidates_sql(spec, where=where)
        merged = {
            **filter_params,
            **params,
            "vector": list(request.vector),
            "top_k": depth,
        }
        return await _fetch(conn, query, merged)

    async def _bm25_rows(
        self,
        conn: Any,
        spec: NeonTableSpec,
        request: NeonQueryRequest,
        where: Any | None,
        filter_params: dict[str, Any],
        depth: int,
    ) -> list[tuple[Any, ...]]:
        if request.text is None:
            raise ValueError("lexical/hybrid search requires query text")
        query, params = self._client.bm25_candidates_sql(
            spec, where=where, schema=self._schema
        )
        merged = {**filter_params, **params, "text": request.text, "top_k": depth}
        return await _fetch(conn, query, merged)


async def _fetch(
    conn: Any, query: Any, params: dict[str, Any]
) -> list[tuple[Any, ...]]:
    cur = await conn.execute(query, params)
    if cur.description is None:
        return []
    return await cur.fetchall()


def _single_leg(rows: list[tuple[Any, ...]]) -> list[QueryRow]:
    return [_row_to_query_row(row, rank, row[5]) for rank, row in enumerate(rows)]


def _fuse(
    vector_rows: list[tuple[Any, ...]],
    bm25_rows: list[tuple[Any, ...]],
    top_k: int,
) -> list[QueryRow]:
    fused = fuse_rrf([[r[0] for r in vector_rows], [r[0] for r in bm25_rows]])
    by_id: dict[str, tuple[Any, ...]] = {r[0]: r for r in vector_rows}
    by_id.update({r[0]: r for r in bm25_rows})
    return [
        _row_to_query_row(by_id[chunk_id], rank, rrf_score)
        for rank, (chunk_id, rrf_score) in enumerate(fused[:top_k])
    ]


def _row_to_query_row(
    row: tuple[Any, ...], rank: int, native_score: float
) -> QueryRow:
    """Map a candidate row ``(id, content, metadata, source_file, chunk_index, native)``."""
    return QueryRow(
        chunk_id=row[0],
        content=row[1],
        metadata=row[2] or {},
        source_file=row[3],
        chunk_index=row[4],
        surfaced_score=surfaced_score(rank),
        native_score=native_score,
        rank=rank,
    )


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
