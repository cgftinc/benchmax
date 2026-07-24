"""Neon internal query layer — the single hybrid-RRF owner + surfaced score.

This module owns everything about *how a query runs and how its results are
ranked*, so no other component blends lexical and vector scores:

- the internal request shape (:class:`NeonQueryRequest`),
- the one reciprocal-rank fusion (:func:`fuse_rrf`) — the SINGLE place lexical and
  vector rankings combine for hybrid mode,
- the public surfaced-score formula (:func:`surfaced_score`) and its raw-native
  companion (:class:`QueryHit`), and
- the executor (:func:`run_query`) that resolves the current version and runs every
  leg of one query in a SINGLE read transaction under a shared advisory lock (so all
  legs read one consistent version), drives the frozen ``NeonClient`` candidate SQL,
  pushes the metadata filter into BOTH retrieval legs (B15), and enables the
  ``lakebase_bm25`` prefilter for any filtered lexical/hybrid query (F7).

``search.py`` re-exports :data:`SURFACED_RANK_K`, :class:`QueryHit`,
:func:`surfaced_score`, :class:`NeonQueryRequest`, and :func:`fuse_rrf` so their
frozen import path (``castform.rag.corpus.neon.search``) keeps working.

Surfaced-score contract (Contract #4)
-------------------------------------
The three native scorers disagree on direction: bm25 ``<@>`` is negative /
lower-better, vector cosine distance is lower-better, RRF is higher-better.
Rather than surface three incomparable scales, the public score is a single
**rank-based reciprocal rank**, uniform across all modes and always
higher-better::

    surfaced_score(rank) = 1 / (SURFACED_RANK_K + rank)      # rank is 0-based

monotonically decreasing in ``rank`` by construction, so relevance-descending is
guaranteed independent of the raw scorer's range. The raw native score is
**preserved separately** (:attr:`QueryHit.native_score`) and never overloaded onto
the surfaced score (NB1): it carries the mode's real backend number (bm25 ``<@>``,
vector distance, or — for hybrid — the fused RRF value) for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from castform.rag.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchMode,
)

if TYPE_CHECKING:
    from psycopg import sql

    from castform.rag.corpus.neon.client import NeonClient
    from castform.rag.corpus.neon.schema import NeonTableSpec

SURFACED_RANK_K = 60
"""Reciprocal-rank constant (the standard RRF ``k``). ``score = 1/(K + rank)``."""

# Hybrid oversampling: each leg fetches deeper than the caller's ``top_k`` so RRF
# has real candidate overlap to fuse before the final truncation (mirrors the
# turbopuffer client-side RRF). Bounded so a huge ``top_k`` can't ask for an
# unbounded scan.
HYBRID_OVERSAMPLE_FACTOR = 2
HYBRID_OVERSAMPLE_CAP = 10000


@dataclass(frozen=True)
class QueryHit:
    """One per-query hit carrying both the surfaced and the raw native score.

    Args:
        chunk_id: The chunk hash.
        surfaced_score: Ordinal ``1/(K + rank)`` — the public relevance number.
        native_score: The mode's raw backend score (bm25 ``<@>``, vector
            distance, or the fused RRF value for hybrid), kept for
            diagnostics/calibration (NB1).
        rank: 0-based rank in this query's native ordering.
    """

    chunk_id: str
    surfaced_score: float
    native_score: float
    rank: int


def surfaced_score(rank: int) -> float:
    """Return the public relevance score for a 0-based result rank.

    Uniform across lexical/vector/hybrid, always higher-better, strictly
    decreasing in ``rank``. This *is* the frozen formula (not backend logic), so
    tests can pin exact numeric values.
    """
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return 1.0 / (SURFACED_RANK_K + rank)


@dataclass(frozen=True)
class NeonQueryRequest:
    """Internal query request handed to the Neon query layer.

    Filtering is orthogonal to mode: any of the three modes runs filtered or
    unfiltered (3 modes x filtered/unfiltered), so ``filter`` is a field, not a
    fourth mode. Hybrid RRF fusion has a single owner — this query layer — so no
    other component blends lexical and vector scores.

    Args:
        mode: Retrieval mode.
        top_k: Maximum results requested.
        text: Query text; required for lexical/hybrid.
        vector: Query embedding; required for vector/hybrid.
        filter: Optional metadata predicate (orthogonal to mode).
        hybrid: Optional hybrid blending knobs.
    """

    mode: SearchMode
    top_k: int = 5
    text: str | None = None
    vector: tuple[float, ...] | None = None
    filter: FilterPredicate | None = None
    hybrid: HybridOptions | None = None


@dataclass(frozen=True)
class QueryRow:
    """One ranked result row — the content-bearing projection + both scores.

    ``QueryHit`` carries only ids/scores; the env-facing ``search`` needs content,
    source, and metadata too, all of which come from the candidate ``SELECT``
    (``READ_COLUMNS`` + ``native_score``). :func:`run_query` returns these.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    source_file: str
    chunk_index: int
    surfaced_score: float
    native_score: float
    rank: int

    def to_hit(self) -> QueryHit:
        """Project onto the frozen id/score ``QueryHit`` surface."""
        return QueryHit(
            self.chunk_id, self.surfaced_score, self.native_score, self.rank
        )


def fuse_rrf(
    ranked_lists: list[list[str]], k: int = SURFACED_RANK_K
) -> list[tuple[str, float]]:
    """Fuse ranked id-lists into one RRF ordering (the single fusion owner).

    Reciprocal-rank fusion with the standard ``k`` (60): a chunk's score is the
    sum over the lists it appears in of ``1/(k + rank)`` (0-based rank). Results
    are ordered by fused score descending, breaking ties on ``chunk_id``
    ascending so the ordering is fully deterministic (the sha256 hex id is a
    stable, total tiebreak — unlike raw dict order).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _hybrid_depth(top_k: int) -> int:
    """Per-leg candidate depth for hybrid fusion (bounded oversampling)."""
    return min(top_k * HYBRID_OVERSAMPLE_FACTOR, HYBRID_OVERSAMPLE_CAP)


def _row_to_result(
    row: tuple[Any, ...], *, rank: int, native_score: float
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


def _prefilter_setup(
    mode: SearchMode, filtered: bool
) -> list[sql.Composable] | None:
    """``SET LOCAL`` list enabling the BM25 prefilter for a filtered lexical/hybrid query.

    ``lakebase_bm25.prefilter = on`` must be set when a filtered BM25 leg runs
    (lexical or hybrid): a plain ``WHERE`` runs AFTER the BM25 top-k scan, so a
    strict filter would silently underfill. It only affects the BM25 index scan, so
    the vector-only path (and any unfiltered query) needs nothing — return ``None``.
    """
    if not (filtered and mode in ("lexical", "hybrid")):
        return None
    from psycopg import sql

    return [sql.SQL("SET LOCAL lakebase_bm25.prefilter = on")]


def _resolve_current_spec(
    conn: Any,
    client: NeonClient,
    logical_name: str,
    text_search_config: str,
) -> NeonTableSpec:
    """Resolve the current published version to a spec, ON the read txn's connection.

    Read inside the same locked transaction as the legs (never via the committing
    :meth:`NeonClient.execute`) so the version it returns is the one the legs then
    query — an activation cannot swap it mid-read (see
    :meth:`NeonClient.read_in_snapshot`).
    """
    from castform.rag.corpus.neon.schema import NeonTableSpec

    cur = conn.execute(client.read_ledger_sql(), {"logical": logical_name})
    for version, _state, is_current in cur.fetchall():
        if is_current:
            return NeonTableSpec(
                logical_name, version, text_search_config=text_search_config
            )
    raise LookupError(f"neon corpus {logical_name!r} has no current published version")


def _resolve_filter(
    request: NeonQueryRequest,
) -> tuple[sql.Composable | None, dict[str, Any]]:
    """Render the request's metadata predicate into a WHERE clause + bound params.

    The single filter-mapping seam shared by the sync and async executors; a
    ``None`` filter yields ``(None, {})``. ``filter_mapper`` is imported lazily —
    it pulls psycopg (the ``neon`` extra) the env-facing ``search.py`` must stay
    importable without.
    """
    if request.filter is None:
        return None, {}
    from castform.rag.corpus.neon.filter_mapper import to_neon_where

    return to_neon_where(request.filter)


def run_query(
    client: NeonClient,
    request: NeonQueryRequest,
    *,
    logical_name: str,
    schema: str,
    text_search_config: str,
) -> list[QueryRow]:
    """Execute one :class:`NeonQueryRequest` and return ranked :class:`QueryRow`s.

    Resolves the current version AND runs every leg in ONE read transaction under a
    shared advisory lock (:meth:`NeonClient.read_in_snapshot`), so all legs read a
    single consistent version even if an activation lands concurrently. Drives the
    frozen ``NeonClient`` candidate SQL for the requested mode; the metadata filter
    is rendered once and pushed into BOTH legs (B15); a filtered lexical/hybrid query
    enables the ``lakebase_bm25`` prefilter (F7). Vector binds as a ``list`` (a tuple
    would adapt as a composite row and the ``::vector`` cast would fail). Hybrid fuses
    the two legs with :func:`fuse_rrf` over an oversampled candidate set, then
    truncates to ``top_k``.
    """
    mode = request.mode
    where, filter_params = _resolve_filter(request)

    def work(conn: Any) -> list[QueryRow]:
        spec = _resolve_current_spec(conn, client, logical_name, text_search_config)
        if mode == "vector":
            rows = _vector_rows(
                conn, client, spec, request, where, filter_params, request.top_k
            )
            return _single_leg(rows)
        if mode == "lexical":
            rows = _bm25_rows(
                conn, client, spec, request, where, filter_params, request.top_k, schema
            )
            return _single_leg(rows)
        if mode == "hybrid":
            depth = _hybrid_depth(request.top_k)
            vector_rows = _vector_rows(
                conn, client, spec, request, where, filter_params, depth
            )
            bm25_rows = _bm25_rows(
                conn, client, spec, request, where, filter_params, depth, schema
            )
            return _fuse(vector_rows, bm25_rows, request.top_k)
        raise ValueError(f"unknown search mode {mode!r}")

    return client.read_in_snapshot(
        logical_name,
        work,
        session_setup=_prefilter_setup(mode, request.filter is not None),
    )


def _fetch(
    conn: Any, query: sql.Composed, params: dict[str, Any]
) -> list[tuple[Any, ...]]:
    """Run one candidate SELECT on the txn connection and return its rows."""
    cur = conn.execute(query, params)
    return cur.fetchall() if cur.description is not None else []


def _vector_rows(
    conn: Any,
    client: NeonClient,
    spec: NeonTableSpec,
    request: NeonQueryRequest,
    where: sql.Composable | None,
    filter_params: dict[str, Any],
    depth: int,
) -> list[tuple[Any, ...]]:
    if request.vector is None:
        raise ValueError("vector/hybrid search requires a query embedding")
    query, params = client.vector_candidates_sql(spec, where=where)
    merged = {**filter_params, **params, "vector": list(request.vector), "top_k": depth}
    return _fetch(conn, query, merged)


def _bm25_rows(
    conn: Any,
    client: NeonClient,
    spec: NeonTableSpec,
    request: NeonQueryRequest,
    where: sql.Composable | None,
    filter_params: dict[str, Any],
    depth: int,
    schema: str,
) -> list[tuple[Any, ...]]:
    if request.text is None:
        raise ValueError("lexical/hybrid search requires query text")
    query, params = client.bm25_candidates_sql(spec, where=where, schema=schema)
    merged = {**filter_params, **params, "text": request.text, "top_k": depth}
    return _fetch(conn, query, merged)


def _single_leg(rows: list[tuple[Any, ...]]) -> list[QueryRow]:
    """Rank a single leg's rows by their (already best-first) native order."""
    return [
        _row_to_result(row, rank=rank, native_score=row[5])
        for rank, row in enumerate(rows)
    ]


def _fuse(
    vector_rows: list[tuple[Any, ...]],
    bm25_rows: list[tuple[Any, ...]],
    top_k: int,
) -> list[QueryRow]:
    """Fuse the two legs with RRF; native_score is the fused RRF value (NB1)."""
    fused = fuse_rrf([[r[0] for r in vector_rows], [r[0] for r in bm25_rows]])
    by_id: dict[str, tuple[Any, ...]] = {r[0]: r for r in vector_rows}
    by_id.update({r[0]: r for r in bm25_rows})  # same projection either leg
    results: list[QueryRow] = []
    for rank, (chunk_id, rrf_score) in enumerate(fused[:top_k]):
        results.append(
            _row_to_result(by_id[chunk_id], rank=rank, native_score=rrf_score)
        )
    return results


# --- async executor (F6/B11) --------------------------------------------------
# The async twin of ``run_query`` for the separate async protocol. It reuses every
# ranking/version/prefilter/fusion decision above (``_resolve_filter``,
# ``_prefilter_setup``, ``_hybrid_depth``, ``_single_leg``, ``_fuse``,
# ``_row_to_result``) so lexical/vector/hybrid ordering and scoring are identical to
# the sync path — only the ``await conn.execute`` I/O differs. Version resolution,
# scoring, and RRF fusion live ONLY here, never in the async transport. The caller
# owns the connection lifecycle (open/register/close); this runs the whole query in
# one transaction under the shared advisory lock on the connection it is handed.


async def run_query_async(
    conn: Any,
    client: NeonClient,
    request: NeonQueryRequest,
    *,
    logical_name: str,
    schema: str,
    text_search_config: str,
) -> list[QueryRow]:
    """Execute one :class:`NeonQueryRequest` on an open async connection.

    Mirrors :func:`run_query`: resolves the current version AND runs every leg in
    ONE transaction under the shared advisory lock, so all legs read a single
    consistent version even under a concurrent activation; renders the metadata
    filter once into BOTH legs (B15); enables the ``lakebase_bm25`` prefilter for a
    filtered lexical/hybrid query (F7); fuses hybrid legs with :func:`fuse_rrf`. The
    caller opens/registers/closes *conn* (transport); no ranking, version, or fusion
    logic lives outside this module.
    """
    mode = request.mode
    where, filter_params = _resolve_filter(request)

    async with conn.transaction():
        await conn.execute(
            client._advisory_lock_shared_stmt(), {"logical": logical_name}
        )
        for statement in _prefilter_setup(mode, request.filter is not None) or []:
            await conn.execute(statement)
        spec = await _resolve_current_spec_async(
            conn, client, logical_name, text_search_config
        )
        if mode == "vector":
            rows = await _vector_rows_async(
                conn, client, spec, request, where, filter_params, request.top_k
            )
            return _single_leg(rows)
        if mode == "lexical":
            rows = await _bm25_rows_async(
                conn, client, spec, request, where, filter_params, request.top_k, schema
            )
            return _single_leg(rows)
        if mode == "hybrid":
            depth = _hybrid_depth(request.top_k)
            vector_rows = await _vector_rows_async(
                conn, client, spec, request, where, filter_params, depth
            )
            bm25_rows = await _bm25_rows_async(
                conn, client, spec, request, where, filter_params, depth, schema
            )
            return _fuse(vector_rows, bm25_rows, request.top_k)
        raise ValueError(f"unknown search mode {mode!r}")


async def _resolve_current_spec_async(
    conn: Any,
    client: NeonClient,
    logical_name: str,
    text_search_config: str,
) -> NeonTableSpec:
    """Async twin of :func:`_resolve_current_spec` (same current-version resolution)."""
    from castform.rag.corpus.neon.schema import NeonTableSpec

    cur = await conn.execute(client.read_ledger_sql(), {"logical": logical_name})
    for version, _state, is_current in await cur.fetchall():
        if is_current:
            return NeonTableSpec(
                logical_name, version, text_search_config=text_search_config
            )
    raise LookupError(f"neon corpus {logical_name!r} has no current published version")


async def _fetch_async(
    conn: Any, query: sql.Composed, params: dict[str, Any]
) -> list[tuple[Any, ...]]:
    """Async twin of :func:`_fetch`."""
    cur = await conn.execute(query, params)
    return await cur.fetchall() if cur.description is not None else []


async def _vector_rows_async(
    conn: Any,
    client: NeonClient,
    spec: NeonTableSpec,
    request: NeonQueryRequest,
    where: sql.Composable | None,
    filter_params: dict[str, Any],
    depth: int,
) -> list[tuple[Any, ...]]:
    """Async twin of :func:`_vector_rows`."""
    if request.vector is None:
        raise ValueError("vector/hybrid search requires a query embedding")
    query, params = client.vector_candidates_sql(spec, where=where)
    merged = {**filter_params, **params, "vector": list(request.vector), "top_k": depth}
    return await _fetch_async(conn, query, merged)


async def _bm25_rows_async(
    conn: Any,
    client: NeonClient,
    spec: NeonTableSpec,
    request: NeonQueryRequest,
    where: sql.Composable | None,
    filter_params: dict[str, Any],
    depth: int,
    schema: str,
) -> list[tuple[Any, ...]]:
    """Async twin of :func:`_bm25_rows`."""
    if request.text is None:
        raise ValueError("lexical/hybrid search requires query text")
    query, params = client.bm25_candidates_sql(spec, where=where, schema=schema)
    merged = {**filter_params, **params, "text": request.text, "top_k": depth}
    return await _fetch_async(conn, query, merged)
