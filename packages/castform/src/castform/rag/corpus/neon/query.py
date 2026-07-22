"""Neon internal query layer — the single hybrid-RRF owner + surfaced score.

This module owns everything about *how a query runs and how its results are
ranked*, so no other component blends lexical and vector scores:

- the internal request shape (:class:`NeonQueryRequest`),
- the one reciprocal-rank fusion (:func:`fuse_rrf`) — the SINGLE place lexical and
  vector rankings combine for hybrid mode,
- the public surfaced-score formula (:func:`surfaced_score`) and its raw-native
  companion (:class:`QueryHit`), and
- the executor (:func:`run_query`) that drives the frozen ``NeonClient`` candidate
  SQL, pushes the metadata filter into BOTH retrieval legs (B15), and enables the
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


def _run_bm25_leg(
    client: NeonClient,
    query: sql.Composed,
    params: dict[str, Any],
    *,
    filtered: bool,
) -> list[tuple[Any, ...]]:
    """Run the BM25 candidate SELECT, enabling the prefilter iff filtered (F7).

    ``lakebase_bm25.prefilter = on`` must be set for a filtered lexical/hybrid
    query: a plain ``WHERE`` runs AFTER the BM25 top-k scan, so a strict filter
    would silently underfill the result set. The GUC is issued ``SET LOCAL`` in
    the SAME transaction as the SELECT, so it is transaction-scoped and cannot
    leak to a reused/pooled connection. The vector leg never needs it (its filter
    is served by the ``meta_gin`` index).
    """
    if not filtered:
        return client.execute(query, params)

    from psycopg import sql

    return client.execute_read_txn(
        query,
        params,
        session_setup=[sql.SQL("SET LOCAL lakebase_bm25.prefilter = on")],
    )


def run_query(
    client: NeonClient,
    spec: NeonTableSpec,
    request: NeonQueryRequest,
    *,
    schema: str,
) -> list[QueryRow]:
    """Execute one :class:`NeonQueryRequest` and return ranked :class:`QueryRow`s.

    Drives the frozen ``NeonClient`` candidate SQL for the requested mode. The
    metadata filter is rendered once and pushed into BOTH legs (B15). Vector
    binds as a ``list`` (a tuple would adapt as a composite row and the
    ``::vector`` cast would fail). Hybrid fuses the two legs with :func:`fuse_rrf`
    over an oversampled candidate set, then truncates to ``top_k``.
    """
    mode = request.mode
    if request.filter is None:
        where: sql.Composable | None = None
        filter_params: dict[str, Any] = {}
    else:
        # Imported lazily: filter_mapper pulls psycopg (the `neon` extra), which the
        # env-facing search.py must stay importable without.
        from castform.rag.corpus.neon.filter_mapper import to_neon_where

        where, filter_params = to_neon_where(request.filter)

    if mode == "vector":
        return _single_leg(
            _vector_rows(client, spec, request, where, filter_params, request.top_k)
        )
    if mode == "lexical":
        return _single_leg(
            _bm25_rows(
                client, spec, request, where, filter_params, request.top_k, schema
            )
        )
    if mode == "hybrid":
        depth = _hybrid_depth(request.top_k)
        vector_rows = _vector_rows(
            client, spec, request, where, filter_params, depth
        )
        bm25_rows = _bm25_rows(
            client, spec, request, where, filter_params, depth, schema
        )
        return _fuse(vector_rows, bm25_rows, request.top_k)
    raise ValueError(f"unknown search mode {mode!r}")


def _vector_rows(
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
    return client.execute(query, merged)


def _bm25_rows(
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
    return _run_bm25_leg(client, query, merged, filtered=request.filter is not None)


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
