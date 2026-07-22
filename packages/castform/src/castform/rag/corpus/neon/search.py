"""NeonSearch — pickle-safe search client + the surfaced-score contract.

Contract-freeze artifact (Slice A). The query execution, RRF fusion, and dedup
are built in Slices 1/4 — the methods here are stubs. What *is* frozen: the
internal query-request shape, the single-owner hybrid-fusion seam, and the exact
public score formula.

Surfaced-score contract (Contract #4)
-------------------------------------
The three native scorers disagree on direction: bm25 ``<@>`` is negative /
lower-better, vector cosine distance is lower-better, RRF is higher-better.
Rather than surface three incomparable scales, the public score is a single
**rank-based reciprocal rank**, uniform across all modes and always
higher-better::

    surfaced_score(rank) = 1 / (SURFACED_RANK_K + rank)      # rank is 0-based

where ``rank`` is the 0-based position of a chunk in a single query's result
list ordered better-first in that mode's native scorer (bm25 ascending ``<@>``,
vector ascending distance, hybrid the fused ordering). This is monotonically
decreasing in rank by construction, so relevance-descending is guaranteed
independent of the raw scorer's range.

The raw native score is **preserved as a separate field** (``QueryHit.native_score``
/ the ``native_score`` result key), never overloaded onto ``max_score`` (NB1): it
carries the mode's real backend number (bm25 ``<@>``, vector distance, or fused
RRF) for diagnostics and calibration. Empirical validation of the raw ``<@>``
range is deferred to the Slice 3 live smoke; the surfaced formula, its
monotonicity, and the dedup rule are frozen here.

Multi-query dedup mirrors the Corpora path (``postgres/source.py``): a chunk hit
by several queries keeps the **max** reciprocal rank across those queries as its
``max_score``, and results sort by the 3-tuple
``(len(queries), not same_file, max_score)`` all descending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from castform.platform.credentials import TokenProvider
from castform.rag.corpus.neon.credentials import resolve_read_dsn_provider
from castform.rag.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchMode,
)

SURFACED_RANK_K = 60
"""Reciprocal-rank constant (the standard RRF ``k``). ``score = 1/(K + rank)``."""


@dataclass(frozen=True)
class QueryHit:
    """One per-query hit carrying both the surfaced and the raw native score.

    Args:
        chunk_id: The chunk hash.
        surfaced_score: Ordinal ``1/(K + rank)`` — the public relevance number.
        native_score: The mode's raw backend score (bm25 ``<@>``, vector
            distance, or fused RRF), kept for diagnostics/calibration (NB1).
        rank: 0-based rank in this query's native ordering.
    """

    chunk_id: str
    surfaced_score: float
    native_score: float
    rank: int


def surfaced_score(rank: int) -> float:
    """Return the public relevance score for a 0-based result rank.

    Uniform across lexical/vector/hybrid, always higher-better, strictly
    decreasing in ``rank``. This *is* implemented (it is the frozen formula, not
    backend logic) so tests can pin exact numeric values.
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


def fuse_rrf(
    ranked_lists: list[list[str]], k: int = SURFACED_RANK_K
) -> list[tuple[str, float]]:
    """Fuse multiple ranked id-lists into one RRF ordering (single owner).

    The one place lexical and vector rankings are combined for hybrid mode.
    Design-lock stub: fusion is built in Slice 1.
    """
    raise NotImplementedError("RRF fusion is built in Slice 1")


class NeonSearch:
    """Pickle-safe Neon corpus search client for RL environments.

    Mirrors ``TpufSearch``: no psycopg import at module load, connection resolved
    per call via a read-only DSN provider, ``_conn`` nulled across pickling. The
    DSN rides the ``str | TokenProvider | None`` seam and resolves to a
    *read-only* grant (search never writes).

    Args:
        table: Logical corpus name to query (the active-version view).
        embed_fn: Embedding function for vector/hybrid modes. Same interface as
            every provider: ``Callable[[list[str]], list[list[float]]]``.
        dsn_provider: Read-only DSN, a provider callable, or ``None`` to read
            ``NEON_CORPUS_DSN_RO`` from the environment at query time.
    """

    def __init__(
        self,
        table: str,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        dsn_provider: str | TokenProvider | None = None,
    ) -> None:
        self._table = table
        self._embed_fn = embed_fn
        self._dsn_provider = resolve_read_dsn_provider  # bound in Slice 4
        self._dsn_provider_arg = dsn_provider
        self._conn: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._conn = None

    def query(self, request: NeonQueryRequest) -> list[QueryHit]:
        """Run one query, returning ``QueryHit`` rows best-first.

        Each hit carries both the surfaced ordinal score and the raw native
        score (NB1). Design-lock stub: SQL execution is built in Slice 1.
        """
        raise NotImplementedError("Neon query execution is built in Slice 1")

    def search_content(self, request: NeonQueryRequest) -> list[str]:
        """Return content strings only (cloudpickle-safe rollout path).

        Design-lock stub: built in Slice 1.
        """
        raise NotImplementedError("Neon query execution is built in Slice 1")
