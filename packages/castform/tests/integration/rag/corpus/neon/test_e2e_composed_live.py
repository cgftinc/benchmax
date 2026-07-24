"""@integration composed end-to-end test for the Neon lakebase corpus provider (Slice 8).

The MANDATORY full-stack integration gate before roll-up. Where Slice 7's
``test_golden_eval_live`` measures golden retrieval *quality*, this module proves the
composed PUBLIC provider surface *plumbs together* end to end against the live
``gitlab_handbook_neon`` corpus (31665 chunks, v1 activated) — nothing here rewards
recall, everything here proves a request travels from the public API through the
query layer to real Neon and back with the right order, shape, and scores.

Every capability is checked against an INDEPENDENT oracle so no assertion is
vacuous: a capability test compares the composed ordering to a raw candidate-SQL /
RRF-fusion ordering issued directly through ``NeonClient``, and the score tests pin
the EXACT surfaced formula (``1/(60+rank)``) plus a finite, mode-appropriate native
score — a single unordered row can never pass.

It exercises three distinct public surfaces:

* :class:`~castform.rag.corpus.neon.source.NeonChunkSource` — the driver-side
  ChunkSource. All FOUR capabilities run through it: lexical (BM25), vector (ANN),
  hybrid (RRF fusion), and FILTERED.
* the FILTER predicate at the ChunkSource layer (B7). The env
  :class:`~castform.rag.corpus.neon.search.NeonSearch` ``search()`` cannot carry a
  filter (proven by signature introspection AND a rejected call), so filtered
  retrieval MUST go through ``NeonChunkSource.search(SearchSpec(filter=...))``. We
  prove the predicate travels: a cross-section decoy present unfiltered vanishes once
  the ``handbook_section`` predicate is applied, and an ``and``-ed ``path_depth``
  clause narrows the set within a section.
* :class:`~castform.rag.corpus.neon.search.NeonSearch` — the pickle-safe, env-facing
  client. It cloudpickle-roundtrips (the reference path keeps the DSN OUT of the
  artifact) and the revived client still searches live.

Neon scale-to-zero: the first query after idle cold-starts a suspended compute. We
rely on the hardened ``NeonClient`` reconnect (connect_timeout + keepalives, B16) that
converts a dead cached socket into a retryable ``OperationalError`` instead of a hang,
wrapped in a bounded, per-call-deadlined retry — never one long blind sleep.

Requires ``NEON_CORPUS_DSN_RO`` + ``PLATFORM_API_KEY`` (for embeddings) and the
``neon`` extra; the module skips when the DSN is absent. Run explicitly::

    uv run --extra neon python -m pytest -m integration \\
        tests/integration/rag/corpus/neon/test_e2e_composed_live.py
"""

from __future__ import annotations

import contextlib
import inspect
import math
import os
import re
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg")
import cloudpickle  # noqa: E402
from psycopg import sql  # noqa: E402

from castform.rag.chunkers.models import Chunk  # noqa: E402
from castform.rag.corpus.embed import platform_embed_fn  # noqa: E402
from castform.rag.corpus.neon.client import NeonClient  # noqa: E402
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA  # noqa: E402
from castform.rag.corpus.neon.query import (  # noqa: E402
    HYBRID_OVERSAMPLE_CAP,
    HYBRID_OVERSAMPLE_FACTOR,
    fuse_rrf,
    surfaced_score,
)
from castform.rag.corpus.neon.schema import (  # noqa: E402
    DEFAULT_TEXT_SEARCH_CONFIG,
    NeonTableSpec,
)
from castform.rag.corpus.neon.search import NeonSearch  # noqa: E402
from castform.rag.corpus.neon.source import NeonChunkSource  # noqa: E402
from castform.rag.corpus.search_schema.builders import all_of, field  # noqa: E402
from castform.rag.corpus.search_schema.search_types import SearchSpec  # noqa: E402

pytestmark = pytest.mark.integration

_ENV_FILE = Path.home() / ".config" / "neon-benchmax.env"

LOGICAL = "gitlab_handbook_neon"
LLM_URL = "https://llm.castform.dev/v1"
TOP_K = 5
EXPECTED_CHUNK_COUNT = 31665
# a ubiquitous handbook term so every score leg returns a full, ranked result set.
PROBE = "communication"
# candidate probes for the filter scenario; the first that yields both a
# cross-section decoy AND same-section depth variety wins.
FILTER_PROBES = ("team", "process", "communication", "policy", "manager", "engineering")
# per-live-call wall-clock ceiling so a genuine hang fails the retry instead of
# blocking forever (the reconnect deadline the cold-start helper must honor).
_CALL_DEADLINE_S = 60.0


def _load_env_file() -> None:
    """Best-effort load of the developer-local env file (RO DSN + platform key)."""
    if os.environ.get("NEON_CORPUS_DSN_RO") and os.environ.get("PLATFORM_API_KEY"):
        return
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        m = re.match(r'^([A-Z_]+)="?([^"]*)"?$', line.strip())
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2)


_load_env_file()
RO_DSN = os.environ.get("NEON_CORPUS_DSN_RO")

if not RO_DSN:
    pytest.skip("NEON_CORPUS_DSN_RO not set", allow_module_level=True)


@contextlib.contextmanager
def _deadline(seconds: float) -> Iterator[None]:
    """Raise ``TimeoutError`` if the wrapped block runs past *seconds* (SIGALRM).

    A hard wall-clock ceiling so a genuinely hung socket read FAILS a test rather
    than blocking indefinitely. Main-thread only (pytest runs sync tests there).
    """

    def _raise(signum: int, frame: Any) -> None:
        raise TimeoutError(f"operation exceeded {seconds}s (possible hang)")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _cold_start_retry(fn: Any, *, attempts: int = 6, backoff: float = 4.0) -> Any:
    """Run *fn* under a per-call deadline, absorbing a Neon scale-to-zero cold start.

    The hardened ``NeonClient`` (B16 connect_timeout + keepalives) surfaces a
    suspended-compute wake or a dead cached socket as a retryable
    ``OperationalError``/``InterfaceError`` rather than hanging on a half-open
    connection. Each attempt is bounded by :data:`_CALL_DEADLINE_S` (so a true hang
    is caught, not waited on) and retried a bounded number of times with a short
    backoff — never one long blind sleep. Non-retryable errors propagate immediately.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            with _deadline(_CALL_DEADLINE_S):
                return fn()
        except (psycopg.OperationalError, psycopg.InterfaceError, TimeoutError) as exc:
            last = exc
            time.sleep(backoff)
    raise AssertionError(f"live neon unreachable after {attempts} attempts: {last}")


def _section(chunk: Chunk) -> str | None:
    return chunk.get_metadata("handbook_section")


def _depth(chunk: Chunk) -> int | None:
    return chunk.get_metadata("path_depth")


def _bm25_oracle(
    ro: NeonClient, spec: NeonTableSpec, text: str, top_k: int
) -> list[tuple[Any, ...]]:
    """Raw BM25 candidate rows via ``NeonClient`` — the independent lexical oracle.

    Rows are ``(id, content, metadata, source_file, chunk_index, native)`` best-first;
    ``native`` is the ``<@>`` score (negative, ascending = best-first).
    """
    query, params = ro.bm25_candidates_sql(spec, schema=CORPUS_SCHEMA)
    return ro.execute(query, {**params, "text": text, "top_k": top_k})


def _vector_oracle(
    ro: NeonClient, spec: NeonTableSpec, vector: list[float], top_k: int
) -> list[tuple[Any, ...]]:
    """Raw ANN candidate rows via ``NeonClient`` — the independent vector oracle.

    ``native`` (``row[5]``) is the cosine distance (>= 0, ascending = best-first).
    """
    query, params = ro.vector_candidates_sql(spec)
    return ro.execute(query, {**params, "vector": list(vector), "top_k": top_k})


# lakebase_bm25 / lakebase_ann serve the ORDER BY from an APPROXIMATE index scan
# while the SELECT re-projects the identical <@> / <=> score into native_score; on
# near-equal candidates the two can invert by a hair (measured live: max 0.013 /
# ~0.28% on one adjacent BM25 tail pair; vector 0.0). So the RE-PROJECTED native
# score is best-first only within this small tolerance — the EXACT ranking teeth is
# the composed==oracle assertion, not native monotonicity. These bound the
# approximation without going vacuous: a real reorder blows past both.
_NATIVE_APPROX_EPS = 0.05  # >> observed 0.013, << the score magnitude / a real inversion
# floor admits up to two adjacent tail near-ties (one swap = rho 0.9, two = 0.8); the
# EPS adjacency bound is what keeps those swaps to genuine near-ties, while a real
# reorder (a single 2-apart displacement already = rho 0.6) still fails. observed 0.9.
_RANK_CORR_MIN = 0.8


def _assert_native_sane(mode: str, natives: list[float]) -> None:
    """Assert every native score is finite and in the mode's documented direction."""
    assert all(math.isfinite(n) for n in natives), f"{mode} native not finite: {natives}"
    if mode == "lexical":
        assert all(n <= 0 for n in natives), f"bm25 native must be <= 0: {natives}"
    elif mode == "vector":
        assert all(n >= 0 for n in natives), f"cosine distance must be >= 0: {natives}"
    else:  # hybrid: native is the fused RRF value, strictly positive
        assert all(n > 0 for n in natives), f"rrf native must be > 0: {natives}"


def _spearman(natives: list[float]) -> float:
    """Spearman rho between the returned position and the native-sorted rank.

    For ``n`` = TOP_K = 5 one adjacent transposition gives 0.9, two give 0.8, and any
    genuine reorder (a single 2-apart displacement already) gives <= 0.6, so a floor
    between separates approximate-index tail near-ties from a real shuffle.
    """
    n = len(natives)
    order = sorted(range(n), key=lambda i: natives[i])
    native_rank = {i: r for r, i in enumerate(order)}
    d2 = sum((pos - native_rank[pos]) ** 2 for pos in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _assert_best_first_approx(natives: list[float]) -> None:
    """Assert best-first up to the index's approximation (not strict monotonicity).

    Three teeth that a real reorder fails but a sub-1% tail near-tie survives: the top
    row carries the global minimum (best) native score, the position/native rank
    correlation clears :data:`_RANK_CORR_MIN`, and no adjacent pair inverts by more
    than :data:`_NATIVE_APPROX_EPS`.
    """
    assert natives[0] == min(natives), f"top row is not the best native score: {natives}"
    rho = _spearman(natives)
    assert rho >= _RANK_CORR_MIN, f"rank correlation {rho:.3f} < {_RANK_CORR_MIN}: {natives}"
    for earlier, later in zip(natives, natives[1:]):
        assert earlier <= later + _NATIVE_APPROX_EPS, f"native inversion too large: {natives}"


def _oracle_ids_for_mode(
    mode: str, ro: NeonClient, spec: NeonTableSpec, query_vector: list[float]
) -> list[str]:
    """Return the independent same-mode candidate-SQL / RRF ranking of ids (top TOP_K).

    The identity oracle shared by the capability and ``search_related`` tests: raw
    ``NeonClient`` candidate SQL for lexical/vector, and an out-of-band RRF fusion of
    both legs (at the query layer's oversample depth) for hybrid.
    """
    if mode == "lexical":
        return [r[0] for r in _bm25_oracle(ro, spec, PROBE, TOP_K)]
    if mode == "vector":
        return [r[0] for r in _vector_oracle(ro, spec, query_vector, TOP_K)]
    depth = min(TOP_K * HYBRID_OVERSAMPLE_FACTOR, HYBRID_OVERSAMPLE_CAP)
    vrows = _vector_oracle(ro, spec, query_vector, depth)
    brows = _bm25_oracle(ro, spec, PROBE, depth)
    fused = fuse_rrf([[r[0] for r in vrows], [r[0] for r in brows]])[:TOP_K]
    return [cid for cid, _ in fused]


@pytest.fixture(scope="module")
def embed_fn() -> Any:
    """Platform embedder for vector/hybrid modes (resolves PLATFORM_API_KEY at call)."""
    return platform_embed_fn(base_url=LLM_URL)


@pytest.fixture(scope="module")
def source(embed_fn: Any) -> NeonChunkSource:
    """Read-only NeonChunkSource over the live corpus, warmed past cold start.

    Warming with a single ``get_chunk_count`` both wakes a suspended compute (via the
    bounded reconnect) and pins that we are querying the expected 31665-chunk version.
    """
    src = NeonChunkSource(LOGICAL, embed_fn=embed_fn, read_dsn_provider=RO_DSN)
    count = _cold_start_retry(src.get_chunk_count)
    assert count == EXPECTED_CHUNK_COUNT, f"unexpected corpus size {count}"
    return src


@pytest.fixture(scope="module")
def ro_client() -> NeonClient:
    """Raw read-only NeonClient for issuing the independent candidate-SQL oracles."""
    client = NeonClient(lambda: RO_DSN)
    _cold_start_retry(lambda: client.execute(sql.SQL("SELECT 1")))
    return client


@pytest.fixture(scope="module")
def spec(ro_client: NeonClient) -> NeonTableSpec:
    """Resolve the corpus's current published version to a table spec (for the oracles)."""
    rows = _cold_start_retry(
        lambda: ro_client.execute(ro_client.read_ledger_sql(), {"logical": LOGICAL})
    )
    for version, _state, is_current in rows:
        if is_current:
            return NeonTableSpec(
                LOGICAL, version, text_search_config=DEFAULT_TEXT_SEARCH_CONFIG
            )
    raise AssertionError(f"no current published version for {LOGICAL!r}")


@pytest.fixture(scope="module")
def query_vector(source: NeonChunkSource) -> list[float]:
    """One live embedding of :data:`PROBE` reused by the vector/hybrid capabilities."""
    vector = _cold_start_retry(lambda: source.embed_query(PROBE))
    assert vector is not None and len(vector) > 0
    return list(vector)


# --- (1) four capabilities through NeonChunkSource, each vs an independent oracle ---


def test_lexical_capability_matches_bm25_oracle(
    source: NeonChunkSource, ro_client: NeonClient, spec: NeonTableSpec
) -> None:
    """Lexical (BM25) search returns TOP_K unique chunks in the exact BM25 candidate order.

    The composed path (``NeonChunkSource.search`` -> query layer) must not reorder the
    backend's ranking: its ids equal a raw ``bm25_candidates_sql`` ordering issued
    directly through ``NeonClient`` (the exact teeth). The re-projected ``<@>`` scores
    are finite, non-positive, and best-first up to the ``lakebase_bm25`` approximate
    scan (:func:`_assert_best_first_approx`) — not strictly monotone at near-ties.
    """
    chunks = _cold_start_retry(
        lambda: source.search(
            SearchSpec(mode="lexical", text_query=PROBE, top_k=TOP_K)
        )
    )
    hashes = [c.hash for c in chunks]
    assert len(hashes) == TOP_K, f"expected {TOP_K} rows, got {len(hashes)}"
    assert len(set(hashes)) == TOP_K, f"duplicate chunk hashes: {hashes}"
    assert all(c.content for c in chunks)

    rows = _cold_start_retry(lambda: _bm25_oracle(ro_client, spec, PROBE, TOP_K))
    assert hashes == [r[0] for r in rows], "composed lexical order != bm25 candidate SQL"
    natives = [r[5] for r in rows]
    _assert_native_sane("lexical", natives)
    _assert_best_first_approx(natives)


def test_vector_capability_matches_ann_oracle(
    source: NeonChunkSource,
    ro_client: NeonClient,
    spec: NeonTableSpec,
    query_vector: list[float],
) -> None:
    """Vector (ANN) search returns TOP_K unique chunks in the exact ANN candidate order.

    Ids equal a raw ``vector_candidates_sql`` ordering on the SAME query embedding (the
    exact teeth); the re-projected cosine distances are finite, non-negative, and
    nearest-first up to the ``lakebase_ann`` approximation (:func:`_assert_best_first_approx`).
    """
    chunks = _cold_start_retry(
        lambda: source.search(
            SearchSpec(mode="vector", vector_query=query_vector, top_k=TOP_K)
        )
    )
    hashes = [c.hash for c in chunks]
    assert len(hashes) == TOP_K, f"expected {TOP_K} rows, got {len(hashes)}"
    assert len(set(hashes)) == TOP_K, f"duplicate chunk hashes: {hashes}"
    assert all(c.content for c in chunks)

    rows = _cold_start_retry(lambda: _vector_oracle(ro_client, spec, query_vector, TOP_K))
    assert hashes == [r[0] for r in rows], "composed vector order != ann candidate SQL"
    natives = [r[5] for r in rows]
    _assert_native_sane("vector", natives)
    _assert_best_first_approx(natives)


def test_hybrid_capability_matches_rrf_fusion_oracle(
    source: NeonChunkSource,
    ro_client: NeonClient,
    spec: NeonTableSpec,
    query_vector: list[float],
) -> None:
    """Hybrid search returns TOP_K unique chunks in the exact independent RRF order.

    Reproduces the query layer's fusion outside it: fetch each leg at the same
    oversampled depth, fuse with :func:`fuse_rrf`, truncate to TOP_K, and require the
    composed ids to match. The fused RRF values are finite, positive, and descending.
    """
    chunks = _cold_start_retry(
        lambda: source.search(
            SearchSpec(
                mode="hybrid",
                text_query=PROBE,
                vector_query=query_vector,
                top_k=TOP_K,
            )
        )
    )
    hashes = [c.hash for c in chunks]
    assert len(hashes) == TOP_K, f"expected {TOP_K} rows, got {len(hashes)}"
    assert len(set(hashes)) == TOP_K, f"duplicate chunk hashes: {hashes}"
    assert all(c.content for c in chunks)

    depth = min(TOP_K * HYBRID_OVERSAMPLE_FACTOR, HYBRID_OVERSAMPLE_CAP)
    vrows = _cold_start_retry(lambda: _vector_oracle(ro_client, spec, query_vector, depth))
    brows = _cold_start_retry(lambda: _bm25_oracle(ro_client, spec, PROBE, depth))
    fused = fuse_rrf([[r[0] for r in vrows], [r[0] for r in brows]])[:TOP_K]
    assert hashes == [cid for cid, _ in fused], "composed hybrid order != independent rrf"
    rrf_scores = [s for _, s in fused]
    _assert_native_sane("hybrid", rrf_scores)
    assert rrf_scores == sorted(rrf_scores, reverse=True), f"rrf not descending: {rrf_scores}"


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_search_related_surfaces_exact_reciprocal_rank(
    source: NeonChunkSource,
    ro_client: NeonClient,
    spec: NeonTableSpec,
    query_vector: list[float],
    mode: str,
) -> None:
    """``search_related`` returns the same-mode ranking with the exact reciprocal-rank score.

    A synthetic source chunk (absent from the corpus, no ``file`` metadata) keeps
    ``same_file`` uniformly false, so the result set is the full TOP_K ranked hits.
    Teeth (would fail on duplicates, a shuffle, or wrong ids): TOP_K UNIQUE hashes; the
    ids equal the independent same-mode oracle ranking; the surfaced scores are exactly
    ``[1/(60+i)]``; and the native scores are best-first — approximate for lexical/vector
    (:func:`_assert_best_first_approx`), strictly descending for hybrid (RRF fuses
    integer ranks, so it is exact).
    """
    synthetic = Chunk(content="", metadata=(), hash="__slice8_synthetic_source__")
    results = _cold_start_retry(
        lambda: source.search_related(synthetic, [PROBE], top_k=TOP_K, mode=mode)
    )
    assert len(results) == TOP_K, f"{mode}: expected {TOP_K} related, got {len(results)}"
    hashes = [r["chunk"].hash for r in results]
    assert len(set(hashes)) == TOP_K, f"{mode}: duplicate related hashes: {hashes}"

    oracle_ids = _cold_start_retry(
        lambda: _oracle_ids_for_mode(mode, ro_client, spec, query_vector)
    )
    assert hashes == oracle_ids, f"{mode}: search_related order != oracle {hashes} vs {oracle_ids}"

    scores = [r["max_score"] for r in results]
    assert scores == [surfaced_score(i) for i in range(TOP_K)], scores

    natives = [r["native_score"] for r in results]
    _assert_native_sane(mode, natives)
    if mode == "hybrid":
        assert natives == sorted(natives, reverse=True), f"rrf native not descending: {natives}"
    else:
        _assert_best_first_approx(natives)


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_env_client_search_surfaced_score_exact_contract(
    embed_fn: Any, mode: str
) -> None:
    """The env ``NeonSearch.search`` honors the documented surfaced-score contract.

    Returns TOP_K unique results whose ``score`` is exactly the frozen ordinal
    ``1/(60+i)`` (so every score is <= 1/60, strictly decreasing) — the full contract,
    not a weakened ``score <= 1`` bound.
    """
    client = NeonSearch(LOGICAL, embed_fn=embed_fn, dsn_provider=RO_DSN)
    results = _cold_start_retry(lambda: client.search(PROBE, mode=mode, top_k=TOP_K))
    assert len(results) == TOP_K, f"{mode}: expected {TOP_K} results, got {len(results)}"
    scores = [r["score"] for r in results]
    assert scores == [surfaced_score(i) for i in range(TOP_K)], scores
    assert scores[0] == pytest.approx(1.0 / 60.0)
    unique = {(r["source"], r["content"]) for r in results}
    assert len(unique) == TOP_K, f"{mode}: results not unique: {unique}"
    assert all(r["content"] for r in results)


# --- (2) the filter predicate travels at the ChunkSource layer (B7) -----------


@pytest.fixture(scope="module")
def filter_scenario(source: NeonChunkSource) -> dict[str, Any]:
    """Derive a live filter scenario from the corpus (no golden-file dependency).

    Picks the first probe that yields BOTH a cross-section decoy (its unfiltered
    lexical top-k spans >=2 sections) and a section among those hits with >=2 distinct
    ``path_depth`` values (so the depth clause has something to narrow). Fails loudly
    if no probe qualifies — that is a real corpus/plumbing regression, never a skip.
    """
    for probe in FILTER_PROBES:
        hits = _cold_start_retry(
            lambda p=probe: source.search(
                SearchSpec(mode="lexical", text_query=p, top_k=15)
            )
        )
        sections = [_section(c) for c in hits]
        gold_section = sections[0] if hits else None
        distinct = {s for s in sections if s}
        if gold_section is None or len(distinct) < 2:
            continue
        cross_decoys = {c.hash for c in hits if _section(c) not in (None, gold_section)}
        if not cross_decoys:
            continue
        depth_section: str | None = None
        depth_hits: list[Chunk] | None = None
        for candidate in dict.fromkeys([gold_section, *distinct]):
            in_section = _cold_start_retry(
                lambda s=candidate: source.search(
                    SearchSpec(
                        mode="lexical",
                        text_query=probe,
                        top_k=25,
                        filter=field("handbook_section").eq(s),
                    )
                )
            )
            depths = {_depth(c) for c in in_section if _depth(c) is not None}
            if len(depths) >= 2:
                depth_section, depth_hits = candidate, in_section
                break
        if depth_section is None:
            continue  # this probe lacks same-section depth variety; try the next
        return {
            "probe": probe,
            "gold_section": gold_section,
            "cross_decoys": cross_decoys,
            "depth_section": depth_section,
            "depth_hits": depth_hits,
        }
    raise AssertionError(
        "no probe yielded both a cross-section decoy and same-section depth "
        "variety on the live corpus"
    )


def test_filter_predicate_removes_cross_section_decoy(
    source: NeonChunkSource, filter_scenario: dict[str, Any]
) -> None:
    """A ``handbook_section`` predicate travels end to end and removes the decoy (B7).

    Via ``NeonChunkSource.search(SearchSpec(filter=...))``: every returned chunk is in
    the gold section, and the cross-section decoy that surfaced unfiltered is gone —
    proving the predicate actually constrains the set at the ChunkSource layer.
    """
    section = filter_scenario["gold_section"]
    filtered = source.search(
        SearchSpec(
            mode="lexical",
            text_query=filter_scenario["probe"],
            top_k=15,
            filter=field("handbook_section").eq(section),
        )
    )
    assert filtered, "filtered search returned nothing"
    assert all(_section(c) == section for c in filtered), "section filter leaked"
    survived = filter_scenario["cross_decoys"] & {c.hash for c in filtered}
    assert not survived, f"cross-section decoy survived the filter: {survived}"


def test_filter_depth_clause_narrows_within_section(
    source: NeonChunkSource, filter_scenario: dict[str, Any]
) -> None:
    """An ``and``-ed ``path_depth`` clause narrows the set inside one section (B7).

    Section equality alone cannot remove a same-section chunk at a different depth;
    the depth clause is what isolates it. Proven live: a same-section other-depth
    chunk present under the section-only filter is absent once the depth clause is
    ``and``-ed in. The scenario fixture guarantees such a decoy exists (no skip).
    """
    section = filter_scenario["depth_section"]
    in_section = filter_scenario["depth_hits"]
    assert section is not None and in_section, "fixture must supply a depth scenario"
    target_depth = _depth(in_section[0])
    same_section_other_depth = {c.hash for c in in_section if _depth(c) != target_depth}
    assert same_section_other_depth, "no same-section other-depth decoy to remove"
    narrowed = source.search(
        SearchSpec(
            mode="lexical",
            text_query=filter_scenario["probe"],
            top_k=25,
            filter=all_of(
                field("handbook_section").eq(section),
                field("path_depth").eq(target_depth),
            ),
        )
    )
    assert narrowed, "section+depth filter returned nothing"
    assert all(
        _section(c) == section and _depth(c) == target_depth for c in narrowed
    ), "section+depth filter leaked a wrong section/depth"
    survived = same_section_other_depth & {c.hash for c in narrowed}
    assert not survived, f"depth clause failed to remove same-section chunk: {survived}"


def test_env_client_search_genuinely_cannot_carry_filter(embed_fn: Any) -> None:
    """The env SearchClient surface genuinely cannot carry a filter — B7's motivation.

    Not a fake negative: we pin the EXACT accepted parameter set of
    ``NeonSearch.search`` (no ``**kwargs`` escape hatch) AND prove that passing a real
    ``FilterPredicate`` is rejected with ``TypeError`` — so a filter cannot ride this
    path even accidentally, forcing filtered retrieval onto the ChunkSource surface.
    """
    params = inspect.signature(NeonSearch.search).parameters
    assert list(params) == ["self", "query", "mode", "top_k"], list(params)

    client = NeonSearch(LOGICAL, embed_fn=embed_fn, dsn_provider=RO_DSN)
    predicate = field("handbook_section").eq("engineering")
    with pytest.raises(TypeError):
        client.search(PROBE, filter=predicate)  # type: ignore[call-arg]


# --- (3) pickle-safe env client roundtrip -------------------------------------


def test_pickle_roundtrip_keeps_dsn_out_of_reference_artifact(embed_fn: Any) -> None:
    """The reference-path env client cloudpickles without leaking the DSN, then searches.

    ``dsn_provider=None`` is the reference path: the DSN is read from
    ``NEON_CORPUS_DSN_RO`` at query time, so the pickled artifact captures the env-var
    NAME, never its value. The revived client still returns live results because the
    environment supplies the DSN on the next connect.
    """
    client = NeonSearch(LOGICAL, embed_fn=embed_fn, dsn_provider=None)
    blob = cloudpickle.dumps(client)
    assert RO_DSN.encode() not in blob, "read DSN leaked into the pickled artifact"
    assert b"NEON_CORPUS_DSN_RO" in blob, "reference to the DSN env var missing"

    revived = cloudpickle.loads(blob)
    assert revived.get_params() == {
        "backend": "neon",
        "table": LOGICAL,
        "schema": CORPUS_SCHEMA,
    }
    results = _cold_start_retry(
        lambda: revived.search(PROBE, mode="lexical", top_k=TOP_K)
    )
    assert results, "revived client returned nothing"
    assert all(r["content"] for r in results)


# --- (4) dead cached connection is retryable, not a hang ----------------------


def test_dead_cached_connection_reconnects_without_hang() -> None:
    """A server-killed cached socket surfaces as a retryable error the client reconnects past.

    Terminates the live backend from a SECOND connection (``pg_terminate_backend``) so
    the cached handle is dead server-side but NOT client-closed — exercising the
    ``OperationalError`` retry branch, not the proactive ``closed`` reconnect. The
    reconnecting read runs under a hard wall-clock deadline so a genuine hang FAILS
    instead of blocking forever, and we assert it landed on a fresh backend pid.
    """
    client = NeonClient(lambda: RO_DSN)
    _cold_start_retry(lambda: client.execute(sql.SQL("SELECT 1")))
    old_pid = client.execute(sql.SQL("SELECT pg_backend_pid()"))[0][0]

    killer = psycopg.connect(RO_DSN, prepare_threshold=None)
    try:
        killer.execute("SELECT pg_terminate_backend(%s)", (old_pid,))
        killer.commit()
    finally:
        killer.close()

    # the cached handle must not be pre-marked closed — otherwise _live_conn would
    # reconnect before executing and the retry branch would never be exercised.
    assert client._conn is not None and client._conn.closed == 0

    with _deadline(30):  # a real hang fails here instead of blocking forever
        rows = client.execute(sql.SQL("SELECT 1"))
    assert rows[0][0] == 1

    new_pid = client.execute(sql.SQL("SELECT pg_backend_pid()"))[0][0]
    assert new_pid != old_pid, "expected reconnection to a fresh backend"
