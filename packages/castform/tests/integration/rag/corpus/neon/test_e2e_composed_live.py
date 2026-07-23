"""@integration composed end-to-end test for the Neon lakebase corpus provider (Slice 8).

The MANDATORY full-stack integration gate before roll-up. Where Slice 7's
``test_golden_eval_live`` measures golden retrieval *quality*, this module proves the
composed PUBLIC provider surface *plumbs together* end to end against the live
``gitlab_handbook_neon`` corpus (31665 chunks, v1 activated) — nothing here rewards
recall, everything here proves a request travels from the public API through the
query layer to real Neon and back with the right shape.

It exercises three distinct public surfaces:

* :class:`~castform.rag.corpus.neon.source.NeonChunkSource` — the driver-side
  ChunkSource. All FOUR capabilities run through it: lexical (BM25), vector (ANN),
  hybrid (RRF fusion), and FILTERED. Ranked results are non-empty and correctly
  ordered; scores are sane (``search_related`` surfaces a descending ``max_score``,
  the env client a descending surfaced score).
* the FILTER predicate at the ChunkSource layer (B7). The env
  :class:`~castform.rag.corpus.neon.search.NeonSearch` ``search()`` carries only
  ``query``/``mode``/``top_k`` — a filter predicate does NOT survive it — so filtered
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
wrapped in a bounded short-backoff retry — never one long blind sleep.

Requires ``NEON_CORPUS_DSN_RO`` + ``PLATFORM_API_KEY`` (for embeddings) and the
``neon`` extra; the module skips when the DSN is absent. Run explicitly::

    uv run --extra neon python -m pytest -m integration \\
        tests/integration/rag/corpus/neon/test_e2e_composed_live.py
"""

from __future__ import annotations

import inspect
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg")
import cloudpickle  # noqa: E402

from castform.rag.chunkers.models import Chunk  # noqa: E402
from castform.rag.corpus.embed import platform_embed_fn  # noqa: E402
from castform.rag.corpus.neon.client import NeonClient  # noqa: E402
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA  # noqa: E402
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
# a ubiquitous handbook term so the score legs return a full, ranked result set.
PROBE = "communication"
# candidate probes for the filter scenario; the first that spans >=2 sections wins.
FILTER_PROBES = ("team", "process", "communication", "policy", "manager", "engineering")


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


def _cold_start_retry(fn: Any, *, attempts: int = 6, backoff: float = 4.0) -> Any:
    """Run *fn*, absorbing a Neon scale-to-zero cold start.

    The hardened ``NeonClient`` (B16 connect_timeout + keepalives) surfaces a
    suspended-compute wake or a dead cached socket as a retryable
    ``OperationalError``/``InterfaceError`` rather than hanging on a half-open
    connection. We retry a bounded number of times with a short backoff — never one
    long blind sleep — so the first query after idle succeeds once the compute wakes.
    Non-retryable errors (bad SQL, assertion failures) propagate immediately.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            last = exc
            time.sleep(backoff)
    raise AssertionError(f"live neon unreachable after {attempts} attempts: {last}")


def _section(chunk: Chunk) -> str | None:
    return chunk.get_metadata("handbook_section")


def _depth(chunk: Chunk) -> int | None:
    return chunk.get_metadata("path_depth")


@pytest.fixture(scope="module")
def embed_fn() -> Any:
    """Platform embedder for vector/hybrid modes (resolves PLATFORM_API_KEY at call)."""
    return platform_embed_fn(base_url=LLM_URL)


@pytest.fixture(scope="module")
def source(embed_fn: Any) -> NeonChunkSource:
    """Read-only NeonChunkSource over the live corpus, warmed past cold start.

    Warming with a single ``get_chunk_count`` both wakes a suspended compute (via the
    bounded reconnect) and pins that we are querying the expected 31665-chunk version,
    so the capability tests below hit a warm compute.
    """
    src = NeonChunkSource(LOGICAL, embed_fn=embed_fn, read_dsn_provider=RO_DSN)
    count = _cold_start_retry(src.get_chunk_count)
    assert count == EXPECTED_CHUNK_COUNT, f"unexpected corpus size {count}"
    return src


@pytest.fixture(scope="module")
def query_vector(source: NeonChunkSource) -> list[float]:
    """One live embedding of :data:`PROBE` reused by the vector/hybrid capabilities."""
    vector = _cold_start_retry(lambda: source.embed_query(PROBE))
    assert vector is not None and len(vector) > 0
    return vector


# --- (1) four capabilities through NeonChunkSource ----------------------------


def test_lexical_capability_returns_ranked_chunks(source: NeonChunkSource) -> None:
    """Lexical (BM25) search returns a non-empty, bounded, content-bearing chunk set."""
    chunks = _cold_start_retry(
        lambda: source.search(
            SearchSpec(mode="lexical", text_query=PROBE, top_k=TOP_K)
        )
    )
    assert chunks, "lexical search returned nothing"
    assert len(chunks) <= TOP_K
    assert all(isinstance(c, Chunk) and c.content for c in chunks)
    assert all(_section(c) is not None for c in chunks)


def test_vector_capability_returns_ranked_chunks(
    source: NeonChunkSource, query_vector: list[float]
) -> None:
    """Vector (ANN) search over a live 3072-dim query embedding returns ranked chunks."""
    chunks = _cold_start_retry(
        lambda: source.search(
            SearchSpec(mode="vector", vector_query=query_vector, top_k=TOP_K)
        )
    )
    assert chunks, "vector search returned nothing"
    assert len(chunks) <= TOP_K
    assert all(isinstance(c, Chunk) and c.content for c in chunks)


def test_hybrid_capability_returns_ranked_chunks(
    source: NeonChunkSource, query_vector: list[float]
) -> None:
    """Hybrid (RRF fusion of both legs) search returns a ranked, non-empty chunk set."""
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
    assert chunks, "hybrid search returned nothing"
    assert len(chunks) <= TOP_K
    assert all(isinstance(c, Chunk) and c.content for c in chunks)


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_search_related_surfaces_descending_max_score(
    source: NeonChunkSource, mode: str
) -> None:
    """``search_related`` surfaces a descending ``max_score`` on every mode.

    A synthetic source chunk (absent from the corpus, no ``file`` metadata) keeps
    ``same_file`` uniformly false, so the result sort collapses to pure ``max_score``
    descending — letting us assert the surfaced score ordering directly.
    """
    synthetic = Chunk(content="", metadata=(), hash="__slice8_synthetic_source__")
    results = _cold_start_retry(
        lambda: source.search_related(synthetic, [PROBE], top_k=TOP_K, mode=mode)
    )
    assert results, f"{mode} search_related returned nothing"
    scores = [r["max_score"] for r in results]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True), scores
    assert all("native_score" in r for r in results)


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_env_client_search_surfaced_score_descending(
    embed_fn: Any, mode: str
) -> None:
    """The env ``NeonSearch.search`` returns dicts with a descending surfaced score.

    Surfaced score is the ordinal ``1/(60+rank)`` (higher-better, uniform across
    modes), so it is strictly in ``(0, 1/60]`` and monotonically non-increasing.
    """
    client = NeonSearch(LOGICAL, embed_fn=embed_fn, dsn_provider=RO_DSN)
    results = _cold_start_retry(lambda: client.search(PROBE, mode=mode, top_k=TOP_K))
    assert results, f"{mode} env search returned nothing"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] == max(scores)
    assert all(0.0 < s <= 1.0 for s in scores)
    assert all(r["content"] for r in results)


# --- (2) the filter predicate travels at the ChunkSource layer (B7) -----------


@pytest.fixture(scope="module")
def filter_scenario(source: NeonChunkSource) -> dict[str, Any]:
    """Derive a live filter scenario from the corpus (no golden-file dependency).

    Picks the first probe whose unfiltered lexical top-k spans >=2 sections (so a
    genuine cross-section decoy exists), then looks for a section among those hits
    with >=2 distinct ``path_depth`` values (so the depth clause has something to
    narrow). Fails loudly if no multi-section probe exists — that would be a real
    corpus/plumbing regression, not a reason to skip.
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
        for candidate in [gold_section, *distinct]:
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
        return {
            "probe": probe,
            "gold_section": gold_section,
            "cross_decoys": cross_decoys,
            "depth_section": depth_section,
            "depth_hits": depth_hits,
        }
    raise AssertionError(
        "no probe produced a multi-section result set on the live corpus"
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
    ``and``-ed in.
    """
    section = filter_scenario["depth_section"]
    in_section = filter_scenario["depth_hits"]
    if section is None or in_section is None:
        pytest.skip("live top hits lack same-section depth variety for this probe")
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


def test_env_client_search_cannot_carry_filter(
    embed_fn: Any, filter_scenario: dict[str, Any]
) -> None:
    """The env SearchClient surface has no filter — the ChunkSource path is required.

    ``NeonSearch.search`` accepts only ``query``/``mode``/``top_k`` (no ``filter``
    parameter), so the same probe returns its natural, unconstrained multi-section
    set here — the contrast that motivates B7's ChunkSource-level filter path.
    """
    assert "filter" not in inspect.signature(NeonSearch.search).parameters
    client = NeonSearch(LOGICAL, embed_fn=embed_fn, dsn_provider=RO_DSN)
    results = _cold_start_retry(
        lambda: client.search(filter_scenario["probe"], mode="lexical", top_k=15)
    )
    sections = {r["metadata"].get("handbook_section") for r in results}
    assert len(sections) >= 2, "env client unexpectedly returned a single section"


# --- (3) pickle-safe env client roundtrip -------------------------------------


def test_env_client_pickle_roundtrip_keeps_dsn_out_of_reference_artifact(
    embed_fn: Any,
) -> None:
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
    """A killed cached socket surfaces as a retryable error the client reconnects past.

    Simulates a Neon scale-to-zero-killed connection by closing the cached handle,
    then asserts the next read transparently reconnects (B16) and returns — the
    hardened path converts a dead socket into a fresh connection rather than hanging.
    """
    from psycopg import sql

    client = NeonClient(lambda: RO_DSN)
    first = _cold_start_retry(lambda: client.execute(sql.SQL("SELECT 1")))
    assert first[0][0] == 1
    client._conn.close()  # simulate the autosuspend-killed cached socket
    second = client.execute(sql.SQL("SELECT 1"))  # bounded reconnect, no hang
    assert second[0][0] == 1
