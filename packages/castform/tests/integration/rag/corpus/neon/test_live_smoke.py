"""@integration live smoke test for the Neon Lakebase corpus provider (Slice 3).

Builds the tiny, deterministic, known-relevance fixture as the WRITER role, then
runs one vector, one BM25, and one metadata-filtered query UNDER THE READ-ONLY
role (NEON_CORPUS_DSN_RO) through the owner-rights reader view — asserting
correctly-ordered ranked rows and, for BM25, the negative-score ASC polarity
(``<@>`` yields lower = more relevant). This exercises the frozen ANN / BM25 / GIN
DDL end to end against a real Neon compute.

Requires NEON_CORPUS_DSN_RW + NEON_CORPUS_DSN_RO (and the ``neon`` extra); the
whole module is skipped when they are absent, so a default ``pytest`` run — which
also deselects ``-m integration`` — never needs live credentials. Run explicitly:

    uv run --extra neon python -m pytest -m integration \\
        tests/integration/rag/corpus/neon/test_live_smoke.py

Neon scale-to-zero: the first connection wakes a suspended compute (cold start),
so the writer build is wrapped in a bounded connect-retry and the read client
relies on the client's own bounded reconnect.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from castform.rag.corpus.neon import sample_fixture as sf  # noqa: E402
from castform.rag.corpus.neon.client import NeonClient  # noqa: E402

pytestmark = pytest.mark.integration

_ENV_FILE = Path.home() / ".config" / "neon-benchmax.env"


def _load_env_file() -> None:
    """Best-effort load of the developer-local env file for the DSNs (no hard dep)."""
    if os.environ.get("NEON_CORPUS_DSN_RW") and os.environ.get("NEON_CORPUS_DSN_RO"):
        return
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        m = re.match(r'^([A-Z_]+)="?([^"]*)"?$', line.strip())
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2)


_load_env_file()
RW_DSN = os.environ.get("NEON_CORPUS_DSN_RW")
RO_DSN = os.environ.get("NEON_CORPUS_DSN_RO")

if not RW_DSN or not RO_DSN:
    pytest.skip(
        "NEON_CORPUS_DSN_RW / NEON_CORPUS_DSN_RO not set (run provision + set env)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def smoke_corpus() -> object:
    """Build + activate the smoke corpus once as the writer, tolerating cold start."""
    writer = NeonClient(lambda: RW_DSN)
    last: Exception | None = None
    for _ in range(4):  # bounded retry for scale-to-zero wake latency
        try:
            sf.load_smoke_corpus(writer)
            return sf.smoke_spec()
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            last = exc
            time.sleep(5)
    raise AssertionError(f"could not build smoke corpus (cold start?): {last}")


def _ro_client() -> NeonClient:
    return NeonClient(lambda: RO_DSN)


def test_vector_query_ro_returns_ordered_neighbors(smoke_corpus: object) -> None:
    """RO vector query returns the known nearest-first ordering, distances ascending."""
    ro = _ro_client()
    query, params = ro.vector_candidates_sql(smoke_corpus)
    rows = ro.execute(query, {**params, "vector": sf.query_vector(), "top_k": 3})
    ids = [r[0] for r in rows]
    distances = [r[-1] for r in rows]  # native_score = cosine distance
    assert ids == list(sf.VECTOR_EXPECTED_ORDER)
    assert distances == sorted(distances)  # nearer (smaller distance) first
    assert distances[0] == pytest.approx(0.0, abs=1e-6)  # exact-match target


def test_bm25_query_ro_negative_score_asc_polarity(smoke_corpus: object) -> None:
    """RO BM25 query: best hit first, scores NEGATIVE (lower = more relevant), ASC."""
    ro = _ro_client()
    query, params = ro.bm25_candidates_sql(smoke_corpus, schema=sf.CORPUS_SCHEMA)
    rows = ro.execute(query, {**params, "text": sf.BM25_QUERY, "top_k": 5})
    assert rows, "bm25 returned no rows"
    ids = [r[0] for r in rows]
    scores = [r[-1] for r in rows]
    assert ids[0] == sf.BM25_EXPECTED_TOP_ID  # most quick/brown/fox terms
    # <@> polarity: every score <= 0, and the best hit is strictly negative.
    assert all(s <= 0 for s in scores)
    assert scores[0] < 0
    # negative-score means ORDER BY ASC is best-first — assert it is actually sorted.
    assert scores == sorted(scores)


def test_filtered_query_ro_applies_containment(smoke_corpus: object) -> None:
    """RO filtered vector query returns only rows matching the @> containment filter."""
    from psycopg import sql

    ro = _ro_client()
    where = sql.SQL("metadata @> %(f0)s::jsonb")
    query, params = ro.vector_candidates_sql(smoke_corpus, where=where)
    rows = ro.execute(
        query,
        {**params, "vector": sf.query_vector(), "top_k": 8, "f0": '{"lang": "fr"}'},
    )
    ids = {r[0] for r in rows}
    assert ids == set(sf.FILTER_EXPECTED_IDS)  # only the french docs


def test_ro_role_cannot_write(smoke_corpus: object) -> None:
    """The read-only surface is SELECT-only — DDL and DML (INSERT/UPDATE) are denied."""
    from psycopg import sql

    from castform.rag.corpus.neon.schema import physical_table_name, view_name

    table = sql.Identifier(physical_table_name(sf.SMOKE_LOGICAL_NAME, sf.SMOKE_VERSION))
    view = sql.Identifier(view_name(sf.SMOKE_LOGICAL_NAME))
    # A fresh client per attempt: a denied statement aborts the transaction, so the
    # next write must not reuse that failed connection.
    denied = (
        sql.SQL("CREATE TABLE _ro_should_not_exist (x int)"),
        sql.SQL("INSERT INTO {} (id) VALUES ('ro-write')").format(table),
        sql.SQL("UPDATE {} SET content = 'x'").format(view),
    )
    for statement in denied:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _ro_client().execute(statement)
