"""@integration live proof of the F7 ``lakebase_bm25.prefilter`` behavior (Slice 4).

The query layer enables ``SET LOCAL lakebase_bm25.prefilter = on`` before every
filtered lexical / hybrid SELECT (``query.py::_prefilter_setup`` fed into
``NeonClient.read_in_snapshot``). The claim it rests on: when the planner drives
the filtered BM25 query through the
``lakebase_bm25`` *ordered index scan*, that access method returns only a bounded
candidate window (empirically ~1000 rows on this dev build) in score order, and a
plain post-scan ``WHERE`` then removes the non-matching rows — so a strict, low
ranked metadata filter can silently UNDERFILL ``top_k`` (down to zero). Pushing
the predicate into the scan via the prefilter GUC restores a full ``top_k``.

This module proves that end to end against a real Neon Lakebase compute:

* the filtered lexical + hybrid query-layer paths recall ONLY matching rows;
* WITHOUT prefilter the BM25 ordered index scan underfills, WITH prefilter it
  fills (the core proof) — the underfill-prone plan is forced with planner GUCs
  because, for a *selective* filter, the planner otherwise prefers the
  ``_meta_gin`` bitmap plan (which never underfills), so the prefilter guard only
  bites in the ordered-index-scan regime that this test pins;
* EXPLAIN shows the ``_ann`` / ``_bm25`` indexes are really used (Index Scan);
* the GUC is transaction-scoped — it resets after COMMIT and after ROLLBACK.

Requires NEON_CORPUS_DSN_RW + NEON_CORPUS_DSN_RO (and the ``neon`` extra); the
module skips when they are absent. Run explicitly::

    uv run --extra neon python -m pytest -m integration \\
        tests/integration/rag/corpus/neon/test_neon_prefilter_live.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg")

from castform.rag.corpus.neon.client import NeonClient  # noqa: E402
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA  # noqa: E402
from castform.rag.corpus.neon.query import (  # noqa: E402
    NeonQueryRequest,
    run_query,
)
from castform.rag.corpus.neon.schema import (  # noqa: E402
    DEFAULT_TEXT_SEARCH_CONFIG,
    EMBEDDING_DIM,
    NeonTableSpec,
    ReadGrantSpec,
    physical_table_name,
    view_name,
)
from castform.rag.corpus.search_schema.search_types import FieldPredicate  # noqa: E402

pytestmark = pytest.mark.integration

_ENV_FILE = Path.home() / ".config" / "neon-benchmax.env"

# --- fixture shape ------------------------------------------------------------
# N rows all lexically relevant to QTERM. Only the last N_TARGET carry
# metadata {"lang": "target"} and they are engineered to rank LOWEST for QTERM
# (one occurrence buried in unique filler vs. 20 occurrences in the rest), so the
# BM25 ordered index scan's bounded candidate window (~1000) is entirely
# non-target and a post-scan filter underfills to zero.
LOGICAL = "benchmax_prefilter_live"
VERSION = 1
RO_ROLE = "benchmax_ro"
QTERM = "quick"
N_ROWS = 1500
N_TARGET = 8
TOP_K = 8
TARGET_LANG = "target"
FILTER = FieldPredicate(field="lang", op="eq", value=TARGET_LANG)

_SPEC = NeonTableSpec(logical_name=LOGICAL, version=VERSION)


def _load_env_file() -> None:
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


def _embedding(i: int) -> list[float]:
    """Deterministic dummy embedding — a few varied leading axes so the ANN index
    has distinct neighbours (the vector leg is not the focus of this module)."""
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = (i % 7) / 7.0
    vec[1] = (i % 5) / 5.0
    vec[2] = (i % 3) / 3.0 + 0.01
    return vec


def _rows() -> list[tuple[Any, ...]]:
    from psycopg.types.json import Jsonb

    target_ids = set(range(N_ROWS - N_TARGET, N_ROWS))
    out: list[tuple[Any, ...]] = []
    for i in range(N_ROWS):
        if i in target_ids:
            lang = TARGET_LANG
            content = f"{QTERM} " + " ".join(f"filler{i}x{j}" for j in range(40))
        else:
            lang = "other"
            content = (f"{QTERM} " * 20) + f"other{i}"
        out.append(
            (f"pf-{i}", content, Jsonb({"lang": lang}), _embedding(i), f"pf-{i}.txt", 0)
        )
    return out


def _drop(writer: NeonClient) -> None:
    from psycopg import sql

    writer.execute(
        sql.SQL("DROP VIEW IF EXISTS {} CASCADE").format(
            sql.Identifier(view_name(LOGICAL))
        )
    )
    writer.execute(
        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.Identifier(physical_table_name(LOGICAL, VERSION))
        )
    )
    writer.execute(
        sql.SQL(
            "DELETE FROM neon_corpus_versions "
            "WHERE logical_name = %(logical)s AND version = %(version)s"
        ),
        {"logical": LOGICAL, "version": VERSION},
    )


@pytest.fixture(scope="module")
def corpus() -> Any:
    """Build + activate the throwaway underfill corpus, tolerating cold start; drop it after."""
    writer = NeonClient(lambda: RW_DSN)
    grant = ReadGrantSpec(schema=CORPUS_SCHEMA, view=view_name(LOGICAL), ro_role=RO_ROLE)
    last: Exception | None = None
    for _ in range(4):  # bounded retry for scale-to-zero wake latency
        try:
            for stmt in writer.create_ledger_sql():
                writer.execute(stmt)
            _drop(writer)  # idempotent reset so allocate_version starts fresh
            writer.build_version(_SPEC, _rows())
            writer.activate(_SPEC, grant)
            break
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            last = exc
            time.sleep(5)
    else:
        raise AssertionError(f"could not build prefilter corpus (cold start?): {last}")
    try:
        yield _SPEC
    finally:
        _drop(NeonClient(lambda: RW_DSN))


def _ro() -> NeonClient:
    return NeonClient(lambda: RO_DSN)


def _run(request: NeonQueryRequest) -> list[Any]:
    """Drive the real query layer (resolve-in-snapshot + legs) under the RO role."""
    return run_query(
        _ro(),
        request,
        logical_name=LOGICAL,
        schema=CORPUS_SCHEMA,
        text_search_config=DEFAULT_TEXT_SEARCH_CONFIG,
    )


# --- (2a) filtered recall: only matching rows come back -----------------------
#
# These two exercise the REAL query layer end to end (FieldPredicate -> filter_mapper
# -> both legs). They pin the live-only key-cast fix: filter_mapper casts the
# metadata KEY param ``%(k)s::text`` inside ``jsonb_build_object`` (which is VARIADIC
# "any" and cannot infer a bound param's type), so filtered eq/in/contains execute
# against real Postgres instead of raising ``IndeterminateDatatype``.
def test_filtered_lexical_recall_returns_only_matching(corpus: Any) -> None:
    """RO filtered lexical query (the real query layer) returns ONLY lang=target rows."""
    rows = _run(NeonQueryRequest(mode="lexical", top_k=TOP_K, text=QTERM, filter=FILTER))
    assert rows, "filtered lexical returned nothing"
    assert all(r.metadata.get("lang") == TARGET_LANG for r in rows)
    assert {r.chunk_id for r in rows} == {
        f"pf-{i}" for i in range(N_ROWS - N_TARGET, N_ROWS)
    }


def test_filtered_hybrid_recall_returns_only_matching(corpus: Any) -> None:
    """RO filtered hybrid query returns ONLY lang=target rows (filter pushed into both legs)."""
    vector = tuple(0.5 for _ in range(EMBEDDING_DIM))
    rows = _run(
        NeonQueryRequest(
            mode="hybrid", top_k=TOP_K, text=QTERM, vector=vector, filter=FILTER
        )
    )
    assert rows, "filtered hybrid returned nothing"
    assert all(r.metadata.get("lang") == TARGET_LANG for r in rows)


# --- (2b) THE CORE PROOF: prefilter fills top_k; a plain WHERE underfills ------


def _forced_bm25_fill(*, prefilter: bool) -> list[tuple[Any, ...]]:
    """Run the filtered BM25 candidate SELECT forcing the ``_bm25`` ordered index
    scan (the underfill-prone plan). ``prefilter`` toggles ONLY the F7 GUC — the
    planner hints are identical — so any fill difference is attributable to it."""
    from psycopg import sql

    ro = _ro()
    query, params = ro.bm25_candidates_sql(
        _SPEC, where=sql.SQL("metadata @> %(f)s::jsonb"), schema=CORPUS_SCHEMA
    )
    merged = {**params, "text": QTERM, "top_k": TOP_K, "f": '{"lang": "target"}'}
    setup = [
        sql.SQL("SET LOCAL enable_seqscan = off"),
        sql.SQL("SET LOCAL enable_bitmapscan = off"),
    ]
    if prefilter:
        setup.append(sql.SQL("SET LOCAL lakebase_bm25.prefilter = on"))

    def work(conn: Any) -> list[tuple[Any, ...]]:
        return conn.execute(query, merged).fetchall()

    return ro.read_in_snapshot(LOGICAL, work, session_setup=setup)


def test_prefilter_fills_topk_while_plain_where_underfills(corpus: Any) -> None:
    """CORE F7 PROOF: on the ``_bm25`` ordered index scan, a plain WHERE underfills
    (the ~1000-row candidate window is all non-target -> zero survive the filter);
    the prefilter GUC pushes the predicate into the scan and fills top_k."""
    without = _forced_bm25_fill(prefilter=False)
    with_pref = _forced_bm25_fill(prefilter=True)

    # Without prefilter: the post-scan filter starves the result set.
    assert len(without) < TOP_K, (
        f"expected underfill without prefilter, got {len(without)}/{TOP_K} rows "
        "— the ordered index scan may no longer bound its candidate window"
    )
    # With prefilter: a full, correct top_k of matching rows.
    assert len(with_pref) == TOP_K
    assert all(r[2].get("lang") == TARGET_LANG for r in with_pref)


def test_natural_plan_fills_topk_without_forcing(corpus: Any) -> None:
    """With no planner hints the selective filter goes through the ``_meta_gin``
    bitmap plan (filter applied before scoring), so top_k fills even without
    prefilter — raw containment keeps this independent of the filter_mapper key bug."""
    from psycopg import sql

    ro = _ro()
    query, params = ro.bm25_candidates_sql(
        _SPEC, where=sql.SQL("metadata @> %(f)s::jsonb"), schema=CORPUS_SCHEMA
    )
    merged = {**params, "text": QTERM, "top_k": TOP_K, "f": '{"lang": "target"}'}

    def work(conn: Any) -> list[tuple[Any, ...]]:
        return conn.execute(query, merged).fetchall()

    rows = ro.read_in_snapshot(LOGICAL, work)
    assert len(rows) == TOP_K
    assert all(r[2].get("lang") == TARGET_LANG for r in rows)


# --- (2c) EXPLAIN proves the ANN / BM25 indexes are actually used -------------


def _explain(query: Any, params: dict[str, Any]) -> str:
    from pgvector.psycopg import register_vector
    from psycopg import sql

    conn = psycopg.connect(RO_DSN, prepare_threshold=None)
    try:
        register_vector(conn)
        cur = conn.execute(sql.SQL("EXPLAIN (ANALYZE, BUFFERS) ") + query, params)
        return "\n".join(row[0] for row in cur.fetchall())
    finally:
        conn.rollback()
        conn.close()


def test_explain_vector_leg_uses_ann_index(corpus: Any) -> None:
    """EXPLAIN of an unfiltered vector query shows an ``_ann`` Index Scan, not a Seq Scan."""
    ro = _ro()
    query, params = ro.vector_candidates_sql(_SPEC)
    plan = _explain(query, {**params, "vector": [0.5] * EMBEDDING_DIM, "top_k": TOP_K})
    assert re.search(r"Index Scan using \S+_ann\b", plan), plan
    assert "Seq Scan" not in plan, plan


def test_explain_bm25_leg_uses_bm25_index(corpus: Any) -> None:
    """EXPLAIN of an unfiltered BM25 query shows a ``_bm25`` Index Scan, not a Seq Scan."""
    ro = _ro()
    query, params = ro.bm25_candidates_sql(_SPEC, schema=CORPUS_SCHEMA)
    plan = _explain(query, {**params, "text": QTERM, "top_k": TOP_K})
    assert re.search(r"Index Scan using \S+_bm25\b", plan), plan
    assert "Seq Scan" not in plan, plan


# --- (2d) the GUC is SET LOCAL: it resets after COMMIT and after ROLLBACK -----


def _prefilter_setting_after(end: str) -> str | None:
    from psycopg import sql

    conn = psycopg.connect(RO_DSN, prepare_threshold=None)
    try:
        conn.execute(sql.SQL("SET LOCAL lakebase_bm25.prefilter = on"))
        getattr(conn, end)()  # commit or rollback
        cur = conn.execute(
            sql.SQL("SELECT current_setting('lakebase_bm25.prefilter', true)")
        )
        value = cur.fetchone()[0]
        conn.commit()
        return value
    finally:
        conn.close()


@pytest.mark.parametrize("end", ["commit", "rollback"])
def test_prefilter_guc_resets_across_transaction_boundary(corpus: Any, end: str) -> None:
    """``SET LOCAL`` is transaction-scoped: after COMMIT and after ROLLBACK a fresh
    txn sees the default (off), so the GUC never leaks onto a pooled connection."""
    value = _prefilter_setting_after(end)
    assert value in (None, "off", ""), f"prefilter leaked after {end}: {value!r}"
