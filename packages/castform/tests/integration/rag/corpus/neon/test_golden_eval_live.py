"""@integration validity gate for the frozen GitLab-handbook golden set (Slice 7).

Runs the committed, frozen golden JSONL against the live Neon corpus
``gitlab_handbook_neon`` and enforces the acceptance criteria:

* per-mode ``hit@k`` / ``MRR`` clear the MEASURED thresholds (baseline + margin,
  written into the provenance by ``measure.py``; frozen defaults as fallback) and
  NO decoy is surfaced — no xfail masks a mode;
* the vector rows are NOT keyword-solvable — a lexical-only (BM25) ablation of the
  same queries recalls far less, so the delta clears
  :data:`LEXICAL_ABLATION_MIN_DELTA` (proves the semantic rows need the vector
  leg, not just BM25);
* the hybrid gate is non-vacuous — its bar sits above BOTH single-mode legs, so
  neither lexical-only nor vector-only clears it; and the section filter does real
  work — its decoys appear unfiltered and vanish once the predicate is applied;
* at FULL-corpus scale the planner really uses the named ``_ann`` / ``_bm25`` /
  ``_meta_gin`` indexes (``EXPLAIN (FORMAT JSON)``) with NO planner override — a
  tiny fixture would seq-scan optimally, so this must run against the real corpus.

Requires ``NEON_CORPUS_DSN_RO`` (+ the ``neon`` extra) and the frozen golden file;
skips otherwise. The corpus must have been ingested by
``examples/gitlab_handbook_bm25_neon/build_corpus.py`` first. Run::

    uv run --extra neon python -m pytest -m integration \\
        tests/integration/rag/corpus/neon/test_golden_eval_live.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg")

from castform.rag.corpus.embed import platform_embed_fn  # noqa: E402
from castform.rag.corpus.neon.client import NeonClient  # noqa: E402
from castform.rag.corpus.neon.eval_metrics import (  # noqa: E402
    aggregate_by_mode,
    evaluate_record,
    hit_at_k,
    lexical_ablation,
    thresholds_for,
)
from castform.rag.corpus.neon.eval_schema import (  # noqa: E402
    NeonEvalRecord,
    NeonEvalThresholds,
)
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA, RO_ROLE  # noqa: E402
from castform.rag.corpus.neon.query import NeonQueryRequest, run_query  # noqa: E402
from castform.rag.corpus.neon.schema import (  # noqa: E402
    DEFAULT_TEXT_SEARCH_CONFIG,
    EMBEDDING_DIM,
    NeonTableSpec,
    index_names,
    physical_table_name,
)
from castform.rag.corpus.search_schema.dsl_parser import dsl_to_predicate  # noqa: E402

pytestmark = pytest.mark.integration

_ENV_FILE = Path.home() / ".config" / "neon-benchmax.env"
_GOLDEN = (
    Path(__file__).resolve().parents[7]
    / "examples"
    / "gitlab_handbook_bm25_neon"
    / "datasets"
    / "gitlab_handbook_neon_golden.jsonl"
)
_PROVENANCE = _GOLDEN.parent / "provenance.json"
LOGICAL = "gitlab_handbook_neon"
LLM_URL = "https://llm.castform.dev/v1"
TOP_K = 5
EXPECTED_CHUNK_COUNT = 31665


def _load_env_file() -> None:
    if os.environ.get("NEON_CORPUS_DSN_RO"):
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
if not _GOLDEN.exists():
    pytest.skip(f"frozen golden not built yet: {_GOLDEN}", allow_module_level=True)


def _records() -> list[NeonEvalRecord]:
    return [
        NeonEvalRecord.model_validate_json(line)
        for line in _GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ro() -> NeonClient:
    return NeonClient(lambda: RO_DSN)


_EMBED = platform_embed_fn(base_url=LLM_URL)


def _embed(text: str) -> tuple[float, ...]:
    return tuple(_EMBED([text])[0])


def _predicate(record: NeonEvalRecord) -> Any:
    return dsl_to_predicate(record.filter_dsl) if record.filter_dsl else None


def _measured_thresholds() -> dict[str, dict]:
    """The measured per-mode thresholds written into the provenance by ``measure.py``.

    Empty when absent (the gate then falls back to the frozen defaults).
    """
    if not _PROVENANCE.exists():
        return {}
    data = json.loads(_PROVENANCE.read_text(encoding="utf-8"))
    return data.get("thresholds", {})


def thresholds_for_mode(mode: str) -> NeonEvalThresholds:
    """Prefer the measured (baseline+margin) threshold; fall back to the frozen default.

    The measured floor is honest: ``measure.py`` sets vector/hybrid to the BM25 (or
    best single-leg) baseline on the SAME queries plus a margin, so clearing it
    proves the vector leg / fusion genuinely contributes rather than that the query
    is BM25-solvable.
    """
    m = _measured_thresholds().get(mode)
    if m:
        return NeonEvalThresholds(
            hit_at_k=m["hit_at_k"], mrr_at_k=m["mrr_at_k"], k=m.get("k", TOP_K)
        )
    return thresholds_for(mode)  # type: ignore[arg-type]


def _ranked_ids(
    record: NeonEvalRecord,
    *,
    force_mode: str | None = None,
    use_filter: bool = True,
) -> list[str]:
    """Run the record through the real query layer; return ranked chunk ids.

    ``use_filter=False`` drops the record's metadata predicate — used to prove a
    filter is non-vacuous (its decoys appear UNFILTERED and vanish once applied).
    """
    mode = force_mode or record.search_mode
    needs_vector = mode in ("vector", "hybrid")
    request = NeonQueryRequest(
        mode=mode,  # type: ignore[arg-type]
        top_k=TOP_K,
        text=record.query if mode in ("lexical", "hybrid") else None,
        vector=_embed(record.query) if needs_vector else None,
        filter=_predicate(record) if use_filter else None,
    )
    rows = run_query(
        _ro(),
        request,
        logical_name=LOGICAL,
        schema=CORPUS_SCHEMA,
        text_search_config=DEFAULT_TEXT_SEARCH_CONFIG,
    )
    return [r.chunk_id for r in rows]


# --- (1) per-mode hit@k / MRR + decoy exclusion -------------------------------


def _mode_metrics(mode: str):
    records = [r for r in _records() if r.search_mode == mode]
    assert records, f"no {mode} records in frozen golden"
    evals = [evaluate_record(r, _ranked_ids(r), TOP_K) for r in records]
    return aggregate_by_mode(evals, TOP_K)[mode]


def _assert_mode(mode: str) -> None:
    m = _mode_metrics(mode)
    th = thresholds_for_mode(mode)
    assert m.passes(th), (
        f"{mode}: hit@{TOP_K}={m.hit_at_k:.3f}(>= {th.hit_at_k}) "
        f"mrr={m.mrr_at_k:.3f}(>= {th.mrr_at_k}) decoys={m.decoy_violations} n={m.n}"
    )


def test_lexical_metrics_clear_thresholds() -> None:
    """Lexical rows (qa-gen keyword-style + curated filter probes) clear the bar."""
    _assert_mode("lexical")


def test_hybrid_metrics_clear_thresholds() -> None:
    """Hybrid rows (curated multi-term + section filter, multi-gold) clear the bar."""
    _assert_mode("hybrid")


def test_vector_metrics_clear_thresholds() -> None:
    """Vector rows clear the MEASURED floor (BM25 baseline + margin).

    No xfail: the threshold is the measured BM25 baseline on the same queries plus a
    margin (:mod:`examples.gitlab_handbook_bm25_neon.measure`), so it is set at a
    height the vector leg actually reaches while still proving a real margin over
    keyword retrieval. A red here is a genuine regression, not a frozen-ceiling
    artifact.
    """
    _assert_mode("vector")


# --- (2) lexical ablation: vector/hybrid rows are not keyword-solvable ---------


def test_lexical_ablation_semantic_rows_not_keyword_solvable() -> None:
    """Vector rows must not be keyword-solvable: a BM25 ablation of the same queries
    recalls far less, so the delta clears the frozen minimum."""
    mode = "vector"
    records = [r for r in _records() if r.search_mode == mode]
    if not records:
        pytest.skip(f"no {mode} records")
    primary = [evaluate_record(r, _ranked_ids(r), TOP_K) for r in records]
    lexical = [
        evaluate_record(r, _ranked_ids(r, force_mode="lexical"), TOP_K) for r in records
    ]
    result = lexical_ablation(primary, lexical)
    assert result.passes, (
        f"{mode} rows look keyword-solvable: primary hit@{TOP_K}="
        f"{result.primary_hit_at_k:.3f} lexical hit@{TOP_K}={result.lexical_hit_at_k:.3f} "
        f"delta={result.delta:.3f}"
    )


# --- (2b) non-vacuous gates: hybrid needs fusion; the filter does real work -----


def test_hybrid_requires_both_single_legs_to_fail() -> None:
    """The hybrid gate has teeth: its bar sits ABOVE both single-mode legs.

    Retrying the same hybrid queries (same section filter) under lexical-only and
    under vector-only must EACH miss the hybrid hit@k bar, while fusion clears it —
    so no single retrieval mode can pass the hybrid gate on its own. This is the
    honest reading of "both legs fail": neither leg alone reaches the fused bar.
    """
    records = [r for r in _records() if r.search_mode == "hybrid"]
    assert records, "no hybrid records in frozen golden"
    th = thresholds_for_mode("hybrid")
    hybrid = hit_at_k([evaluate_record(r, _ranked_ids(r), TOP_K) for r in records])
    lexical = hit_at_k(
        [evaluate_record(r, _ranked_ids(r, force_mode="lexical"), TOP_K) for r in records]
    )
    vector = hit_at_k(
        [evaluate_record(r, _ranked_ids(r, force_mode="vector"), TOP_K) for r in records]
    )
    assert hybrid >= th.hit_at_k, (
        f"hybrid hit@{TOP_K}={hybrid:.3f} below bar {th.hit_at_k}"
    )
    assert lexical < th.hit_at_k and vector < th.hit_at_k, (
        f"hybrid gate vacuous: a single leg clears the fused bar {th.hit_at_k} "
        f"(lexical-only={lexical:.3f}, vector-only={vector:.3f}, hybrid={hybrid:.3f})"
    )


def test_filter_decoys_appear_unfiltered_then_vanish() -> None:
    """The section predicate does real work, not a no-op AND.

    For every curated filter row the same-token cross-section decoys must appear in
    an UNFILTERED top-``k`` of the query (they are genuine confounders) and then
    VANISH once the predicate is applied, while the gold survives. A filter whose
    decoys never surface unfiltered would be a vacuous gate.
    """
    records = [r for r in _records() if r.capability.startswith("filter_")]
    assert records, "no curated filter records in frozen golden"
    appeared = 0
    filtered_decoys = 0
    gold_hits = 0
    for r in records:
        unfiltered = _ranked_ids(r, use_filter=False)[:TOP_K]
        filtered = _ranked_ids(r, use_filter=True)[:TOP_K]
        decoys = set(r.decoy_chunk_hashes)
        gold = set(r.gold_chunk_hashes)
        if any(d in unfiltered for d in decoys):
            appeared += 1
        if any(d in filtered for d in decoys):
            filtered_decoys += 1
        if any(g in filtered for g in gold):
            gold_hits += 1
    assert appeared >= 1, (
        "filter gate vacuous: no decoy surfaced in any unfiltered top-k, so the "
        "predicate never has a confounder to exclude"
    )
    assert filtered_decoys == 0, f"{filtered_decoys} rows leaked a decoy under the filter"
    assert gold_hits == len(records), (
        f"gold missing under filter for {len(records) - gold_hits}/{len(records)} rows"
    )


# --- (3) EXPLAIN at scale proves the named indexes are used (no override) ------


def _current_spec(client: NeonClient) -> NeonTableSpec:
    rows = client.execute(client.read_ledger_sql(), {"logical": LOGICAL})
    for version, _state, is_current in rows:
        if is_current:
            return NeonTableSpec(
                LOGICAL, version, text_search_config=DEFAULT_TEXT_SEARCH_CONFIG
            )
    raise AssertionError(f"no current published version for {LOGICAL!r}")


def _explain_json(query: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    from pgvector.psycopg import register_vector
    from psycopg import sql

    conn = psycopg.connect(RO_DSN, prepare_threshold=None)
    try:
        register_vector(conn)
        cur = conn.execute(sql.SQL("EXPLAIN (FORMAT JSON) ") + query, params)
        return cur.fetchone()[0]  # [{"Plan": {...}}]
    finally:
        conn.rollback()
        conn.close()


def _index_names_in_plan(
    plan_json: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Walk the plan tree; return (index names used, node types seen)."""
    indexes: list[str] = []
    node_types: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        node_types.append(str(node.get("Node Type", "")))
        if "Index Name" in node:
            indexes.append(str(node["Index Name"]))
        for child in node.get("Plans", []) or []:
            walk(child)

    walk(plan_json[0]["Plan"])
    return indexes, node_types


def test_explain_context_is_ro_full_scale_default_planner() -> None:
    """The EXPLAIN gate runs under the RO role, against the full 31665x3072 corpus,
    with default planner GUCs — so an index-scan result reflects the real planner,
    not an ``enable_seqscan=off`` override or a shrunken fixture."""
    conn = psycopg.connect(RO_DSN, prepare_threshold=None)
    try:
        assert conn.execute("SELECT current_user").fetchone()[0] == RO_ROLE
        for guc in ("enable_seqscan", "enable_indexscan", "enable_bitmapscan"):
            value = conn.execute(f"SELECT current_setting('{guc}')").fetchone()[0]
            assert value == "on", (guc, value)
        n = conn.execute(
            "SELECT count(*) FROM benchmax_corpus.gitlab_handbook_neon"
        ).fetchone()[0]
        assert n == EXPECTED_CHUNK_COUNT, n
    finally:
        conn.rollback()
        conn.close()
    # embedding width is the frozen 3072 on the physical version table
    from psycopg import sql

    ro = _ro()
    spec = _current_spec(ro)
    dim = ro.execute(
        sql.SQL("SELECT vector_dims(embedding) FROM {} LIMIT 1").format(
            sql.Identifier(physical_table_name(LOGICAL, spec.version))
        )
    )[0][0]
    assert dim == EMBEDDING_DIM == 3072, dim


def test_explain_vector_leg_uses_ann_index_at_scale() -> None:
    ro = _ro()
    spec = _current_spec(ro)
    query, params = ro.vector_candidates_sql(spec)
    plan = _explain_json(
        query, {**params, "vector": [0.1] * EMBEDDING_DIM, "top_k": TOP_K}
    )
    indexes, node_types = _index_names_in_plan(plan)
    assert index_names(LOGICAL, spec.version)["ann"] in indexes, (indexes, node_types)
    assert "Seq Scan" not in node_types, node_types


def test_explain_bm25_leg_uses_bm25_index_at_scale() -> None:
    ro = _ro()
    spec = _current_spec(ro)
    query, params = ro.bm25_candidates_sql(spec, schema=CORPUS_SCHEMA)
    plan = _explain_json(query, {**params, "text": "engineering", "top_k": TOP_K})
    indexes, node_types = _index_names_in_plan(plan)
    assert index_names(LOGICAL, spec.version)["bm25"] in indexes, (indexes, node_types)
    assert "Seq Scan" not in node_types, node_types


def test_explain_metadata_filter_uses_meta_gin_at_scale() -> None:
    """A selective ``handbook_section`` containment goes through ``_meta_gin`` with
    no planner override (the natural plan for a selective jsonb filter)."""
    from psycopg import sql

    ro = _ro()
    spec = _current_spec(ro)
    # rarest section so the containment is selective enough that the planner prefers
    # the gin index over a seq scan on its own.
    table = physical_table_name(LOGICAL, spec.version)
    section_rows = ro.execute(
        sql.SQL(
            "SELECT metadata->>'handbook_section' AS s, count(*) c "
            "FROM {} GROUP BY 1 ORDER BY c ASC LIMIT 1"
        ).format(sql.Identifier(table))
    )
    rare_section = section_rows[0][0]
    query = sql.SQL(
        "SELECT id FROM {} WHERE metadata @> %(f)s::jsonb LIMIT %(k)s"
    ).format(sql.Identifier(table))
    plan = _explain_json(
        query, {"f": f'{{"handbook_section": "{rare_section}"}}', "k": TOP_K}
    )
    indexes, node_types = _index_names_in_plan(plan)
    assert index_names(LOGICAL, spec.version)["meta_gin"] in indexes, (
        rare_section,
        indexes,
        node_types,
    )
    assert "Seq Scan" not in node_types, (rare_section, node_types)
