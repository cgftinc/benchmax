"""Descriptive before/after report for the frozen golden (Step-5 reporting only).

Measures, over live Neon retrieval, how much the R3 equivalence-set expansion lifts
the VECTOR metric — scoring the SAME retrieval results against the raw single-anchor
gold (from ``verdicts_v2``, pre-expansion) versus the equivalence-set gold (the
frozen record) — and the BM25-only baseline on the same vector queries plus the
vector margin over it. Writes nothing; it only prints numbers for the report. The
gate thresholds are written by ``measure.py``; pass/fail is enforced by the live
integration gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from measure import K, LLM_URL, _load_env, _ranked, _records
from castform.rag.corpus.embed import platform_embed_fn
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.eval_metrics import is_hit, reciprocal_rank

_DATASETS = Path(__file__).resolve().parent / "datasets"
_GOLDEN = _DATASETS / "gitlab_handbook_neon_golden.jsonl"
_VERDICTS = _DATASETS / "verdicts_v2.jsonl"


def _raw_gold_by_query() -> dict[str, list[str]]:
    """Map query -> raw single-anchor gold hashes (kept rows only), pre-expansion."""
    out: dict[str, list[str]] = {}
    for line in _VERDICTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("verdict") != "drop":
            out[row["query"]] = list(row.get("gold_chunk_hashes", []))
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    dsn = _load_env()
    client = NeonClient(lambda: dsn)
    embed = platform_embed_fn(base_url=LLM_URL)
    records = _records(_GOLDEN)
    raw_gold = _raw_gold_by_query()

    vector = [r for r in records if r.search_mode == "vector"]
    vqueries = sorted({r.query for r in vector})
    vmap = dict(zip(vqueries, embed(vqueries), strict=True))

    # one retrieval pass per vector query, scored three ways
    exp_hit, exp_rr, raw_hit, raw_rr, bm25_hit, bm25_rr = ([] for _ in range(6))
    n_expanded = 0
    for r in vector:
        vec = tuple(vmap[r.query])
        vec_ids = _ranked(client, r.query, mode="vector", vector=vec, predicate=None, top_k=10)
        lex_ids = _ranked(client, r.query, mode="lexical", vector=None, predicate=None, top_k=10)
        expanded = set(r.gold_chunk_hashes)
        raw = set(raw_gold.get(r.query, r.gold_chunk_hashes))
        if len(expanded) > len(raw):
            n_expanded += 1
        exp_hit.append(is_hit(expanded, vec_ids, K))
        exp_rr.append(reciprocal_rank(expanded, vec_ids, K))
        raw_hit.append(is_hit(raw, vec_ids, K))
        raw_rr.append(reciprocal_rank(raw, vec_ids, K))
        bm25_hit.append(is_hit(expanded, lex_ids, K))
        bm25_rr.append(reciprocal_rank(expanded, lex_ids, K))

    report = {
        "vector_rows": len(vector),
        "vector_rows_with_expanded_gold": n_expanded,
        "raw_exact_hash": {"hit_at_5": round(_mean(raw_hit), 4), "mrr_at_5": round(_mean(raw_rr), 4)},
        "equivalence_set": {"hit_at_5": round(_mean(exp_hit), 4), "mrr_at_5": round(_mean(exp_rr), 4)},
        "bm25_baseline_same_queries": {"hit_at_5": round(_mean(bm25_hit), 4), "mrr_at_5": round(_mean(bm25_rr), 4)},
        "vector_margin_over_bm25_hit_at_5": round(_mean(exp_hit) - _mean(bm25_hit), 4),
        "equivalence_lift_hit_at_5": round(_mean(exp_hit) - _mean(raw_hit), 4),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
