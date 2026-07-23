"""Measure retrieval quality on the frozen golden and set honest thresholds.

Computes per-mode hit@k / MRR from live Neon retrieval, plus the BM25-only baseline
on the SAME queries. The three thresholds are set consistently:

* lexical — a fixed floor;
* vector — the BM25 baseline on the SAME queries PLUS a margin, so a green gate
  proves the vector leg genuinely beats keyword retrieval rather than that the query
  is BM25-solvable;
* hybrid — a fixed SMOKE floor. Hybrid is a DEFERRED capability: the RRF fusion
  mechanism is real and unit-tested in Slice 4, but a rigorous fusion-necessity gate
  is unbuildable on this lexical- AND vector-strong corpus (deferred to Path X — see
  ``hybrid_rows``). The smoke floor only asserts the fused path runs and finds gold;
  ``hybrid_fusion_lift_hit_at_5`` in the baselines is retained as the honest
  (non-)signal.

Thresholds + baselines are written into the dataset provenance manifest; the live
gate reads them from there.

This is a measurement tool, not a CI step; it runs live retrieval against the
active corpus (RO) and embeds each query once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from castform.rag.corpus.embed import platform_embed_fn
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.eval_metrics import evaluate_record, hit_at_k
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA
from castform.rag.corpus.neon.query import NeonQueryRequest, run_query
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG
from castform.rag.corpus.search_schema.dsl_parser import dsl_to_predicate

K = 5
DEFAULT_MARGIN = 0.15
LEXICAL_FLOOR_HIT = 0.80
LEXICAL_FLOOR_MRR = 0.60
# Hybrid is a DEFERRED smoke capability (fusion-necessity is unbuildable on this
# lexical- AND vector-strong corpus; see hybrid_rows). Its threshold is a loose smoke
# floor that only asserts the fused path runs and finds gold — NOT a baseline+margin
# fusion claim. hybrid_fusion_lift_hit_at_5 stays in the baselines as the honest
# (non-)signal.
HYBRID_SMOKE_HIT = 0.80
HYBRID_SMOKE_MRR = 0.50
LLM_URL = "https://llm.castform.dev/v1"
LOGICAL = "gitlab_handbook_neon"
_ENV_FILE = Path.home() / ".config" / "neon-benchmax.env"


def _load_env() -> str:
    if not os.environ.get("NEON_CORPUS_DSN_RO") and _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            m = re.match(r'^([A-Z_]+)="?([^"]*)"?$', line.strip())
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2)
    dsn = os.environ.get("NEON_CORPUS_DSN_RO")
    if not dsn:
        raise SystemExit("NEON_CORPUS_DSN_RO not set")
    return dsn


def _records(path: Path) -> list[NeonEvalRecord]:
    return [
        NeonEvalRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ranked(
    client: NeonClient,
    query: str,
    *,
    mode: str,
    vector: tuple[float, ...] | None,
    predicate: Any,
    top_k: int,
) -> list[str]:
    needs_vec = mode in ("vector", "hybrid")
    request = NeonQueryRequest(
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
        text=query if mode in ("lexical", "hybrid") else None,
        vector=vector if needs_vec else None,
        filter=predicate,
    )
    rows = run_query(
        client,
        request,
        logical_name=LOGICAL,
        schema=CORPUS_SCHEMA,
        text_search_config=DEFAULT_TEXT_SEARCH_CONFIG,
    )
    return [r.chunk_id for r in rows]


def measure(
    golden: Path, *, base_url: str = LLM_URL, margin: float = DEFAULT_MARGIN
) -> dict:
    """Measure metrics + baselines and return a report with proposed thresholds."""
    dsn = _load_env()
    records = _records(golden)
    client = NeonClient(lambda: dsn)
    embed = platform_embed_fn(base_url=base_url)

    need = sorted({r.query for r in records if r.search_mode in ("vector", "hybrid")})
    vmap = dict(zip(need, embed(need), strict=True)) if need else {}

    def pred(r: NeonEvalRecord) -> Any:
        return dsl_to_predicate(r.filter_dsl) if r.filter_dsl else None

    def evals(recs: list[NeonEvalRecord], mode: str) -> list:
        out = []
        for r in recs:
            vec = tuple(vmap[r.query]) if r.query in vmap else None
            ids = _ranked(
                client, r.query, mode=mode, vector=vec, predicate=pred(r), top_k=10
            )
            out.append(evaluate_record(r, ids, K))
        return out

    report: dict[str, Any] = {"per_mode": {}, "baselines": {}, "margin": margin}
    per_mode_recs = {
        m: [r for r in records if r.search_mode == m]
        for m in ("lexical", "vector", "hybrid")
    }
    for mode, recs in per_mode_recs.items():
        if not recs:
            continue
        ev5 = evals(recs, mode)
        ev10 = [
            evaluate_record(
                r,
                _ranked(
                    client,
                    r.query,
                    mode=mode,
                    vector=tuple(vmap[r.query]) if r.query in vmap else None,
                    predicate=pred(r),
                    top_k=10,
                ),
                10,
            )
            for r in recs
        ]
        h5, h10 = hit_at_k(ev5), hit_at_k(ev10)
        mrr5 = sum(e.reciprocal_rank for e in ev5) / len(ev5)
        # Invariant: hit@k / MRR are per-query means of a per-query any-hit boolean
        # and a reciprocal rank, so each is in [0, 1] by construction. Assert it so a
        # metric regression (e.g. summing total gold hits over the equivalence set)
        # is caught here, not shipped into a threshold.
        assert 0.0 <= h5 <= 1.0 and 0.0 <= h10 <= 1.0 and 0.0 <= mrr5 <= 1.0, (
            mode, h5, h10, mrr5
        )
        report["per_mode"][mode] = {
            "n": len(recs),
            "hit_at_5": round(h5, 4),
            "hit_at_10": round(h10, 4),
            "mrr_at_5": round(mrr5, 4),
            "decoy_violations": sum(e.decoy_violation for e in ev5),
        }

    # BM25-only baseline on vector queries (vector rows carry no filter).
    v_recs = per_mode_recs["vector"]
    if v_recs:
        v_lex = evals(v_recs, "lexical")
        report["baselines"]["vector_bm25_hit_at_5"] = round(hit_at_k(v_lex), 4)
        report["baselines"]["vector_bm25_mrr_at_5"] = round(
            sum(e.reciprocal_rank for e in v_lex) / len(v_lex), 4
        )
    # hybrid single-leg baselines (each with the row's filter applied), retained as the
    # honest fusion (non-)signal only. The hybrid threshold itself is a fixed SMOKE
    # floor (deferred capability), NOT baseline+margin — see _thresholds / hybrid_rows.
    h_recs = per_mode_recs["hybrid"]
    if h_recs:
        h_lex = evals(h_recs, "lexical")
        h_vec = evals(h_recs, "vector")
        report["baselines"]["hybrid_lexical_only_hit_at_5"] = round(hit_at_k(h_lex), 4)
        report["baselines"]["hybrid_vector_only_hit_at_5"] = round(hit_at_k(h_vec), 4)
        report["baselines"]["hybrid_lexical_only_mrr_at_5"] = round(
            sum(e.reciprocal_rank for e in h_lex) / len(h_lex), 4
        )
        report["baselines"]["hybrid_vector_only_mrr_at_5"] = round(
            sum(e.reciprocal_rank for e in h_vec) / len(h_vec), 4
        )
        hyb_hit = report["per_mode"].get("hybrid", {}).get("hit_at_5", 0.0)
        best_leg = max(
            report["baselines"]["hybrid_lexical_only_hit_at_5"],
            report["baselines"]["hybrid_vector_only_hit_at_5"],
        )
        # fusion lift: how much the fused hit@5 beats the best single leg. <=0 means a
        # single mode already solves the rows, so a "beats both legs" gate is vacuous.
        report["baselines"]["hybrid_fusion_lift_hit_at_5"] = round(hyb_hit - best_leg, 4)

    report["thresholds"] = _thresholds(report, margin)
    return report


def _thresholds(report: dict, margin: float) -> dict:
    """Derive per-mode thresholds: a fixed floor for lexical, the BM25 baseline +
    margin for vector (which must beat keyword retrieval), and a loose smoke floor for
    hybrid (a DEFERRED capability — see :data:`HYBRID_SMOKE_HIT`).

    Vector is clamped to ``[0, 1]`` — a hit@k / MRR threshold above 1.0 is unreachable
    and hides, rather than proves, a capability. ``hybrid_fusion_lift_hit_at_5`` in the
    baselines stays as the honest fusion (non-)signal even though hybrid is smoke.
    """

    def clamp(x: float) -> float:
        return round(min(max(x, 0.0), 1.0), 3)

    out: dict[str, dict] = {}
    lex = report["per_mode"].get("lexical")
    if lex:
        out["lexical"] = {"hit_at_k": LEXICAL_FLOOR_HIT, "mrr_at_k": LEXICAL_FLOOR_MRR, "k": K}
    base = report["baselines"]
    if "vector_bm25_hit_at_5" in base:
        out["vector"] = {
            "hit_at_k": clamp(base["vector_bm25_hit_at_5"] + margin),
            "mrr_at_k": clamp(base["vector_bm25_mrr_at_5"] + margin * 0.7),
            "k": K,
        }
    if report["per_mode"].get("hybrid"):
        # Smoke floor only — deferred capability, no fusion-necessity claim.
        out["hybrid"] = {"hit_at_k": HYBRID_SMOKE_HIT, "mrr_at_k": HYBRID_SMOKE_MRR, "k": K}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument(
        "--provenance", type=Path, help="provenance.json to write thresholds into"
    )
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument(
        "--write", action="store_true", help="write thresholds into --provenance"
    )
    args = p.parse_args()

    report = measure(args.golden, margin=args.margin)
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.write and args.provenance and args.provenance.exists():
        manifest = json.loads(args.provenance.read_text())
        manifest["thresholds"] = report["thresholds"]
        manifest["threshold_basis"] = {
            "baselines": report["baselines"],
            "margin": args.margin,
            "rule": (
                "lexical=fixed floor; vector=bm25 baseline+margin; "
                "hybrid=fixed smoke floor (deferred capability, no fusion claim)"
            ),
        }
        manifest["measured_metrics"] = report["per_mode"]
        args.provenance.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote thresholds into {args.provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
