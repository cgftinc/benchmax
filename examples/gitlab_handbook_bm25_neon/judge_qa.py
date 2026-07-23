"""Standalone answerability audit of the on-disk qa rows (Step 2 / Step 3).

Runs the blind answerability judge (:mod:`qa_validity`) over ONLY the natural-language
rows — the reused lexical + vector qa-gen output on disk — and writes a complete,
self-describing ``verdicts_v2.jsonl`` (one row per pair: keep/fix/drop + intrinsic
reason + a citation quoting the gold text that answers the query + a gold preview).
Curated retrieval-precision probes are NOT judged here (category error); they are
authored answerable-by-construction and validated by the live gate instead.

The judge pass is the ONLY spend, is death-tolerant, and is resumable: every new
verdict rewrites the full file immediately, and on restart the existing file is
loaded as a cache so already-judged pairs are not re-spent. The same file is then
fed to ``build_golden.py --verdicts`` so the frozen-set drops reconcile exactly with
this audit (no second judge spend).

Run (creds sourced), detached with a heartbeat log::

    uv run --extra neon python examples/gitlab_handbook_bm25_neon/judge_qa.py \\
        --work-dir examples/gitlab_handbook_bm25_neon/work/handbook_repo \\
        --qa-dir examples/gitlab_handbook_bm25_neon/datasets \\
        --out examples/gitlab_handbook_bm25_neon/datasets/verdicts_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from castform.rag.qa_generation.neon_entrypoint import neon_llm_url

from build_golden import (
    _STYLE_KEYWORD,
    _STYLE_PARAPHRASE,
    _LEXICAL_SYSTEM_PROMPT,
    _VECTOR_SYSTEM_PROMPT,
    _VECTOR_MAX_OVERLAP,
    bind_source,
    run_qa_pass,
)
from handbook_corpus import (
    HANDBOOK_COMMIT,
    HANDBOOK_SUBDIR,
    ChunkerParams,
    build_collection,
    git_tracked_docs,
    sparse_checkout,
)
from qa_validity import (
    NATURAL_CAPABILITIES,
    QaValidityJudge,
    Verdict,
    make_verdict_row,
    pair_key,
)

# Defect ceiling: above this the generator is indicted and we stop, not silently
# ship a mostly-broken dataset (Step 3).
DEFECT_CEILING = 0.35


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _load_done(out_path: Path) -> dict[str, dict]:
    """Load already-written verdict rows keyed by pair identity (resume support)."""
    done: dict[str, dict] = {}
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["query"] + "\x00" + "|".join(sorted(row.get("gold_chunk_hashes", [])))
        done[key] = row
    return done


def _write(out_path: Path, rows: list[dict]) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-domain", default="castform.dev")
    p.add_argument("--lexical-samples", type=int, default=60)
    p.add_argument("--vector-samples", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mirror", type=Path, default=None, help="second copy path")
    args = p.parse_args()

    llm_url = neon_llm_url(args.base_domain)
    _log(f"judge endpoint: {llm_url}")

    params = ChunkerParams()
    _log(f"sparse-checkout {HANDBOOK_SUBDIR}@{HANDBOOK_COMMIT[:8]} -> {args.work_dir}")
    docs_dir = sparse_checkout(args.work_dir, commit=HANDBOOK_COMMIT)
    tracked = git_tracked_docs(args.work_dir, HANDBOOK_SUBDIR, params.file_extensions)
    collection = build_collection(docs_dir, params=params, files=tracked)
    collection_hashes = {c.hash for c in collection}
    _log(f"collection built: {len(collection_hashes)} chunks")

    source = bind_source(collection, args.base_domain)
    lexical = run_qa_pass(
        source, collection_hashes, search_mode="lexical", style_dist=_STYLE_KEYWORD,
        system_prompt=_LEXICAL_SYSTEM_PROMPT, n_samples=args.lexical_samples,
        out_dir=args.qa_dir / "qa_lexical", base_domain=args.base_domain,
        seed=args.seed, max_overlap=None, reuse=True,
    )
    vector = run_qa_pass(
        source, collection_hashes, search_mode="vector", style_dist=_STYLE_PARAPHRASE,
        system_prompt=_VECTOR_SYSTEM_PROMPT, n_samples=args.vector_samples,
        out_dir=args.qa_dir / "qa_vector", base_domain=args.base_domain,
        seed=args.seed + 1, max_overlap=_VECTOR_MAX_OVERLAP, reuse=True,
    )
    records = lexical + vector
    assert all(r.capability in NATURAL_CAPABILITIES for r in records)
    _log(f"natural rows to judge: {len(records)} (lexical={len(lexical)} vector={len(vector)})")

    by_hash = {c.hash: c for c in collection.chunks}
    done = _load_done(args.out)
    if done:
        _log(f"resume: {len(done)} verdicts already on disk")
    judge = QaValidityJudge(base_url=llm_url)

    rows: list[dict] = []
    judged = 0
    for i, rec in enumerate(records):
        key = pair_key(rec)
        if key in done:
            rows.append(done[key])
            continue
        gold_chunks = [by_hash[h] for h in rec.gold_chunk_hashes if h in by_hash]
        if not gold_chunks:
            v = Verdict("keep", "valid", "no gold content resolvable — kept", 0.0)
        else:
            v = judge.judge(rec.query, gold_chunks)
        rows.append(make_verdict_row(rec, gold_chunks, natural=True, verdict=v))
        judged += 1
        _write(args.out, rows)  # crash-consistent: full file after every judgment
        if judged % 10 == 0 or v.verdict == "drop":
            _log(f"judged {i + 1}/{len(records)} (new={judged}) last={v.verdict}/{v.reason}")
    _write(args.out, rows)
    if args.mirror:
        args.mirror.parent.mkdir(parents=True, exist_ok=True)
        _write(args.mirror, rows)

    tally: dict[str, int] = {}
    dropped_by_reason: dict[str, int] = {}
    for row in rows:
        tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1
        if row["verdict"] == "drop":
            dropped_by_reason[row["reason"]] = dropped_by_reason.get(row["reason"], 0) + 1
    n = len(rows)
    drops = tally.get("drop", 0)
    defect_rate = round(drops / max(n, 1), 4)
    summary = {
        "natural_judged": n,
        "verdict_tally": tally,
        "dropped_by_reason": dropped_by_reason,
        "defect_rate": defect_rate,
        "defect_ceiling": DEFECT_CEILING,
        "over_ceiling": defect_rate > DEFECT_CEILING,
        "judge_resolved_model": judge.resolved_model,
        "out": str(args.out),
    }
    _log("SUMMARY " + json.dumps(summary, sort_keys=True))
    (args.out.parent / "verdicts_v2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
