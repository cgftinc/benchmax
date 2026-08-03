#!/usr/bin/env python3
"""P2: kNN router - retrieve similar train tasks, copy what worked.

Baseline: does a trivial statistical method already route well? TF-IDF
cosine over instruction text (deliberately lexical: the dumb baseline
should be dumb; the similarity function is the seam to swap for a neural
embedding later). For each test task: find the k nearest train tasks,
average their per-route outcomes, pick the route with the best estimate
(ties -> cheapest on train). No LLM, no cost.

At small N this degenerates toward global per-route base rates - that is
the point of the rung, not a bug.

Writes picks JSONL rows: {task_id, model, reasoning, router_cost_usd: 0}.

Usage:
  python knn_router.py dataset.jsonl [-k 3] [--test-frac 0.2]
      [--out picks_knn.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

from scoreboard import load_matrix, split_tasks, split_tasks_by_repo

_WORD = re.compile(r"[a-z][a-z0-9_.-]{2,}")


def tfidf_vectors(docs: dict[str, str]) -> dict[str, dict[str, float]]:
    tfs = {tid: Counter(_WORD.findall(text.lower())) for tid, text in docs.items()}
    df = Counter(w for tf in tfs.values() for w in tf)
    n = len(docs)
    vecs = {}
    for tid, tf in tfs.items():
        v = {w: c * math.log(n / df[w]) for w, c in tf.items() if df[w] < n}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[tid] = {w: x / norm for w, x in v.items()}
    return vecs


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("-k", type=int, default=3, help="neighbours")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-strategy",
                    choices=["global-temporal", "repo-temporal"],
                    default="global-temporal")
    ap.add_argument("--out", type=Path, default=Path("picks_knn.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    matrix, tasks, routes, dates, repos = load_matrix(args.dataset)
    if args.split_strategy == "repo-temporal":
        train, test = split_tasks_by_repo(tasks, dates, repos)
    else:
        train, test = split_tasks(tasks, dates, args.test_frac)

    task_dirs = {r["task_id"]: r["task_dir"] for r in rows}
    docs = {t: Path(task_dirs[t], "instruction.md").read_text() for t in tasks}
    vecs = tfidf_vectors(docs)
    train_cost = {m: st.mean(matrix[(t, m)][1] for t in train) for m in routes}

    with args.out.open("w") as out:
        for tid in test:
            nn = sorted(train, key=lambda t: -cosine(vecs[tid], vecs[t]))[:args.k]
            est = {m: st.mean(matrix[(t, m)][0] for t in nn) for m in routes}
            best = max(est.values())
            pick = min((m for m in routes if est[m] == best),
                       key=lambda m: train_cost[m])
            row = {
                "task_id": tid, "model": pick, "router_cost_usd": 0.0,
                "reasoning": f"kNN k={args.k}: est p={best:.2f} from "
                             + ", ".join(f"{t}({cosine(vecs[tid], vecs[t]):.2f})"
                                         for t in nn),
            }
            out.write(json.dumps(row) + "\n")
            print(f"{tid}: {pick:<26} {row['reasoning'][:70]}")

    print(f"\n{len(test)} routed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
