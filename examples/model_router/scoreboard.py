#!/usr/bin/env python3
"""The one policy table every router rung reports into.

Reads dataset.jsonl (from build_dataset.py), builds the (task, route)
outcome matrix, and prints solve rate / $ per policy:

  always-<route>   one row per route (always-frontier is the bar that matters)
  random           expected value of a uniformly random route
  oracle           ceiling: max p_hat per task, cheapest route among ties,
                   credited at that route's actual p_hat (never 1.0)
  router           optional, from --picks

The routable subset (tasks where routes disagree) is reported separately.

Temporal split: tasks are ordered by merged_at; the latest --test-frac are
the test split. --split {all,train,test} selects what is scored.

--picks: JSONL rows {"task_id": ..., "model": ..., "router_cost_usd": 0.0}.
Router $/task includes router_cost_usd (fully-loaded cost).

Usage:
  python scoreboard.py dataset.jsonl --split all
  python scoreboard.py dataset.jsonl --split test --picks picks.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_matrix(dataset: Path):
    """(task, route) -> (p_hat, mean_cost); plus task -> merged_at."""
    cells, dates = defaultdict(list), {}
    for line in dataset.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cells[(r["task_id"], r["route"])].append(r)
        dates[r["task_id"]] = r.get("merged_at") or ""
    routes = sorted({k[1] for k in cells})
    tasks = sorted({k[0] for k in cells})
    full = [t for t in tasks if all((t, r) in cells for r in routes)]
    if dropped := sorted(set(tasks) - set(full)):
        print(f"dropped {len(dropped)} tasks missing routes: {', '.join(dropped)}")
    matrix = {
        (t, r): (st.mean(x["reward"] for x in cells[(t, r)]),
                 st.mean(x["cost_usd"] or 0.0 for x in cells[(t, r)]))
        for t in full for r in routes
    }
    return matrix, full, routes, dates


def split_tasks(tasks, dates, test_frac):
    """Temporal split: latest test_frac of tasks (by merged_at) are test."""
    ordered = sorted(tasks, key=lambda t: dates.get(t) or "")
    n_test = max(1, round(len(ordered) * test_frac))
    return ordered[:-n_test], ordered[-n_test:]


def report(matrix, tasks, routes, picks, label):
    if not tasks:
        print(f"\n== {label} == (no tasks)")
        return
    print(f"\n== {label} ==  {len(tasks)} tasks x {len(routes)} routes")
    print(f"{'policy':26} {'solve':>7} {'$/task':>8} {'$ total':>9} {'Perf/$':>8}")

    def row(name, per_task):
        p = st.mean(q for q, _ in per_task)
        tot = sum(c for _, c in per_task)
        print(f"{name:26} {p*100:6.1f}% {tot/len(per_task):8.2f} {tot:9.2f} "
              f"{(p*100/tot if tot else float('nan')):8.2f}")

    for r in routes:
        row(f"always-{r}", [matrix[(t, r)] for t in tasks])
    row("random", [(st.mean(matrix[(t, r)][0] for r in routes),
                    st.mean(matrix[(t, r)][1] for r in routes)) for t in tasks])
    if picks:
        sel = [(t, picks[t]) for t in tasks if picks.get(t, {}).get("model") in routes]
        if sel:
            per_task = [(matrix[(t, m["model"])][0],
                         matrix[(t, m["model"])][1] + (m.get("router_cost_usd") or 0))
                        for t, m in ((t, picks[t]) for t, _ in sel)]
            row(f"router ({len(sel)}/{len(tasks)})", per_task)
    # Oracle ceiling: max p_hat, cheapest among ties, credited at actual p_hat.
    row("ORACLE (ceiling)", [
        min((matrix[(t, r)] for r in routes
             if matrix[(t, r)][0] == max(matrix[(t, x)][0] for x in routes)),
            key=lambda pc: pc[1])
        for t in tasks
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--picks", type=Path,
                    help="JSONL {task_id, model, router_cost_usd}")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    matrix, tasks, routes, dates = load_matrix(args.dataset)
    train, test = split_tasks(tasks, dates, args.test_frac)
    print(f"temporal split: {len(train)} train "
          f"(..{dates.get(train[-1], '?')[:10]}), {len(test)} test "
          f"({dates.get(test[0], '?')[:10]}..)")
    scored = {"all": tasks, "train": train, "test": test}[args.split]

    picks = None
    if args.picks:
        picks = {json.loads(l)["task_id"]: json.loads(l)
                 for l in args.picks.read_text().splitlines() if l.strip()}

    report(matrix, scored, routes, picks, f"{args.split} tasks")
    routable = [t for t in scored
                if len({matrix[(t, r)][0] >= 0.5 for r in routes}) > 1]
    report(matrix, routable, routes, picks, "routable subset (routes disagree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
