#!/usr/bin/env python3
"""P1: prompted router with a per-route profile built from collected traces.

ACRouter's "+stats" move over the zero-shot baseline: same router LLM, but
the model-pool description is generated mechanically from TRAIN-split
outcomes in dataset.jsonl (solve rate, cost, per-task wins/losses by title).
Never hand-written, never from the test split - otherwise we are the router.

Routes each TEST task; writes picks JSONL rows:
  {task_id, model, reasoning, router_model, router_cost_usd}

Usage:
  python profile_router.py dataset.jsonl [--router-model claude-sonnet-4-6]
      [--test-frac 0.2] [--out picks_profile.jsonl]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from baseline_router import build_task_prompt, route
from scoreboard import load_matrix, split_tasks, split_tasks_by_repo

PROFILE_HEADER = """\
You are a coding task router. Your objective is to maximize the \
performance-cost trade-off: choose the model that achieves the best quality \
for its cost on this task.

## Available Models - measured track record

The stats below are measured outcomes on earlier tasks from this same
repository (solve rate, mean cost per task, and named wins/losses). Trust
measured quirks over general reputation.
"""

PROFILE_FOOTER = """\
## Instructions

Analyze the task and choose the model that maximizes quality relative to
cost, using the track record above. Prefer cheaper models when quality is
comparable, but route to an expensive model when the record shows it is the
only reliable solver for this kind of task.

Respond with ONLY a JSON object:
{"model": "<model_name>", "reasoning": "<brief explanation>"}
"""


def task_titles(rows: list[dict]) -> dict[str, str]:
    titles = {}
    for r in rows:
        if r["task_id"] in titles:
            continue
        try:
            first = Path(r["task_dir"], "instruction.md").read_text().splitlines()[0]
            titles[r["task_id"]] = first.lstrip("# ").strip()
        except Exception:
            titles[r["task_id"]] = r["task_id"]
    return titles


def build_profile(matrix: dict, train: list[str], routes: list[str],
                  titles: dict[str, str]) -> str:
    """Render train-split stats into the router's model-pool section."""
    stats = {}
    for m in routes:
        cells = [matrix[(t, m)] for t in train]
        stats[m] = (st.mean(p for p, _ in cells), st.mean(c for _, c in cells))
    lines = [PROFILE_HEADER]
    for i, m in enumerate(sorted(routes, key=lambda m: -stats[m][1]), 1):
        p, c = stats[m]
        solved = [t for t in train if matrix[(t, m)][0] >= 0.5]
        unique = [t for t in solved
                  if all(matrix[(t, o)][0] < 0.5 for o in routes if o != m)]
        missed = [t for t in train if matrix[(t, m)][0] < 0.5
                  and sum(matrix[(t, o)][0] >= 0.5 for o in routes) >= len(routes) - 2]
        lines.append(f"{i}. **{m}**: solved {len(solved)}/{len(train)} "
                     f"({100*p:.0f}%), ${c:.2f}/task avg.")
        if unique:
            lines.append("   Only model to solve: "
                         + "; ".join(f'"{titles[t]}"' for t in unique))
        if missed:
            lines.append("   Failed (though most others solved): "
                         + "; ".join(f'"{titles[t]}"' for t in missed))
        lines.append("")
    lines.append(PROFILE_FOOTER)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--router-model", default="claude-sonnet-4-6")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-strategy",
                    choices=["global-temporal", "repo-temporal"],
                    default="global-temporal")
    ap.add_argument("--route-split", choices=["test", "train"], default="test",
                    help="train = answer-sheet diagnostic: the profile "
                         "contains these tasks' own outcomes")
    ap.add_argument("--out", type=Path, default=Path("picks_profile.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    matrix, tasks, routes, dates, repos = load_matrix(args.dataset)
    if args.split_strategy == "repo-temporal":
        train, test = split_tasks_by_repo(tasks, dates, repos)
    else:
        train, test = split_tasks(tasks, dates, args.test_frac)
    titles = task_titles(rows)
    task_dirs = {r["task_id"]: r["task_dir"] for r in rows}

    profile = build_profile(matrix, train, routes, titles)
    target = train if args.route_split == "train" else test
    print(f"profile from {len(train)} train tasks; "
          f"routing {len(target)} {args.route_split} tasks")

    picks = []
    with args.out.open("w") as out:
        for tid in target:
            try:
                pick = route(build_task_prompt(Path(task_dirs[tid])),
                             args.router_model,
                             system_prompt=profile, allowed=set(routes))
            except Exception as e:
                print(f"{tid}: ERROR {e}", file=sys.stderr)
                continue
            row = {"task_id": tid, "router_model": args.router_model, **pick}
            out.write(json.dumps(row) + "\n")
            picks.append(row)
            print(f"{tid}: {pick['model']:<26} {pick['reasoning'][:70]}")

    print(f"\n{len(picks)}/{len(target)} routed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
