#!/usr/bin/env python3
"""Flatten collected harbor trials into one dataset.jsonl for router work.

Walks <jobs_dir>/**/<task>__<suffix>/result.json, joins each trial with its
task's metadata (task.toml) and gate verdict (manifest.json), audits its
trajectory for cheat commands, and emits one JSON row per kept trial.

Kept:    clean real-agent trials with a reward, on gate-passed tasks,
         audit-clean. Agent timeouts count as benchmark outcomes.
Dropped: oracle/nop trials, agent infrastructure failures, non-gated tasks,
         tainted trials (all loudly).

Row: task_id, repo, task_dir, merged_at, difficulty, quality_score, route
(model), harness, reward, cost_usd, n_input_tokens, n_cache_tokens,
n_output_tokens, trial_dir.

merged_at is the ground-truth commit's date, resolved from a local clone
(--repo-root/<repo>); it orders tasks for the temporal split.

Usage:
  python build_dataset.py harbor_runs --tasks-dir harbor_tasks/click \
      --out dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

from audit_trajectories import REAL_AGENTS, audit_trial, patterns


def is_infrastructure_failure(result: dict) -> bool:
    """Whether an agent exception makes the verifier reward unusable.

    Harbor still invokes the verifier after many agent/API failures. That
    commonly produces reward=0 for an untouched checkout, which must not be
    interpreted as a model attempt. A timeout is the one exception: the
    benchmark deliberately gives agents a fixed execution budget, so the
    resulting workspace remains a valid outcome.
    """
    exception_type = ((result.get("exception_info") or {})
                      .get("exception_type"))
    return bool(exception_type and exception_type != "AgentTimeoutError")


def load_tasks(tasks_dirs: list[Path]) -> tuple[dict, dict]:
    """task_id -> metadata dict, task_id -> gate verdict."""
    meta, verdicts = {}, {}
    for tdir in tasks_dirs:
        manifest = {}
        mf = tdir / "manifest.json"
        if mf.exists():
            manifest = json.loads(mf.read_text())
        for task in sorted(p for p in tdir.iterdir() if p.is_dir()):
            toml = task / "task.toml"
            if not toml.exists():
                continue
            m = tomllib.load(toml.open("rb")).get("metadata", {})
            m["task_dir"] = str(task)
            meta[task.name] = m
            verdicts[task.name] = manifest.get(task.name, {}).get("verdict", "ungated")
    return meta, verdicts


def commit_date(repo_dir: Path, commit: str, cache: dict) -> str | None:
    if commit in cache:
        return cache[commit]
    res = subprocess.run(["git", "-C", str(repo_dir), "show", "-s",
                          "--format=%cI", commit],
                         capture_output=True, text=True)
    cache[commit] = res.stdout.strip() or None if res.returncode == 0 else None
    return cache[commit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_dirs", nargs="+", type=Path)
    ap.add_argument("--tasks-dir", action="append", required=True, type=Path,
                    dest="tasks_dirs")
    ap.add_argument("--out", type=Path, default=Path("dataset.jsonl"))
    ap.add_argument("--repo-root", type=Path, default=Path("."),
                    help="dir holding local repo clones, for merged_at dates")
    ap.add_argument("--require-route", action="append", default=[],
                    help="retain only tasks with a kept trial for every named "
                         "route; repeat for a rectangular router dataset")
    args = ap.parse_args()

    meta, verdicts = load_tasks(args.tasks_dirs)
    dates: dict[str, str | None] = {}
    rows, drops = [], Counter()

    for jobs_dir in args.jobs_dirs:
        for result in sorted(jobs_dir.rglob("result.json")):
            trial_dir = result.parent
            if not (trial_dir / "agent").is_dir():
                continue  # job-level result.json, not a trial
            try:
                d = json.loads(result.read_text())
            except Exception:
                drops["unreadable"] += 1
                continue
            harness = (d.get("agent_info") or {}).get("name")
            if harness not in REAL_AGENTS:
                continue
            route = ((d.get("config") or {}).get("agent") or {}).get("model_name")
            reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            task_id = trial_dir.name.split("__")[0]
            if is_infrastructure_failure(d):
                exception_type = (d["exception_info"].get("exception_type")
                                  or "unknown")
                drops[f"agent-infra:{exception_type}"] += 1
                continue
            if reward is None or not route:
                drops["no-reward-or-model"] += 1
                continue
            if task_id not in meta:
                drops["unknown-task"] += 1
                continue
            if verdicts[task_id] != "pass":
                drops[f"gate:{verdicts[task_id]}"] += 1
                continue
            m = meta[task_id]
            findings = audit_trial(trial_dir, patterns(m.get("repo")))
            if findings:
                drops["audit-tainted"] += 1
                print(f"DROP tainted {trial_dir}: {findings[0]}", file=sys.stderr)
                continue
            ar = d.get("agent_result") or {}
            rows.append({
                "task_id": task_id,
                "repo": m.get("repo"),
                "task_dir": m.get("task_dir"),
                "merged_at": commit_date(args.repo_root / str(m.get("repo")),
                                         m.get("ground_truth_commit", ""), dates),
                "difficulty": m.get("difficulty"),
                "quality_score": m.get("quality_score"),
                "route": route,
                "harness": harness,
                "reward": float(reward),
                "cost_usd": ar.get("cost_usd"),
                "n_input_tokens": ar.get("n_input_tokens"),
                "n_cache_tokens": ar.get("n_cache_tokens"),
                "n_output_tokens": ar.get("n_output_tokens"),
                "trial_dir": str(trial_dir),
            })

    required_routes = set(args.require_route)
    if required_routes:
        observed_routes = {r["route"] for r in rows}
        if missing := sorted(required_routes - observed_routes):
            ap.error("required route has no kept trials: " + ", ".join(missing))
        routes_by_task: dict[str, set[str]] = {}
        for row in rows:
            routes_by_task.setdefault(row["task_id"], set()).add(row["route"])
        complete_tasks = {
            task_id for task_id, routes in routes_by_task.items()
            if required_routes <= routes
        }
        before = len(rows)
        rows = [r for r in rows if r["task_id"] in complete_tasks]
        print(f"required-route filter retained {len(complete_tasks)} tasks "
              f"and {len(rows)}/{before} trials")

    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_tasks = len({r["task_id"] for r in rows})
    n_routes = len({r["route"] for r in rows})
    print(f"{len(rows)} trials -> {args.out}  ({n_tasks} tasks, {n_routes} routes)")
    for k, v in sorted(drops.items()):
        print(f"  dropped {v}: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
