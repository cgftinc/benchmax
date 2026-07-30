#!/usr/bin/env python3
"""Filter converted harbor tasks by verifier discrimination.

Runs each task k times with the oracle agent (PR diff applied) and k times
with the nop agent (no action), then keeps only tasks whose verifier
discriminates: every oracle run scores 1.0 and every nop run scores 0.0.

Usage: python gate_tasks.py <tasks_dir> [-k 3] [--jobs-dir harbor_runs]

Writes <tasks_dir>/manifest.json:
  { "<task_id>": {"verdict": "pass|oracle_fail|nop_pass|flaky|error",
                   "oracle": [..rewards..], "nop": [..rewards..]}, ... }

Verdicts:
  pass         oracle all 1.0, nop all 0.0  -> trainable
  oracle_fail  oracle consistently != 1.0   -> conversion/env bug, fix converter
  nop_pass     nop consistently 1.0         -> verifier can't discriminate, drop
  flaky        rewards disagree across runs -> nondeterministic verifier, drop
  error        trial errored / reward missing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import shutil

_LOCAL_HARBOR = Path(__file__).parent / ".venv" / "bin" / "harbor"
HARBOR = _LOCAL_HARBOR if _LOCAL_HARBOR.exists() else Path(
    shutil.which("harbor") or "harbor"
)


def run_agent(tasks_dir: Path, agent: str, k: int, jobs_dir: Path,
              concurrency: int) -> dict[str, list[float]]:
    """Run one harbor job; return task_id -> list of rewards."""
    before = {p.name for p in jobs_dir.iterdir()} if jobs_dir.exists() else set()
    res = subprocess.run(
        [str(HARBOR), "run", "-p", str(tasks_dir), "-a", agent,
         "-k", str(k), "-n", str(concurrency), "--jobs-dir", str(jobs_dir)],
        capture_output=True, text=True,
    )
    new_jobs = sorted({p.name for p in jobs_dir.iterdir()} - before)
    if not new_jobs:
        sys.exit(f"harbor run -a {agent} produced no job dir:\n{res.stderr[-500:]}")
    job_dir = jobs_dir / new_jobs[-1]

    rewards: dict[str, list[float]] = defaultdict(list)
    stats = json.loads((job_dir / "result.json").read_text())["stats"]
    for eval_stats in stats["evals"].values():
        for reward_str, trials in eval_stats["reward_stats"]["reward"].items():
            for trial in trials:
                task_id = trial.rsplit("__", 1)[0]
                rewards[task_id].append(float(reward_str))
    return rewards


def verdict(oracle: list[float], nop: list[float], k: int) -> str:
    if len(oracle) < k or len(nop) < k:
        return "error"
    if len(set(oracle)) > 1 or len(set(nop)) > 1:
        return "flaky"
    if all(r == 1.0 for r in oracle) and all(r == 0.0 for r in nop):
        return "pass"
    if all(r == 1.0 for r in nop):
        return "nop_pass"
    return "oracle_fail"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_dir", type=Path)
    ap.add_argument("-k", type=int, default=3, help="runs per agent per task")
    ap.add_argument("--jobs-dir", type=Path, default=Path("harbor_runs"))
    ap.add_argument("-n", "--concurrency", type=int, default=4)
    args = ap.parse_args()

    task_ids = sorted(p.name for p in args.tasks_dir.iterdir()
                      if p.is_dir() and (p / "task.toml").exists())
    print(f"gating {len(task_ids)} tasks (k={args.k} per agent)")
    oracle = run_agent(args.tasks_dir, "oracle", args.k, args.jobs_dir,
                       args.concurrency)
    nop = run_agent(args.tasks_dir, "nop", args.k, args.jobs_dir,
                    args.concurrency)

    manifest = {
        tid: {
            "verdict": verdict(oracle.get(tid, []), nop.get(tid, []), args.k),
            "oracle": oracle.get(tid, []),
            "nop": nop.get(tid, []),
        }
        for tid in task_ids
    }
    out = args.tasks_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for entry in manifest.values():
        counts[entry["verdict"]] += 1
    for tid, entry in sorted(manifest.items()):
        if entry["verdict"] != "pass":
            print(f"  DROP {tid}: {entry['verdict']} "
                  f"(oracle={entry['oracle']} nop={entry['nop']})")
    print(f"manifest -> {out}")
    print("  " + ", ".join(f"{v}={n}" for v, n in sorted(counts.items())))
    return 0 if counts.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
