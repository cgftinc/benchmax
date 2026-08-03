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
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

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
        reward_groups = eval_stats.get("reward_stats", {}).get("reward", {})
        for reward_str, trials in reward_groups.items():
            for trial in trials:
                task_id = trial.rsplit("__", 1)[0]
                rewards[task_id].append(float(reward_str))
    return rewards


def run_missing(
    tasks_dir: Path,
    agent: str,
    missing: dict[int, list[str]],
    jobs_dir: Path,
    concurrency: int,
) -> dict[str, list[float]]:
    """Run only the missing repetitions, grouped by required run count."""
    rewards: dict[str, list[float]] = defaultdict(list)
    for run_count, task_ids in sorted(missing.items()):
        if not task_ids:
            continue
        print(
            f"resuming {agent}: {len(task_ids)} tasks need "
            f"{run_count} additional run(s)"
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{tasks_dir.name}-{agent}-resume-",
            dir=tasks_dir.parent,
        ) as temporary_dir:
            subset = Path(temporary_dir)
            for task_id in task_ids:
                shutil.copytree(
                    tasks_dir / task_id,
                    subset / task_id,
                    symlinks=True,
                )
            resumed = run_agent(
                subset, agent, run_count, jobs_dir, concurrency
            )
        for task_id, values in resumed.items():
            rewards[task_id].extend(values)
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
    ap.add_argument(
        "--resume-errors",
        action="store_true",
        help=(
            "preserve completed rewards in manifest.json and rerun only "
            "the repetitions missing from error verdicts"
        ),
    )
    ap.add_argument(
        "--promote-to",
        type=Path,
        help="copy newly passing task directories into this farm directory",
    )
    args = ap.parse_args()

    task_ids = sorted(p.name for p in args.tasks_dir.iterdir()
                      if p.is_dir() and (p / "task.toml").exists())
    out = args.tasks_dir / "manifest.json"
    if args.resume_errors:
        if not out.exists():
            sys.exit(f"cannot resume without an existing manifest: {out}")
        previous = json.loads(out.read_text())
        unknown = sorted(set(previous) - set(task_ids))
        if unknown:
            sys.exit(
                "manifest contains tasks absent from the task directory: "
                + ", ".join(unknown)
            )
        accumulated = {
            task_id: {
                "oracle": list(previous.get(task_id, {}).get("oracle", [])),
                "nop": list(previous.get(task_id, {}).get("nop", [])),
            }
            for task_id in task_ids
        }
        retry_ids = [
            task_id for task_id in task_ids
            if previous.get(task_id, {}).get("verdict") == "error"
            or len(accumulated[task_id]["oracle"]) < args.k
            or len(accumulated[task_id]["nop"]) < args.k
        ]
        print(
            f"resuming {len(retry_ids)} error/incomplete tasks "
            f"(k={args.k} per agent)"
        )
        for agent in ("oracle", "nop"):
            missing: dict[int, list[str]] = defaultdict(list)
            for task_id in retry_ids:
                remaining = args.k - len(accumulated[task_id][agent])
                if remaining > 0:
                    missing[remaining].append(task_id)
            resumed = run_missing(
                args.tasks_dir,
                agent,
                missing,
                args.jobs_dir,
                args.concurrency,
            )
            for task_id, values in resumed.items():
                accumulated[task_id][agent].extend(values)
        oracle = {
            task_id: entry["oracle"]
            for task_id, entry in accumulated.items()
        }
        nop = {
            task_id: entry["nop"]
            for task_id, entry in accumulated.items()
        }
    else:
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

    if args.promote_to:
        args.promote_to.mkdir(parents=True, exist_ok=True)
        farm_manifest_path = args.promote_to / "manifest.json"
        farm_manifest = (
            json.loads(farm_manifest_path.read_text())
            if farm_manifest_path.exists()
            else {}
        )
        promoted = 0
        already_present = 0
        for tid, entry in manifest.items():
            if entry["verdict"] != "pass":
                continue
            farm_manifest[tid] = entry
            destination = args.promote_to / tid
            if destination.exists():
                already_present += 1
                continue
            shutil.copytree(args.tasks_dir / tid, destination)
            promoted += 1
        farm_manifest_path.write_text(
            json.dumps(dict(sorted(farm_manifest.items())), indent=2) + "\n"
        )
        print(
            f"promoted {promoted} tasks -> {args.promote_to} "
            f"({already_present} already present)"
        )
    return 0 if counts.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
