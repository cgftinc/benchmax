#!/usr/bin/env python3
"""Emit the farm tasks a model still needs: no completed clean trial yet.

A (task, model) pair is DONE when any trial result.json under
harbor_runs/farm-<model>/ has a non-None reward and no infrastructure
exception. AgentTimeoutError is a real outcome under the benchmark's time
budget; other agent exceptions (auth, rate limit, network, cancellation,
setup, etc.) are infrastructure even if Harbor runs the verifier afterward
and assigns the untouched checkout a numeric 0. Pairs with >= --max-attempts
trials but no clean result are retired so the fill loop terminates. Prints one
task id per line.

Usage: python fill_missing.py <model> [--farm harbor_tasks/farm]
       [--max-attempts 3]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path


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


def model_result_paths(runs_dir: Path, model: str):
    """Yield trial results from base and suffixed farm runs for a model."""
    prefix = f"farm-{model}"
    if not runs_dir.exists():
        return
    for root in sorted(runs_dir.iterdir()):
        if not root.is_dir() or not (
            root.name == prefix or root.name.startswith(prefix + "-")
        ):
            continue
        for result in root.rglob("result.json"):
            if "__" in result.parent.name:
                yield result


def coverage_state(
    model: str,
    *,
    farm: Path,
    runs_dir: Path,
    max_attempts: int,
) -> tuple[list[str], set[str], set[str], Counter[str]]:
    """Return missing, done, retired task IDs and attempt counts."""
    tasks = sorted(
        path.name for path in farm.iterdir()
        if path.is_dir() and (path / "task.toml").exists()
    )
    task_set = set(tasks)
    attempts: Counter[str] = Counter()
    done: set[str] = set()
    for result_path in model_result_paths(runs_dir, model):
        task = result_path.parent.name.split("__", 1)[0]
        if task not in task_set:
            continue
        attempts[task] += 1
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        reward = (
            ((result.get("verifier_result") or {}).get("rewards") or {})
            .get("reward")
        )
        if (
            isinstance(reward, (int, float))
            and not isinstance(reward, bool)
            and not is_infrastructure_failure(result)
        ):
            done.add(task)

    retired = {
        task for task in tasks
        if task not in done and attempts[task] >= max_attempts
    }
    missing = [
        task for task in tasks
        if task not in done and task not in retired
    ]
    return missing, done, retired, attempts


def materialize_fill_set(farm: Path, target: Path, task_ids: list[str]) -> None:
    """Atomically replace a model fill directory with the exact missing set."""
    if target.resolve() == farm.resolve():
        raise ValueError("fill target must not be the farm itself")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for task_id in task_ids:
            source = farm / task_id
            if not (source / "task.toml").exists():
                raise ValueError(f"farm task is missing task.toml: {source}")
            shutil.copytree(source, staging / task_id, symlinks=True)
        if target.exists():
            if not target.is_dir():
                raise ValueError(
                    f"fill target exists and is not a directory: {target}"
                )
            shutil.rmtree(target)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--farm", type=Path, default=Path("harbor_tasks/farm"))
    ap.add_argument("--runs-dir", type=Path, default=Path("harbor_runs"))
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument(
        "--materialize",
        type=Path,
        help="atomically build this Harbor task directory from missing tasks",
    )
    args = ap.parse_args()
    if args.max_attempts < 1:
        ap.error("--max-attempts must be positive")

    missing, done, retired, _ = coverage_state(
        args.model,
        farm=args.farm,
        runs_dir=args.runs_dir,
        max_attempts=args.max_attempts,
    )
    if args.materialize:
        materialize_fill_set(args.farm, args.materialize, missing)
        print(
            f"materialized {len(missing)} tasks -> {args.materialize} "
            f"({len(done)} done, {len(retired)} retired)",
            file=sys.stderr,
        )
    for task in missing:
        print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
