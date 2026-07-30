#!/usr/bin/env python3
"""Convert codeprobe-mined task dirs into harbor task dirs.

Usage: python convert_to_harbor.py <repo_dir> <out_dir> --repo-url <url>

For each task under <repo_dir>/.codeprobe/tasks/<id>/ emits:

  <out_dir>/<id>/
    instruction.md        codeprobe instruction minus the host Task Contract
    task.toml             timeouts + provenance metadata
    environment/Dockerfile  leak-guarded checkout of the base commit + deps
    solution/solve.sh     applies the original PR diff (oracle)
    solution/fix.patch
    tests/test.sh         overlays PR-era test files, runs verify command,
                          writes /logs/verifier/reward.txt (1.0 / 0.0)
    tests/overlay/...     PR-era snapshots of the tests the PR touched

Design notes (see PLAN.md):
- Base state = ground_truth_commit^ , fetched by SHA so no future history
  (or the fix itself) is reachable inside the sandbox.
- Codeprobe's own verifier never restores the PR's tests; we overlay them
  at verify time, SWE-bench test-patch style, so reward measures "did the
  agent implement the PR's behavior".
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

DOCKERFILE_TEMPLATE = """\
FROM python:3.12-slim

RUN apt-get update \\
    && apt-get install -y --no-install-recommends git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leak guard: fetch only the base commit's history (no refs to the fix).
RUN git init -q . \\
    && git remote add origin {repo_url} \\
    && git fetch -q --depth {fetch_depth} origin {base_sha} \\
    && git checkout -q FETCH_HEAD \\
    && git remote remove origin

RUN pip install --no-cache-dir -e . pytest
"""

TEST_SH_TEMPLATE = """\
#!/bin/bash
# Verifier: overlay the PR-era test files, run the mined verify command.
# Reward is binary: 1.0 iff the PR's own acceptance tests pass.
set -u
cd /app

# Overlay PR-era test files (agent never saw these).
if [ -d /tests/overlay ]; then
    (cd /tests/overlay && find . -type f) | while read -r f; do
        mkdir -p "/app/$(dirname "$f")"
        cp "/tests/overlay/$f" "/app/$f"
    done
fi
{delete_lines}
{verify_command}
exit_code=$?

mkdir -p /logs/verifier
if [ "$exit_code" -eq 0 ]; then
    echo 1.0 > /logs/verifier/reward.txt
else
    echo 0.0 > /logs/verifier/reward.txt
fi
echo "verify exit=$exit_code" >&2
exit 0
"""

SOLVE_SH = """\
#!/bin/bash
# Oracle: apply the original PR diff.
set -euo pipefail
cd /app
git apply /solution/fix.patch
"""

TASK_TOML_TEMPLATE = """\
version = "1.0"

[metadata]
source = "codeprobe"
repo = {repo!r}
repo_url = {repo_url!r}
task_id = {task_id!r}
difficulty = {difficulty!r}
quality_score = {quality}
ground_truth_commit = {gt!r}
base_commit = {base!r}
verify_command = {verify_command!r}

[verifier]
timeout_sec = 600.0
{verifier_network}

[agent]
timeout_sec = 1800.0
{agent_network}

[environment]
build_timeout_sec = 900.0
"""

# Agent egress allowlist: model APIs + agent-CLI install sources only.
# Blocks the upstream-fetch cheat (github, pypi, mirrors all unreachable).
AGENT_ALLOWLIST = """\
network_mode = "allowlist"
allowed_hosts = [
    "*.anthropic.com",
    "downloads.claude.ai",
    "api.openai.com",
    "auth.openai.com",
    "chatgpt.com",
    "*.chatgpt.com",
    "registry.npmjs.org",
    "*.npmjs.org",
    "deb.debian.org",
    "security.debian.org",
]"""

RULES_SECTION = """
## Rules

- Work only from the code in this repository checkout. Do not fetch the
  upstream repository, its history, releases, or any mirror of it (no
  `git fetch`/`clone`/`ls-remote`, no installing this project from a
  package index). Solutions derived from upstream code count as failure.
- Network access beyond your model API is restricted and monitored.
"""


def git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def sanitize_verify_command(command: str) -> str:
    """Drop non-Python file args from pytest commands.

    Codeprobe's test-file heuristic can classify docs (e.g. docs/testing.md)
    as test files; pytest exits 4 (usage error) on non-collectable args.
    """
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "pytest":
        return command
    kept = [t for t in tokens if not ("/" in t or "." in t) or t.endswith(".py")
            or t.startswith("-")]
    return shlex.join(kept)


def is_overlay_file(path: str) -> bool:
    """True for genuine test files only.

    Codeprobe's test_files heuristic classifies by filename, so source like
    src/click/testing.py (the fix itself) lands in it; overlaying that would
    hand the verifier the solution. Only overlay files living under a tests
    directory or named like pytest test modules.
    """
    parts = Path(path).parts
    name = parts[-1]
    if "src" in parts:
        return False
    if parts[0] in ("tests", "test", "testing"):
        return True
    return name == "conftest.py" or (
        name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
    )


def strip_task_contract(instruction: str) -> str:
    """Drop the host-specific '## Task Contract' section."""
    lines, out, skipping = instruction.splitlines(), [], False
    for line in lines:
        if line.startswith("## "):
            skipping = line.strip() == "## Task Contract"
        if not skipping:
            out.append(line)
    return "\n".join(out).strip() + "\n"


def convert_task(task_dir: Path, repo: Path, out_root: Path, repo_url: str,
                 fetch_depth: int, agent_network: str) -> str:
    meta = json.loads((task_dir / "metadata.json").read_text())
    gt_json = json.loads((task_dir / "tests" / "ground_truth.json").read_text())
    task_id = meta["id"]
    gt = meta["metadata"]["ground_truth_commit"]
    base = git(repo, "rev-parse", f"{gt}^").strip()
    verify_command = sanitize_verify_command(meta["verification"]["command"])

    out = out_root / task_id
    (out / "environment").mkdir(parents=True, exist_ok=True)
    (out / "solution").mkdir(exist_ok=True)
    (out / "tests" / "overlay").mkdir(parents=True, exist_ok=True)

    # instruction.md
    instruction = (task_dir / "instruction.md").read_text()
    (out / "instruction.md").write_text(
        strip_task_contract(instruction) + RULES_SECTION
    )

    # task.toml
    if agent_network == "allowlist":
        agent_net, verifier_net = AGENT_ALLOWLIST, 'network_mode = "no-network"'
    else:
        # Local docker on macOS cannot enforce phase network policies
        # (harbor rejects non-public modes); rely on the instruction rules
        # + audit_trajectories.py there instead.
        agent_net = verifier_net = 'network_mode = "public"'
    (out / "task.toml").write_text(TASK_TOML_TEMPLATE.format(
        agent_network=agent_net,
        verifier_network=verifier_net,
        repo=meta["repo"],
        repo_url=repo_url,
        task_id=task_id,
        difficulty=meta["metadata"]["difficulty"],
        quality=meta["metadata"]["quality_score"],
        gt=gt,
        base=base,
        verify_command=verify_command,
    ))

    # environment/Dockerfile
    (out / "environment" / "Dockerfile").write_text(DOCKERFILE_TEMPLATE.format(
        repo_url=repo_url, base_sha=base, fetch_depth=fetch_depth
    ))

    # solution: the PR diff
    patch = git(repo, "diff", "--binary", base, gt)
    (out / "solution" / "fix.patch").write_text(patch)
    solve = out / "solution" / "solve.sh"
    solve.write_text(SOLVE_SH)
    solve.chmod(0o755)

    # tests: overlay PR-era test files; delete ones the PR removed
    changed = git(repo, "diff", "--name-status", base, gt).splitlines()
    test_files = set(gt_json.get("test_files", []))
    deleted: list[str] = []
    for line in changed:
        status, *paths = line.split("\t")
        path = paths[-1]
        if path not in test_files or not is_overlay_file(path):
            continue
        if status.startswith("D"):
            deleted.append(path)
            continue
        dest = out / "tests" / "overlay" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            subprocess.run(
                ["git", "-C", str(repo), "show", f"{gt}:{path}"],
                capture_output=True, check=True,
            ).stdout
        )

    delete_lines = "\n".join(f"rm -f /app/{shlex.quote(p)}" for p in deleted)
    test_sh = out / "tests" / "test.sh"
    test_sh.write_text(TEST_SH_TEMPLATE.format(
        verify_command=verify_command, delete_lines=delete_lines
    ))
    test_sh.chmod(0o755)
    return task_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--fetch-depth", type=int, default=200)
    ap.add_argument("--agent-network", choices=["allowlist", "public"],
                    default="allowlist",
                    help="allowlist = enforced egress (Linux/Modal); public = "
                         "no enforcement (local macOS docker), instruction + "
                         "audit layers only")
    args = ap.parse_args()

    tasks_root = args.repo_dir / ".codeprobe" / "tasks"
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    if not task_dirs:
        print(f"no tasks under {tasks_root}", file=sys.stderr)
        return 1
    for task_dir in task_dirs:
        task_id = convert_task(
            task_dir, args.repo_dir, args.out_dir, args.repo_url,
            args.fetch_depth, args.agent_network,
        )
        print(f"converted {task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
