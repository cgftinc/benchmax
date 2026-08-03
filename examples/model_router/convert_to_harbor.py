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
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROFILES_DIR = Path(__file__).with_name("environment_profiles")
DEFAULT_BASE_IMAGE = "python:3.12-slim"
DEFAULT_INSTALL_CMD = "pip install --no-cache-dir -e . pytest"

DOCKERFILE_TEMPLATE = """\
FROM {base_image}

RUN apt-get update \\
    && apt-get install -y --no-install-recommends git ca-certificates{apt_packages} \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leak guard: fetch only the base commit's history (no refs to the fix).
# Submodules are separate upstream repos (build tooling, vendored libs),
# not this repo's history, so initializing them leaks nothing.
RUN git init -q . \\
    && git remote add origin {repo_url} \\
    && git fetch -q --depth {fetch_depth} origin {base_sha} \\
    && git checkout -q FETCH_HEAD \\
    && git submodule update --init --quiet \\
    && git remote remove origin

RUN {install_cmd}{post_install_line}
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
{profile_line}
difficulty = {difficulty!r}
quality_score = {quality}
ground_truth_commit = {gt!r}
base_commit = {base!r}
verify_command = {verify_command!r}

[verifier]
timeout_sec = {verifier_timeout}
{verifier_network}

[agent]
timeout_sec = {agent_timeout}
{agent_network}

[environment]
build_timeout_sec = {build_timeout}
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


@dataclass(frozen=True)
class EnvironmentProfile:
    """Declarative repository build/test environment configuration."""

    name: str
    repo: str
    repo_url: str
    base_image: str
    apt_packages: tuple[str, ...]
    install_cmd: str
    post_install_cmd: str
    fetch_depth: int
    build_timeout_sec: float
    agent_timeout_sec: float
    verifier_timeout_sec: float
    path: Path


def _profile_path(reference: str | Path,
                  profiles_dir: Path = DEFAULT_PROFILES_DIR) -> Path:
    """Resolve either a profile name or an explicit TOML path."""
    candidate = Path(reference)
    if candidate.suffix == ".toml" or candidate.parent != Path("."):
        return candidate
    return profiles_dir / f"{candidate.name}.toml"


def load_environment_profile(
    reference: str | Path,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> EnvironmentProfile:
    """Load and strictly validate an environment profile."""
    path = _profile_path(reference, profiles_dir)
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"environment profile not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid environment profile TOML {path}: {exc}") from exc

    allowed = {
        "schema_version", "name", "repo", "repo_url", "base_image",
        "apt_packages", "install_cmd", "post_install_cmd", "fetch_depth",
        "timeouts",
    }
    if unknown := sorted(set(data) - allowed):
        raise ValueError(f"{path}: unknown profile keys: {', '.join(unknown)}")
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")

    required_strings = ("name", "repo", "repo_url", "base_image", "install_cmd")
    for key in required_strings:
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{path}: {key} must be a non-empty string")
    if path.stem != data["name"]:
        raise ValueError(
            f"{path}: profile name {data['name']!r} must match filename {path.stem!r}"
        )

    apt_packages = data.get("apt_packages", [])
    if (not isinstance(apt_packages, list)
            or not all(isinstance(item, str) and item for item in apt_packages)):
        raise ValueError(f"{path}: apt_packages must be an array of non-empty strings")
    post_install_cmd = data.get("post_install_cmd", "")
    if not isinstance(post_install_cmd, str):
        raise ValueError(f"{path}: post_install_cmd must be a string")
    fetch_depth = data.get("fetch_depth", 200)
    if not isinstance(fetch_depth, int) or isinstance(fetch_depth, bool) or fetch_depth < 1:
        raise ValueError(f"{path}: fetch_depth must be a positive integer")

    timeouts = data.get("timeouts", {})
    if not isinstance(timeouts, dict):
        raise ValueError(f"{path}: timeouts must be a table")
    allowed_timeouts = {"build_sec", "agent_sec", "verifier_sec"}
    if unknown := sorted(set(timeouts) - allowed_timeouts):
        raise ValueError(f"{path}: unknown timeout keys: {', '.join(unknown)}")

    def timeout(name: str, default: float) -> float:
        value = timeouts.get(name, default)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or value <= 0):
            raise ValueError(f"{path}: timeouts.{name} must be positive")
        return float(value)

    return EnvironmentProfile(
        name=data["name"],
        repo=data["repo"],
        repo_url=data["repo_url"],
        base_image=data["base_image"],
        apt_packages=tuple(apt_packages),
        install_cmd=data["install_cmd"],
        post_install_cmd=post_install_cmd,
        fetch_depth=fetch_depth,
        build_timeout_sec=timeout("build_sec", 900.0),
        agent_timeout_sec=timeout("agent_sec", 1800.0),
        verifier_timeout_sec=timeout("verifier_sec", 600.0),
        path=path,
    )


def git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def file_at_commit(repo: Path, commit: str, path: str) -> bytes | None:
    res = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        capture_output=True,
    )
    return res.stdout if res.returncode == 0 else None


def locked_dependency_version(repo: Path, commit: str, name: str) -> str | None:
    """Resolve an exact dependency version from the historical checkout.

    Prefer uv.lock, then exact pins in requirements files. This lets one
    conversion batch use each task's own test-tool era rather than whatever
    PyPI serves when the benchmark is built.
    """
    if raw := file_at_commit(repo, commit, "uv.lock"):
        try:
            lock = tomllib.loads(raw.decode())
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            lock = {}
        for package in lock.get("package", []):
            if package.get("name", "").lower() == name.lower():
                return package.get("version")

    paths = git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
    requirements = sorted(
        (p for p in paths
         if "requirement" in p.lower() and p.endswith((".txt", ".in"))),
        key=lambda p: ("emscripten" in p.lower(), "test" not in p.lower(), p),
    )
    for path in requirements:
        raw = file_at_commit(repo, commit, path)
        if not raw:
            continue
        for line in raw.decode(errors="replace").splitlines():
            match = re.match(
                rf"\s*{re.escape(name)}\s*==\s*([^;\s]+)", line, re.I
            )
            if match:
                return match.group(1)
    return None


def sanitize_verify_command(command: str) -> str:
    """Drop pytest positional paths that are not collectable test modules.

    Codeprobe's test-file heuristic can classify source, docs, golden-output
    fixtures, and example scripts as tests. They may need to be overlaid for
    a real test runner, but passing them directly to pytest causes collection
    warnings/errors or duplicate-module failures.
    """
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "pytest":
        return command
    kept = [tokens[0]]
    for token in tokens[1:]:
        if token.startswith("-"):
            kept.append(token)
            continue
        path = token.split("::", 1)[0]
        looks_like_path = "/" in path or bool(Path(path).suffix)
        if not looks_like_path or is_runnable_test_file(path):
            kept.append(token)
    return shlex.join(kept)


def is_runnable_test_file(path: str) -> bool:
    """Whether a Python path should be passed directly to pytest.

    Most projects use test_*.py/*_test.py. Pytest itself additionally
    collects testing/python/*.py, while its example_scripts tree is fixture
    data consumed by pytester and must not be collected directly.
    """
    parts = Path(path).parts
    if not parts or Path(path).suffix != ".py":
        return False
    name = parts[-1]
    if name in {"__init__.py", "conftest.py"} or "src" in parts:
        return False
    if parts[0] == "testing":
        return not {"example_scripts", "plugins_integration"}.intersection(parts)
    return name.startswith("test_") or name.endswith("_test.py")


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
                 fetch_depth: int, agent_network: str,
                 apt_packages: str, install_cmd: str,
                 post_install_cmd: str, *, base_image: str = DEFAULT_BASE_IMAGE,
                 profile_name: str = "", expected_repo: str = "",
                 build_timeout_sec: float = 900.0,
                 agent_timeout_sec: float = 1800.0,
                 verifier_timeout_sec: float = 600.0) -> str:
    meta = json.loads((task_dir / "metadata.json").read_text())
    gt_json = json.loads((task_dir / "tests" / "ground_truth.json").read_text())
    if expected_repo and meta.get("repo") != expected_repo:
        raise ValueError(
            f"{task_dir.name}: profile {profile_name!r} expects repo "
            f"{expected_repo!r}, task reports {meta.get('repo')!r}"
        )
    task_id = meta["id"]
    gt = meta["metadata"]["ground_truth_commit"]
    base = git(repo, "rev-parse", f"{gt}^").strip()
    verify_command = sanitize_verify_command(meta["verification"]["command"])
    if "{pytest_version}" in install_cmd or "{pytest_version}" in post_install_cmd:
        pytest_version = locked_dependency_version(repo, base, "pytest")
        if not pytest_version:
            raise RuntimeError(
                f"{task_id}: no exact pytest version in {base}'s lock/requirements"
            )
        install_cmd = install_cmd.replace("{pytest_version}", pytest_version)
        post_install_cmd = post_install_cmd.replace(
            "{pytest_version}", pytest_version
        )

    out = out_root / task_id
    (out / "environment").mkdir(parents=True, exist_ok=True)
    (out / "solution").mkdir(exist_ok=True)
    (out / "tests" / "overlay").mkdir(parents=True, exist_ok=True)

    # instruction.md
    instruction = (task_dir / "instruction.md").read_text()
    (out / "instruction.md").write_text(
        strip_task_contract(instruction) + RULES_SECTION
    )

    # environment/Dockerfile
    (out / "environment" / "Dockerfile").write_text(DOCKERFILE_TEMPLATE.format(
        base_image=base_image, repo_url=repo_url, base_sha=base,
        fetch_depth=fetch_depth,
        apt_packages=f" {apt_packages}" if apt_packages else "",
        install_cmd=install_cmd,
        post_install_line=(f"\nRUN {post_install_cmd}"
                           if post_install_cmd else ""),
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

    # No overlaid tests and no test deletions = the verifier would measure
    # only the pre-PR suite; reward cannot reflect the PR's behavior. Skip.
    overlaid = sorted(
        str(p.relative_to(out / "tests" / "overlay"))
        for p in (out / "tests" / "overlay").rglob("*.py")
    )
    if not overlaid and not deleted:
        shutil.rmtree(out)
        return f"SKIP {task_id} (no verifiable PR tests)"

    # A bare `pytest` (all file args sanitized away) would run the full
    # suite: slow, flaky, and unrelated to the PR. Point it at the PR's own
    # overlaid test files instead - exactly the reward definition.
    if verify_command == "pytest":
        runnable_overlaid = [p for p in overlaid if is_runnable_test_file(p)]
        if not runnable_overlaid:
            shutil.rmtree(out)
            return f"SKIP {task_id} (no collectable PR test modules)"
        verify_command = shlex.join(["pytest", *runnable_overlaid])

    delete_lines = "\n".join(f"rm -f /app/{shlex.quote(p)}" for p in deleted)
    test_sh = out / "tests" / "test.sh"
    test_sh.write_text(TEST_SH_TEMPLATE.format(
        verify_command=verify_command, delete_lines=delete_lines
    ))
    test_sh.chmod(0o755)

    # task.toml (last: verify_command is final only after the overlay fix-up)
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
        profile_line=(f"environment_profile = {profile_name!r}"
                      if profile_name else ""),
        difficulty=meta["metadata"]["difficulty"],
        quality=meta["metadata"]["quality_score"],
        gt=gt,
        base=base,
        verify_command=verify_command,
        build_timeout=build_timeout_sec,
        agent_timeout=agent_timeout_sec,
        verifier_timeout=verifier_timeout_sec,
    ))
    return task_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument(
        "--profile",
        help="environment profile name (for example numpy) or TOML path",
    )
    ap.add_argument("--repo-url")
    ap.add_argument("--base-image")
    ap.add_argument("--fetch-depth", type=int)
    ap.add_argument("--agent-network", choices=["allowlist", "public"],
                    default="allowlist",
                    help="allowlist = enforced egress (Linux/Modal); public = "
                         "no enforcement (local macOS docker), instruction + "
                         "audit layers only")
    ap.add_argument("--apt", default=None,
                    help="extra apt packages, e.g. 'build-essential' for "
                         "repos with compiled extensions")
    ap.add_argument("--install-cmd",
                    default=None,
                    help="full install command for the Dockerfile RUN line; "
                         "override for repos needing test extras, build env "
                         "vars, or multi-step installs (&&-joined); literal "
                         "{pytest_version} resolves from each base commit's "
                         "uv.lock or exact requirements pin")
    ap.add_argument("--post-install-cmd", default=None,
                    help="optional second Docker RUN, useful for cheap "
                         "commit-era pin overrides while retaining the main "
                         "environment layer cache")
    ap.add_argument("--task-id", action="append", default=[],
                    help="convert only this mined task ID; repeat for subsets")
    args = ap.parse_args()

    try:
        profile = load_environment_profile(args.profile) if args.profile else None
    except ValueError as exc:
        ap.error(str(exc))
    if not profile and not args.repo_url:
        ap.error("--repo-url is required unless --profile supplies it")

    repo_url = args.repo_url or profile.repo_url
    base_image = args.base_image or (profile.base_image if profile else DEFAULT_BASE_IMAGE)
    fetch_depth = (args.fetch_depth if args.fetch_depth is not None
                   else profile.fetch_depth if profile else 200)
    apt_packages = (args.apt if args.apt is not None
                    else shlex.join(profile.apt_packages) if profile else "")
    install_cmd = (args.install_cmd if args.install_cmd is not None
                   else profile.install_cmd if profile else DEFAULT_INSTALL_CMD)
    post_install_cmd = (
        args.post_install_cmd if args.post_install_cmd is not None
        else profile.post_install_cmd if profile else ""
    )

    tasks_root = args.repo_dir / ".codeprobe" / "tasks"
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    if args.task_id:
        wanted = set(args.task_id)
        task_dirs = [p for p in task_dirs if p.name in wanted]
        if missing := sorted(wanted - {p.name for p in task_dirs}):
            print("unknown task IDs: " + ", ".join(missing), file=sys.stderr)
            return 1
    if not task_dirs:
        print(f"no tasks under {tasks_root}", file=sys.stderr)
        return 1
    for task_dir in task_dirs:
        task_id = convert_task(
            task_dir, args.repo_dir, args.out_dir, repo_url,
            fetch_depth, args.agent_network, apt_packages, install_cmd,
            post_install_cmd,
            base_image=base_image,
            profile_name=profile.name if profile else "",
            expected_repo=profile.repo if profile else "",
            build_timeout_sec=profile.build_timeout_sec if profile else 900.0,
            agent_timeout_sec=profile.agent_timeout_sec if profile else 1800.0,
            verifier_timeout_sec=profile.verifier_timeout_sec if profile else 600.0,
        )
        print(f"converted {task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
