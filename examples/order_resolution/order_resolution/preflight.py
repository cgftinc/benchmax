"""Prepared-worktree and BenchMAX contract checks for implementation."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, get_type_hints

from benchmax.envs import BaseEnv, BaseRollout
from benchmax.envs.base.env import _prepare_example

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
BENCHMAX_ROOT = EXAMPLE_ROOT.parents[1]
SUPERPROJECT_ROOT = Path(
    subprocess.run(
        ["git", "-C", str(BENCHMAX_ROOT), "rev-parse", "--show-superproject-working-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
)
BENCHMAX_GITLINK = "core/benchmax"
SUPERPROJECT_ALLOWED_PATHS = (BENCHMAX_GITLINK,)
BENCHMAX_ALLOWED_PATHS = ("examples/order_resolution/", "uv.lock")
EXPECTED_BENCHMAX_VERSION_PREFIX = "0.2."
EXPECTED_BASE_ENV_MODULE = "benchmax.envs.base.env"


class PreflightError(RuntimeError):
    """A prepared-worktree or API-contract mismatch."""


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise PreflightError(f"git {' '.join(args)} failed in {root}: {detail}") from error
    return result.stdout.rstrip()


def _status_paths(root: Path) -> list[str]:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    paths: list[str] = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        paths.append(path)
    return paths


def _assert_allowed_status(root: Path, allowed: tuple[str, ...]) -> list[str]:
    paths = _status_paths(root)
    unexpected = [
        path
        for path in paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in allowed)
    ]
    if unexpected:
        raise PreflightError(f"unexpected changes in {root}: {', '.join(unexpected)}")
    return paths


def _gitlink_at(commit: str) -> str:
    entry = _git(SUPERPROJECT_ROOT, "ls-tree", commit, BENCHMAX_GITLINK)
    fields = entry.split()
    if len(fields) != 4 or fields[0] != "160000" or fields[1] != "commit":
        raise PreflightError(f"{BENCHMAX_GITLINK} is not a gitlink at {commit}")
    return fields[2]


def _benchmax_version() -> str:
    package_file = BENCHMAX_ROOT / "packages/benchmax/pyproject.toml"
    with package_file.open("rb") as source:
        version = tomllib.load(source)["project"]["version"]
    if not isinstance(version, str):
        raise PreflightError("BenchMAX project.version is not a string")
    if not version.startswith(EXPECTED_BENCHMAX_VERSION_PREFIX):
        raise PreflightError(
            f"BenchMAX {version} does not match audited {EXPECTED_BENCHMAX_VERSION_PREFIX}x"
        )
    return version


def _contract() -> dict[str, Any]:
    if BaseEnv.__module__ != EXPECTED_BASE_ENV_MODULE:
        raise PreflightError(
            f"BaseEnv resolved to {BaseEnv.__module__}, expected {EXPECTED_BASE_ENV_MODULE}"
        )
    source = Path(cast(str, inspect.getsourcefile(BaseEnv))).resolve()
    expected_source = (BENCHMAX_ROOT / "packages/benchmax/src/benchmax/envs/base/env.py").resolve()
    if source != expected_source:
        raise PreflightError(f"BaseEnv resolved to {source}, expected {expected_source}")

    messages, example_args = _prepare_example(
        {
            "prompt_messages": [{"role": "user", "content": "contract probe"}],
            "hidden": "opaque",
        }
    )
    if messages != [{"role": "user", "content": "contract probe"}]:
        raise PreflightError("BaseEnv no longer preserves explicit prompt_messages")
    if example_args != {"hidden": "opaque"}:
        raise PreflightError("BaseEnv no longer exposes non-prompt payload as example_args")

    required_coroutines = ("create_dataset", "list_tools", "compute_reward", "run_tool")
    for name in required_coroutines:
        if not inspect.iscoroutinefunction(getattr(BaseEnv, name)):
            raise PreflightError(f"BaseEnv.{name} is no longer async")
    reward_annotation = get_type_hints(BaseEnv.compute_reward).get("rollout")
    if reward_annotation is not BaseRollout:
        raise PreflightError("BaseEnv.compute_reward no longer receives BaseRollout")
    if not hasattr(BaseEnv, "max_turns") or not hasattr(BaseEnv, "max_tool_calls"):
        raise PreflightError("BaseEnv no longer owns rollout/tool limits")

    source_bytes = source.read_bytes()
    return {
        "module": BaseEnv.__module__,
        "source": str(source.relative_to(BENCHMAX_ROOT)),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "create_dataset": str(inspect.signature(BaseEnv.create_dataset)),
        "list_tools": str(inspect.signature(BaseEnv.list_tools)),
        "run_tool": str(inspect.signature(BaseEnv.run_tool)),
        "compute_reward": str(inspect.signature(BaseEnv.compute_reward)),
        "rollout_context": str(inspect.signature(BaseEnv.rollout_context)),
        "payload_contract": "prompt_messages are explicit; remaining payload is example_args",
        "limits_owner": "environment",
    }


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read existing manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"existing manifest {path} must contain a JSON object")
    return value


def _assert_manifest_stable(existing: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("schema_version", "python", "base_env_contract"):
        if existing.get(key) != current.get(key):
            raise PreflightError(f"existing manifest disagrees with current {key}")
    for key in ("superproject", "benchmax"):
        old = dict(existing.get(key, {}))
        new = dict(current.get(key, {}))
        old.pop("observed_changes", None)
        new.pop("observed_changes", None)
        if old != new:
            raise PreflightError(f"existing manifest disagrees with current {key}")


def run_preflight(manifest_path: Path) -> dict[str, Any]:
    """Verify the prepared implementation base and write its frozen manifest."""

    superproject_head = _git(SUPERPROJECT_ROOT, "rev-parse", "HEAD")
    origin_main = _git(SUPERPROJECT_ROOT, "rev-parse", "refs/remotes/origin/main")
    fork_point = _git(SUPERPROJECT_ROOT, "merge-base", "HEAD", origin_main)
    if fork_point != origin_main:
        raise PreflightError(
            "feature branch fork point does not match the fetched origin/main; re-planning required"
        )

    gitlink_sha = _gitlink_at(fork_point)
    benchmax_head = _git(BENCHMAX_ROOT, "rev-parse", "HEAD")
    if benchmax_head != gitlink_sha:
        raise PreflightError(
            f"initialized BenchMAX HEAD {benchmax_head} does not match gitlink {gitlink_sha}"
        )
    branch = _git(BENCHMAX_ROOT, "symbolic-ref", "--short", "HEAD")
    if branch == "main":
        raise PreflightError("create a BenchMAX feature branch before implementation")

    superproject_changes = _assert_allowed_status(SUPERPROJECT_ROOT, SUPERPROJECT_ALLOWED_PATHS)
    benchmax_changes = _assert_allowed_status(BENCHMAX_ROOT, BENCHMAX_ALLOWED_PATHS)
    created_at = datetime.now(UTC).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "superproject": {
            "head_sha": superproject_head,
            "origin_main_sha": origin_main,
            "fork_point_sha": fork_point,
            "benchmax_gitlink_sha": gitlink_sha,
            "allowed_paths": list(SUPERPROJECT_ALLOWED_PATHS),
            "observed_changes": superproject_changes,
        },
        "benchmax": {
            "head_sha": benchmax_head,
            "branch": branch,
            "version": _benchmax_version(),
            "allowed_paths": list(BENCHMAX_ALLOWED_PATHS),
            "observed_changes": benchmax_changes,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "base_env_contract": _contract(),
    }
    existing = _load_existing_manifest(manifest_path)
    if existing is not None:
        manifest["created_at"] = existing.get("created_at", created_at)
        _assert_manifest_stable(existing, manifest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
