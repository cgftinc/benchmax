"""Code-first project specification for router-training workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from castform_router.training_environment import build_training_workspace


def load_project_spec(path: Path) -> dict[str, Any]:
    """Load and validate one versioned JSON project specification."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("project spec must be a JSON object")
    validate_project_spec(value)
    return value


def validate_project_spec(spec: dict[str, Any]) -> None:
    """Validate the stable, user-authored portion of a router project."""

    if spec.get("schema_version") != "1":
        raise ValueError("schema_version must be '1'")
    repositories = spec.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("repositories must be a non-empty array")
    if len(repositories) > 100:
        raise ValueError("repositories are limited to 100")

    auth_profiles = spec.get("auth_profiles", {})
    if not isinstance(auth_profiles, dict):
        raise ValueError("auth_profiles must be an object")
    for name, profile in auth_profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("auth profile names must be non-empty strings")
        if not isinstance(profile, dict):
            raise ValueError(f"auth profile {name} must be an object")

    for index, repository in enumerate(repositories):
        if not isinstance(repository, dict):
            raise ValueError(f"repository {index + 1} must be an object")
        repo = repository.get("repo") or repository.get("url")
        if not isinstance(repo, str) or not repo:
            raise ValueError(f"repository {index + 1} needs repo or url")
        auth_profile = repository.get("auth_profile")
        inline_auth = repository.get("auth")
        if auth_profile is not None and inline_auth is not None:
            raise ValueError(
                f"repository {index + 1} cannot set auth and auth_profile together"
            )
        if auth_profile is not None and auth_profile not in auth_profiles:
            raise ValueError(
                f"repository {index + 1} references unknown auth profile "
                f"{auth_profile}"
            )

    routes = spec.get("allowed_routes")
    if (
        not isinstance(routes, list)
        or len(routes) < 2
        or not all(isinstance(route, str) for route in routes)
    ):
        raise ValueError("allowed_routes must contain at least two route IDs")

    pull_requests = spec.get("pull_requests", {})
    if not isinstance(pull_requests, dict):
        raise ValueError("pull_requests must be an object")
    limit = pull_requests.get("limit_per_repo", 20)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("pull_requests.limit_per_repo must be between 1 and 20")
    eval_ratio = pull_requests.get("eval_ratio", 0.2)
    if (
        isinstance(eval_ratio, bool)
        or not isinstance(eval_ratio, (int, float))
        or not 0 <= float(eval_ratio) < 1
    ):
        raise ValueError("pull_requests.eval_ratio must be between 0 and 1")
    exclude_labels = pull_requests.get("exclude_labels", [])
    if not isinstance(exclude_labels, list) or not all(
        isinstance(label, str) for label in exclude_labels
    ):
        raise ValueError("pull_requests.exclude_labels must be an array of strings")

    benchmark = spec.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be an object")


def create_training_project(
    spec: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Generate a workspace from a validated project specification."""

    validate_project_spec(spec)
    auth_profiles = spec.get("auth_profiles", {})
    repositories = []
    for repository in spec["repositories"]:
        auth_profile = repository.get("auth_profile")
        auth = (
            auth_profiles[auth_profile]
            if auth_profile is not None
            else repository.get("auth")
        )
        repositories.append(
            {
                "full_name": repository.get("repo"),
                "html_url": repository.get("url"),
                "default_branch": repository.get("revision") or "main",
                "visibility": repository.get("visibility") or "unknown",
                "verification": "configured_code_first",
                "auth": auth,
            }
        )

    benchmark = spec.get("benchmark", {})
    pull_requests = spec.get("pull_requests", {})
    result = build_training_workspace(
        output_root,
        repositories=repositories,
        selected_route_ids=spec["allowed_routes"],
        tasks_per_repo=int(
            benchmark.get(
                "tasks_per_repo",
                pull_requests.get("limit_per_repo", 20),
            )
        ),
        repetitions=int(benchmark.get("repetitions", 1)),
        average_run_cost_usd=float(
            benchmark.get("average_run_cost_usd", 1.0)
        ),
        privacy_mode=str(
            benchmark.get("execution", "castform_hosted")
        ),
    )

    workspace = Path(result["workspace_path"])
    (workspace / "project.spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["files"] = sorted([*result["files"], "project.spec.json"])
    return result
