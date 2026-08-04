"""Mine merged GitHub pull requests into reviewable Benchmax tasks."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

RequestJson = Callable[[str, dict[str, str]], Any]


def materialize_pull_request_tasks(
    workspace: Path,
    *,
    settings: dict[str, Any] | None = None,
    request_json: RequestJson | None = None,
) -> dict[str, Any]:
    """Fetch merged PRs and populate temporal train/eval task files."""

    settings = settings or {}
    request_json = request_json or _request_json
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("workspace manifest has no repositories")

    limit_per_repo = int(settings.get("limit_per_repo", 20))
    eval_ratio = float(settings.get("eval_ratio", 0.2))
    include_body = bool(settings.get("include_body", False))
    exclude_labels = {
        str(label).lower() for label in settings.get("exclude_labels", [])
    }

    tasks = []
    repository_counts = {}
    for repository in repositories:
        full_name = repository["full_name"]
        token = _repository_token(repository.get("auth", {}))
        pull_requests = _fetch_merged_pull_requests(
            full_name,
            limit=limit_per_repo,
            token=token,
            request_json=request_json,
        )
        repository_tasks = [
            task
            for pull_request in pull_requests
            if (
                task := _pull_request_task(
                    full_name,
                    pull_request,
                    include_body=include_body,
                    exclude_labels=exclude_labels,
                )
            )
            is not None
        ]
        repository_counts[full_name] = len(repository_tasks)
        tasks.extend(repository_tasks)

    if not tasks:
        raise ValueError("no eligible merged pull requests were found")
    tasks.sort(key=lambda task: (task["source"]["merged_at"], task["task_id"]))
    eval_count = _evaluation_count(len(tasks), eval_ratio)
    split_at = len(tasks) - eval_count
    train_tasks = tasks[:split_at]
    eval_tasks = tasks[split_at:]

    _write_jsonl(workspace / "benchmax" / "tasks" / "train.jsonl", train_tasks)
    _write_jsonl(workspace / "benchmax" / "tasks" / "eval.jsonl", eval_tasks)

    benchmark = manifest["benchmark"]
    benchmark["planned_tasks"] = len(tasks)
    benchmark["planned_rollouts"] = (
        len(tasks)
        * len(manifest["candidate_routes"])
        * int(benchmark["repetitions"])
    )
    benchmark["estimated_evaluation_cost_usd"] = round(
        benchmark["planned_rollouts"]
        * float(benchmark["average_run_cost_usd"]),
        2,
    )
    manifest["status"] = "awaiting_verifier_review"
    manifest["pull_request_mining"] = {
        "source": "github_merged_pull_requests",
        "include_body": include_body,
        "exclude_labels": sorted(exclude_labels),
        "repository_counts": repository_counts,
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "split_strategy": "temporal_newest_held_out",
        "verifier_status": "needs_review",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": manifest["status"],
        "repositories": repository_counts,
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "planned_rollouts": benchmark["planned_rollouts"],
        "estimated_evaluation_cost_usd": benchmark[
            "estimated_evaluation_cost_usd"
        ],
    }


def _fetch_merged_pull_requests(
    full_name: str,
    *,
    limit: int,
    token: str | None,
    request_json: RequestJson,
) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CastformRouterCLI/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    merged = []
    for page in range(1, 6):
        url = (
            f"https://api.github.com/repos/{full_name}/pulls"
            f"?state=closed&sort=updated&direction=desc&per_page=100&page={page}"
        )
        values = request_json(url, headers)
        if not isinstance(values, list):
            raise ValueError(f"GitHub returned an invalid PR list for {full_name}")
        merged.extend(
            value
            for value in values
            if isinstance(value, dict) and value.get("merged_at")
        )
        if len(merged) >= limit or len(values) < 100:
            break
    return merged[:limit]


def _pull_request_task(
    full_name: str,
    pull_request: dict[str, Any],
    *,
    include_body: bool,
    exclude_labels: set[str],
) -> dict[str, Any] | None:
    labels = {
        str(label.get("name", "")).lower()
        for label in pull_request.get("labels", [])
        if isinstance(label, dict)
    }
    if labels & exclude_labels or pull_request.get("draft"):
        return None
    number = pull_request.get("number")
    title = pull_request.get("title")
    base = pull_request.get("base")
    head = pull_request.get("head")
    if (
        not isinstance(number, int)
        or not isinstance(title, str)
        or not isinstance(base, dict)
        or not isinstance(base.get("sha"), str)
        or not isinstance(head, dict)
        or not isinstance(head.get("sha"), str)
    ):
        return None

    task_text = title.strip()
    body = pull_request.get("body")
    if include_body and isinstance(body, str) and body.strip():
        task_text = f"{task_text}\n\n{body.strip()[:6000]}"
    owner, name = full_name.split("/", 1)
    return {
        "task_id": f"{owner}-{name}-pr-{number}",
        "repository": full_name,
        "base_commit": base["sha"],
        "task_text": task_text,
        "verifier": {
            "type": "repository_native",
            "status": "needs_review",
            "commands": [],
        },
        "source": {
            "type": "github_pull_request",
            "number": number,
            "url": pull_request.get("html_url"),
            "merged_at": pull_request["merged_at"],
            "reference_head_commit": head["sha"],
            "reference_merge_commit": pull_request.get("merge_commit_sha"),
            "labels": sorted(labels),
        },
    }


def _repository_token(auth: object) -> str | None:
    if not isinstance(auth, dict):
        raise ValueError("repository auth must be an object")
    strategy = auth.get("strategy", "public")
    if strategy == "public":
        return None
    token_env = (
        auth.get("token_env")
        if strategy == "token_env"
        else auth.get("installation_token_env")
    )
    if not isinstance(token_env, str):
        raise ValueError(
            "GitHub App PR mining needs installation_token_env containing "
            "a short-lived installation token"
        )
    token = os.getenv(token_env)
    if not token:
        raise ValueError(f"required GitHub credential is not set: {token_env}")
    return token


def _evaluation_count(total: int, ratio: float) -> int:
    if total < 2 or ratio <= 0:
        return 0
    return min(total - 1, max(1, math.ceil(total * ratio)))


def _request_json(url: str, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403, 404}:
            raise ValueError(
                "GitHub could not read pull requests. Check repository access "
                "and the configured token."
            ) from error
        raise ValueError(f"GitHub returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ValueError("GitHub pull-request fetch failed") from error


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
