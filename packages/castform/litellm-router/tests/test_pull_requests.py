from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from castform_router.pull_requests import materialize_pull_request_tasks
from castform_router.training_environment import build_training_workspace


def _pull_request(
    number: int,
    *,
    merged_at: str,
    label: str = "bug",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix issue {number}",
        "body": f"Implementation details for {number}",
        "html_url": f"https://github.com/pallets/click/pull/{number}",
        "merged_at": merged_at,
        "merge_commit_sha": f"merge-{number}",
        "draft": False,
        "labels": [{"name": label}],
        "base": {"sha": f"base-{number}"},
        "head": {"sha": f"head-{number}"},
    }


def test_materializes_merged_prs_into_temporal_splits(tmp_path: Path) -> None:
    result = build_training_workspace(
        tmp_path,
        repositories=[
            {
                "full_name": "pallets/click",
                "auth": {"strategy": "public"},
            }
        ],
        selected_route_ids=[
            "claude-code/sonnet@anthropic",
            "codex/5.6-balanced@openai",
        ],
        tasks_per_repo=20,
        repetitions=2,
        average_run_cost_usd=0.5,
        privacy_mode="customer_runner",
    )
    workspace = Path(result["workspace_path"])
    pull_requests = [
        _pull_request(1, merged_at="2026-01-01T00:00:00Z"),
        _pull_request(2, merged_at="2026-02-01T00:00:00Z"),
        _pull_request(3, merged_at="2026-03-01T00:00:00Z"),
        _pull_request(
            4,
            merged_at="2026-04-01T00:00:00Z",
            label="dependencies",
        ),
    ]

    summary = materialize_pull_request_tasks(
        workspace,
        settings={
            "limit_per_repo": 20,
            "eval_ratio": 0.34,
            "exclude_labels": ["dependencies"],
        },
        request_json=lambda _url, _headers: pull_requests,
    )

    train = [
        json.loads(line)
        for line in (
            workspace / "benchmax" / "tasks" / "train.jsonl"
        ).read_text().splitlines()
    ]
    evaluation = [
        json.loads(line)
        for line in (
            workspace / "benchmax" / "tasks" / "eval.jsonl"
        ).read_text().splitlines()
    ]
    manifest = json.loads((workspace / "manifest.json").read_text())

    assert [task["task_id"] for task in train] == ["pallets-click-pr-1"]
    assert [task["task_id"] for task in evaluation] == [
        "pallets-click-pr-2",
        "pallets-click-pr-3",
    ]
    assert evaluation[-1]["base_commit"] == "base-3"
    assert evaluation[-1]["verifier"]["status"] == "needs_review"
    assert summary["planned_rollouts"] == 12
    assert manifest["status"] == "awaiting_verifier_review"


def test_pr_body_is_opt_in(tmp_path: Path) -> None:
    result = build_training_workspace(
        tmp_path,
        repositories=[{"full_name": "pallets/click"}],
        selected_route_ids=[
            "claude-code/sonnet@anthropic",
            "codex/5.6-balanced@openai",
        ],
        tasks_per_repo=10,
        repetitions=1,
        average_run_cost_usd=0.5,
        privacy_mode="customer_runner",
    )
    workspace = Path(result["workspace_path"])

    materialize_pull_request_tasks(
        workspace,
        settings={"include_body": False},
        request_json=lambda _url, _headers: [
            _pull_request(1, merged_at="2026-01-01T00:00:00Z")
        ],
    )
    task = json.loads(
        (workspace / "benchmax" / "tasks" / "train.jsonl")
        .read_text()
        .splitlines()[0]
    )

    assert task["task_text"] == "Fix issue 1"
