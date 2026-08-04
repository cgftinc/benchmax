from __future__ import annotations

import json
from pathlib import Path

import pytest
from castform_router.training_environment import (
    build_training_workspace,
    parse_github_repo,
)


def test_parse_github_repo_accepts_url_and_slug() -> None:
    assert parse_github_repo("castform-ai/benchmax")["full_name"] == (
        "castform-ai/benchmax"
    )
    assert parse_github_repo(
        "https://github.com/castform-ai/benchmax.git"
    )["html_url"] == "https://github.com/castform-ai/benchmax"


def test_parse_github_repo_rejects_other_hosts() -> None:
    with pytest.raises(ValueError, match="only https://github.com"):
        parse_github_repo("https://example.com/org/repo")


def test_build_training_workspace_writes_versioned_contract(
    tmp_path: Path,
) -> None:
    result = build_training_workspace(
        tmp_path,
        repositories=[
            {
                "html_url": "https://github.com/castform-ai/benchmax",
                "default_branch": "main",
                "visibility": "public",
                "verification": "verified_public",
            }
        ],
        selected_route_ids=[
            "claude-code/sonnet@anthropic",
            "codex/5.6-balanced@openai",
        ],
        tasks_per_repo=10,
        repetitions=2,
        average_run_cost_usd=1.25,
        privacy_mode="customer_runner",
    )

    workspace = Path(result["workspace_path"])
    manifest = json.loads((workspace / "manifest.json").read_text())
    contract = json.loads(
        (workspace / "router" / "training_contract.json").read_text()
    )

    assert manifest["status"] == "awaiting_task_extraction"
    assert manifest["benchmark"]["planned_rollouts"] == 40
    assert manifest["benchmark"]["estimated_evaluation_cost_usd"] == 50.0
    assert len(contract["learned_model_input"]["candidate_routes"]) == 2
    assert len(contract["learned_model_output"]["predictions"]) == 2
    assert "live_price" in contract["policy_only_fields"]
    assert (
        manifest["router"]["base_model"]
        == "Qwen/Qwen3.5-0.8B"
    )
    assert (workspace / "router" / "training_config.json").exists()
    assert (workspace / "benchmax" / "tasks" / "train.jsonl").exists()
    assert "manifest.json" in result["files"]


def test_build_training_workspace_requires_two_routes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two routes"):
        build_training_workspace(
            tmp_path,
            repositories=[{"full_name": "castform-ai/benchmax"}],
            selected_route_ids=["claude-code/sonnet@anthropic"],
            tasks_per_repo=10,
            repetitions=2,
            average_run_cost_usd=1.0,
            privacy_mode="customer_runner",
        )
