"""Deterministic browser walkthrough for the Benchmax model-router workflow.

The walkthrough never executes repository code or model calls. It advances
through the real workflow plan and returns clearly labeled illustrative
artifacts so a user can understand the product before launching a runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEMO_STAGES = (
    ("setup", "Prepare runner"),
    ("mine", "Mine repository history"),
    ("convert", "Build Harbor tasks"),
    ("gate", "Qualify verifiers"),
    ("collect", "Benchmark routes"),
    ("dataset", "Build training dataset"),
    ("router", "Run router baseline"),
    ("scoreboard", "Compare policies"),
)
DEMO_SCHEMA_VERSION = "3"


def load_demo_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(value, dict)
            and value.get("schema_version") == DEMO_SCHEMA_VERSION
        ):
            return value
    state = _initial_state(workspace)
    _write_state(path, state)
    return state


def advance_demo(workspace: Path) -> dict[str, Any]:
    state = load_demo_state(workspace)
    next_index = int(state["current_stage"]) + 1
    if next_index >= len(DEMO_STAGES):
        return state
    stage_id, label = DEMO_STAGES[next_index]
    event = _stage_event(workspace, stage_id, label)
    state["current_stage"] = next_index
    state["events"].append(event)
    for index, stage in enumerate(state["stages"]):
        stage["status"] = (
            "completed"
            if index <= next_index
            else "next"
            if index == next_index + 1
            else "pending"
        )
    state["complete"] = next_index == len(DEMO_STAGES) - 1
    _write_state(_state_path(workspace), state)
    return state


def reset_demo(workspace: Path) -> dict[str, Any]:
    state = _initial_state(workspace)
    _write_state(_state_path(workspace), state)
    return state


def _initial_state(workspace: Path) -> dict[str, Any]:
    manifest = _read_object(workspace / "manifest.json")
    plan = _read_object(
        workspace / "benchmax" / "model_router" / "workflow-plan.json"
    )
    credit_stages = {
        str(step.get("stage"))
        for step in plan.get("steps", [])
        if step.get("spends_model_credits")
    }
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "mode": "simulation",
        "notice": (
            "Illustrative local walkthrough. No repository code, tests, "
            "model calls, or paid runner jobs are executed."
        ),
        "workspace_id": manifest["workspace_id"],
        "current_stage": -1,
        "complete": False,
        "stages": [
            {
                "id": stage_id,
                "label": label,
                "status": "next" if index == 0 else "pending",
                "spends_model_credits": stage_id in credit_stages,
            }
            for index, (stage_id, label) in enumerate(DEMO_STAGES)
        ],
        "events": [],
    }


def _stage_event(
    workspace: Path,
    stage_id: str,
    label: str,
) -> dict[str, Any]:
    manifest = _read_object(workspace / "manifest.json")
    plan = _read_object(
        workspace / "benchmax" / "model_router" / "workflow-plan.json"
    )
    commands = [
        step["command"]
        for step in plan.get("steps", [])
        if step.get("stage") == stage_id
    ]
    repositories = manifest.get("repositories", [])
    routes = manifest.get("candidate_routes", [])
    candidates = int(manifest.get("benchmark", {}).get("planned_tasks", 20))
    qualified = max(1, round(candidates * 0.7))
    review = max(1, round(candidates * 0.1))
    rejected = max(0, candidates - qualified - review)
    repetitions = int(manifest.get("benchmark", {}).get("repetitions", 1))

    common = {
        "id": stage_id,
        "label": label,
        "status": "completed",
        "simulation": True,
        "commands": commands,
    }
    if stage_id == "setup":
        return {
            **common,
            "summary": "Prepared an isolated runner and installed the upstream workflow.",
            "metrics": [
                {"label": "Workflow", "value": "examples/model_router"},
                {"label": "Network", "value": plan.get("agent_network", "allowlist")},
                {"label": "Credits", "value": "$0"},
            ],
            "artifact": "benchmax/model_router/workflow-plan.json",
            "output": {
                "source": plan.get("source"),
                "packages": ["codeprobe>=0.13,<0.14", "harbor>=0.18,<0.19"],
            },
        }
    if stage_id == "mine":
        preview_tasks = [
            {
                "candidate_id": "click-pr-preview-01",
                "title": "Show custom error message in hidden prompt",
                "repository": "pallets/click",
                "status": "illustrative",
            },
            {
                "candidate_id": "click-pr-preview-02",
                "title": "Fix parsing when a parameter is named help",
                "repository": "pallets/click",
                "status": "illustrative",
            },
            {
                "candidate_id": "click-pr-preview-03",
                "title": "Add missing spacing to deprecation label",
                "repository": "pallets/click",
                "status": "illustrative",
            },
            {
                "candidate_id": "click-pr-preview-04",
                "title": "Preserve parameter order when formatting help",
                "repository": "pallets/click",
                "status": "illustrative",
            },
        ]
        return {
            **common,
            "summary": (
                "Previewed how CodeProbe would mine repository history. "
                "This simulation does not clone Click or inspect real PRs."
            ),
            "metrics": [
                {"label": "Repositories", "value": str(len(repositories))},
                {"label": "PR mining limit", "value": f"Up to {candidates}"},
                {"label": "Preview shown", "value": str(len(preview_tasks))},
                {
                    "label": "Live mining",
                    "value": (
                        "Model enrichment"
                        if plan.get("codeprobe_llm")
                        else "$0 · deterministic"
                    ),
                },
            ],
            "artifact": "repos/<repo>/.codeprobe/tasks/",
            "artifact_notice": (
                "Planned live-run path. No task files exist in this $0 "
                "simulation. A live run writes up to the configured PR limit; "
                "it may produce fewer candidates."
            ),
            "output": {
                "repositories": [
                    repository.get("full_name") for repository in repositories
                ],
                "configured_pr_limit": candidates,
                "actual_candidates_mined": None,
                "preview_count": len(preview_tasks),
                "candidate_tasks": preview_tasks,
                "preview_only": True,
                "live_mining_may_spend_credits": bool(
                    plan.get("codeprobe_llm")
                ),
            },
        }
    if stage_id == "convert":
        return {
            **common,
            "summary": "Converted candidates into leak-guarded Harbor tasks.",
            "metrics": [
                {"label": "Harbor tasks", "value": str(candidates)},
                {"label": "Base checkout", "value": "isolated"},
                {"label": "PR test overlay", "value": "enabled"},
            ],
            "artifact": "benchmax/model_router/harbor_tasks/<repo>/",
            "output": {
                "task_files": [
                    "instruction.md",
                    "task.toml",
                    "environment/Dockerfile",
                    "solution/fix.patch",
                    "tests/test.sh",
                ],
                "anti_leak": [
                    "fresh git init",
                    "fetch base commit only",
                    "drop remote",
                ],
            },
        }
    if stage_id == "gate":
        return {
            **common,
            "summary": "Kept only tasks whose verifier discriminates reliably.",
            "metrics": [
                {"label": "Qualified", "value": str(qualified)},
                {"label": "Review", "value": str(review)},
                {"label": "Rejected", "value": str(rejected)},
            ],
            "artifact": "harbor_tasks/<repo>/manifest.json",
            "output": {
                "example_pass": {
                    "verdict": "pass",
                    "oracle": [1.0, 1.0, 1.0],
                    "nop": [0.0, 0.0, 0.0],
                },
                "drop_reasons": ["oracle_fail", "nop_pass", "flaky", "error"],
            },
        }
    if stage_id == "collect":
        route_results = [_route_result(route) for route in routes]
        return {
            **common,
            "summary": "Simulated Harbor trials across every allowed route.",
            "metrics": [
                {"label": "Qualified tasks", "value": str(qualified)},
                {"label": "Routes", "value": str(len(routes))},
                {
                    "label": "Rollouts",
                    "value": str(qualified * len(routes) * repetitions),
                },
            ],
            "artifact": "benchmax/model_router/harbor_runs/",
            "output": {
                "route_results": route_results,
                "would_spend_credits_live": True,
                "simulation_cost_usd": 0,
            },
        }
    if stage_id == "dataset":
        rows = qualified * len(routes) * repetitions
        return {
            **common,
            "summary": "Audited trajectories and flattened clean trials into JSONL.",
            "metrics": [
                {"label": "Rows", "value": str(rows)},
                {"label": "Train tasks", "value": str(max(1, round(qualified * 0.8)))},
                {"label": "Test tasks", "value": str(max(1, qualified - round(qualified * 0.8)))},
            ],
            "artifact": "benchmax/model_router/dataset.jsonl",
            "output": {
                "fields": [
                    "task_id",
                    "repo",
                    "merged_at",
                    "route",
                    "harness",
                    "reward",
                    "cost_usd",
                    "token_counts",
                ],
                "filters": ["gate-passed", "audit-clean", "real-agent"],
                "split": "temporal: earlier train, later test",
            },
        }
    if stage_id == "router":
        router_rung = str(plan.get("router_rung") or "knn")
        router_model = str(
            plan.get("router_model") or "claude-sonnet-4-6"
        )
        rung_details = {
            "knn": {
                "label": "kNN · k=3",
                "summary": (
                    "Ran the TF-IDF nearest-neighbor router over earlier "
                    "task instructions."
                ),
                "cost": "$0",
                "reasoning": "kNN k=3 over earlier task instructions",
            },
            "profile": {
                "label": "Profile",
                "summary": (
                    "Asked a model to route using the task plus train-split "
                    "route performance."
                ),
                "cost": "Model calls",
                "reasoning": (
                    f"Profile prompt scored by {router_model} using earlier "
                    "route outcomes"
                ),
            },
            "baseline": {
                "label": "Zero-shot",
                "summary": (
                    "Asked a model to select a route from the task prompt "
                    "without training examples."
                ),
                "cost": "Model calls",
                "reasoning": (
                    f"Zero-shot route pick from {router_model}"
                ),
            },
        }
        details = rung_details.get(router_rung, rung_details["knn"])
        return {
            **common,
            "summary": details["summary"],
            "metrics": [
                {"label": "Router", "value": details["label"]},
                {"label": "Router cost", "value": details["cost"]},
                {
                    "label": "Output",
                    "value": f"picks_{router_rung}.jsonl",
                },
            ],
            "artifact": (
                "benchmax/model_router/router_outputs/"
                f"picks_{router_rung}.jsonl"
            ),
            "output": {
                "example_pick": {
                    "task_id": "c943271a",
                    "model": _best_demo_model(routes),
                    "router_cost_usd": (
                        0.0 if router_rung == "knn" else None
                    ),
                    "reasoning": details["reasoning"],
                },
                "router_rung": router_rung,
                "router_model": (
                    None if router_rung == "knn" else router_model
                ),
            },
        }
    return {
        **common,
        "summary": "Compared routing policies on the temporal test split.",
        "metrics": [
            {"label": "Primary metric", "value": "solve rate"},
            {"label": "Cost metric", "value": "$ / task"},
            {"label": "Subset", "value": "routable tasks"},
        ],
        "artifact": "scoreboard stdout / router report",
        "output": {
            "rows": _scoreboard_rows(routes),
            "decision": (
                "Collect more tasks before training; this walkthrough is too "
                "small for an 800M-model claim."
            ),
        },
    }


def _route_result(route: dict[str, Any]) -> dict[str, Any]:
    model = str(route.get("harbor_model") or route.get("model"))
    known = {
        "claude-opus-5": (0.50, 1.77),
        "claude-sonnet-4-6": (0.263, 1.01),
        "claude-haiku-4-5": (0.263, 0.50),
        "gpt-5.6-luna": (0.263, 0.22),
        "gpt-5.6-terra": (0.184, 0.30),
        "gpt-5.6-sol": (0.211, 1.09),
    }
    solve_rate, cost = known.get(model, (0.30, 0.40))
    return {
        "route_id": route.get("route_id"),
        "model": model,
        "solve_rate": solve_rate,
        "cost_per_task_usd": cost,
    }


def _best_demo_model(routes: list[dict[str, Any]]) -> str:
    results = [_route_result(route) for route in routes]
    if not results:
        return "unknown"
    return str(
        max(results, key=lambda result: result["solve_rate"])["model"]
    )


def _scoreboard_rows(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = [_route_result(route) for route in routes]
    if not results:
        return []
    rows = [
        {
            "policy": f"always-{result['model']}",
            "solve_rate": result["solve_rate"],
            "cost_per_task_usd": result["cost_per_task_usd"],
        }
        for result in results
    ]
    mean_solve = sum(row["solve_rate"] for row in rows) / len(rows)
    mean_cost = sum(row["cost_per_task_usd"] for row in rows) / len(rows)
    best = max(rows, key=lambda row: row["solve_rate"])
    rows.extend(
        [
            {
                "policy": "random",
                "solve_rate": round(mean_solve, 3),
                "cost_per_task_usd": round(mean_cost, 2),
            },
            {
                "policy": "router-kNN",
                "solve_rate": min(0.95, round(best["solve_rate"] + 0.05, 3)),
                "cost_per_task_usd": round(mean_cost * 0.8, 2),
            },
            {
                "policy": "ORACLE (ceiling)",
                "solve_rate": min(1.0, round(best["solve_rate"] + 0.18, 3)),
                "cost_per_task_usd": round(mean_cost, 2),
            },
        ]
    )
    return rows


def _state_path(workspace: Path) -> Path:
    return workspace / "benchmax" / "model_router" / "browser-demo-state.json"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
