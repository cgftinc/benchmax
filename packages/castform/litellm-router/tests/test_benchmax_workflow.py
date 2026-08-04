from __future__ import annotations

import json
from pathlib import Path

import pytest
from castform_router.benchmax_workflow import (
    REQUIRED_SCRIPTS,
    build_benchmax_plan,
    select_plan_steps,
    write_benchmax_plan,
)
from castform_router.project import create_training_project


def _project(tmp_path: Path) -> Path:
    spec = {
        "schema_version": "1",
        "auth_profiles": {"public": {"strategy": "public"}},
        "repositories": [
            {
                "repo": "pallets/click",
                "revision": "main",
                "auth_profile": "public",
            }
        ],
        "pull_requests": {
            "limit_per_repo": 20,
            "eval_ratio": 0.2,
        },
        "allowed_routes": [
            "claude-code/sonnet@anthropic",
            "codex/5.6-balanced@openai",
        ],
        "benchmark": {
            "repetitions": 1,
            "execution": "castform_hosted",
        },
    }
    result = create_training_project(spec, output_root=tmp_path / "runs")
    return Path(result["workspace_path"])


def _workflow(tmp_path: Path) -> Path:
    workflow = tmp_path / "examples" / "model_router"
    workflow.mkdir(parents=True)
    for script in REQUIRED_SCRIPTS:
        (workflow / script).write_text("# upstream fixture\n", encoding="utf-8")
    return workflow


def test_plan_uses_existing_benchmax_scripts(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    workflow = _workflow(tmp_path)

    plan = build_benchmax_plan(workspace, workflow_dir=workflow)
    commands = [" ".join(step["command"]) for step in plan["steps"]]

    assert plan["implementation"] == "benchmax/examples/model_router"
    assert any("codeprobe mine" in command for command in commands)
    mine_steps = [
        step
        for step in plan["steps"]
        if step["stage"] == "mine" and step["name"].startswith("mine-")
    ]
    assert mine_steps
    assert all(not step["spends_model_credits"] for step in mine_steps)
    assert all("--no-llm" in step["command"] for step in mine_steps)
    assert any("convert_to_harbor.py" in command for command in commands)
    assert any("gate_tasks.py" in command for command in commands)
    assert any(
        "harbor run" in command and "claude-sonnet-4-6" in command
        for command in commands
    )
    assert any(
        "harbor run" in command and "gpt-5.6-terra" in command
        for command in commands
    )
    assert any("build_dataset.py" in command for command in commands)
    assert any("knn_router.py" in command for command in commands)
    assert any("scoreboard.py" in command for command in commands)


def test_plan_can_opt_into_codeprobe_model_enrichment(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    plan = build_benchmax_plan(
        workspace,
        workflow_dir=_workflow(tmp_path),
        codeprobe_llm=True,
    )

    mine_steps = [
        step
        for step in plan["steps"]
        if step["stage"] == "mine" and step["name"].startswith("mine-")
    ]

    assert mine_steps
    assert all(step["spends_model_credits"] for step in mine_steps)
    assert all("--no-llm" not in step["command"] for step in mine_steps)


def test_plan_records_benchmax_source_and_status(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    plan = build_benchmax_plan(workspace, workflow_dir=_workflow(tmp_path))

    plan_path = write_benchmax_plan(workspace, plan)
    manifest = json.loads((workspace / "manifest.json").read_text())
    gated_steps = select_plan_steps(
        plan,
        from_stage="mine",
        through_stage="gate",
    )

    assert plan_path.is_file()
    assert manifest["status"] == "ready_for_benchmax_mining"
    assert manifest["benchmax_workflow"]["implementation"] == (
        "benchmax/examples/model_router"
    )
    assert {step["stage"] for step in gated_steps} == {
        "mine",
        "convert",
        "gate",
    }


@pytest.mark.parametrize(
    ("router_rung", "script", "spends_credits"),
    [
        ("knn", "knn_router.py", False),
        ("profile", "profile_router.py", True),
        ("baseline", "baseline_router.py", True),
    ],
)
def test_plan_supports_each_router_rung(
    tmp_path: Path,
    router_rung: str,
    script: str,
    spends_credits: bool,
) -> None:
    workspace = _project(tmp_path)
    plan = build_benchmax_plan(
        workspace,
        workflow_dir=_workflow(tmp_path),
        router_rung=router_rung,
        router_model="claude-sonnet-4-6",
    )

    router_step = next(
        step for step in plan["steps"] if step["stage"] == "router"
    )
    scoreboard_step = next(
        step for step in plan["steps"] if step["stage"] == "scoreboard"
    )

    assert script in " ".join(router_step["command"])
    assert router_step["spends_model_credits"] is spends_credits
    assert f"picks_{router_rung}.jsonl" in " ".join(
        scoreboard_step["command"]
    )
    assert plan["router_model"] == "claude-sonnet-4-6"
