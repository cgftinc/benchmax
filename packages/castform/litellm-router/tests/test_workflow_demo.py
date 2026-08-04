from __future__ import annotations

from pathlib import Path

from castform_router.benchmax_workflow import (
    REQUIRED_SCRIPTS,
    build_benchmax_plan,
    write_benchmax_plan,
)
from castform_router.project import create_training_project
from castform_router.workflow_demo import (
    DEMO_STAGES,
    advance_demo,
    load_demo_state,
    reset_demo,
)


def _workspace(tmp_path: Path, *, router_rung: str = "knn") -> Path:
    spec = {
        "schema_version": "1",
        "repositories": [
            {
                "repo": "pallets/click",
                "revision": "main",
                "auth": {"strategy": "public"},
            }
        ],
        "pull_requests": {"limit_per_repo": 10, "eval_ratio": 0.2},
        "allowed_routes": [
            "claude-code/opus@anthropic",
            "codex/5.6-balanced@openai",
        ],
        "benchmark": {"repetitions": 1, "execution": "castform_hosted"},
    }
    result = create_training_project(spec, output_root=tmp_path / "runs")
    workspace = Path(result["workspace_path"])
    workflow = tmp_path / "examples" / "model_router"
    workflow.mkdir(parents=True)
    for script in REQUIRED_SCRIPTS:
        (workflow / script).write_text("# fixture\n", encoding="utf-8")
    write_benchmax_plan(
        workspace,
        build_benchmax_plan(
            workspace,
            workflow_dir=workflow,
            router_rung=router_rung,
            router_model="claude-sonnet-4-6",
        ),
    )
    return workspace


def test_demo_advances_across_real_workflow_stages(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = load_demo_state(workspace)

    assert state["mode"] == "simulation"
    assert state["stages"][0]["status"] == "next"
    for index, (stage_id, _label) in enumerate(DEMO_STAGES):
        state = advance_demo(workspace)
        assert state["events"][index]["id"] == stage_id
        assert state["events"][index]["commands"]

    assert state["complete"] is True
    assert state["events"][-1]["output"]["rows"]
    assert (
        workspace
        / "benchmax"
        / "model_router"
        / "browser-demo-state.json"
    ).is_file()


def test_demo_reset_does_not_execute_or_keep_events(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    advance_demo(workspace)

    state = reset_demo(workspace)

    assert state["current_stage"] == -1
    assert state["events"] == []
    assert state["notice"].startswith("Illustrative local walkthrough")


def test_mining_demo_distinguishes_limit_from_real_candidates(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    advance_demo(workspace)

    state = advance_demo(workspace)
    event = state["events"][-1]

    assert event["id"] == "mine"
    assert event["output"]["configured_pr_limit"] == 10
    assert event["output"]["actual_candidates_mined"] is None
    assert event["output"]["preview_only"] is True
    assert len(event["output"]["candidate_tasks"]) == 4
    assert "No task files exist" in event["artifact_notice"]


def test_demo_router_stage_reflects_selected_profile_rung(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, router_rung="profile")

    state = load_demo_state(workspace)
    router_stage = next(
        stage for stage in state["stages"] if stage["id"] == "router"
    )
    assert router_stage["spends_model_credits"] is True

    for _ in range(7):
        state = advance_demo(workspace)
    event = state["events"][-1]

    assert event["id"] == "router"
    assert event["output"]["router_rung"] == "profile"
    assert event["output"]["router_model"] == "claude-sonnet-4-6"
    assert event["artifact"].endswith("picks_profile.jsonl")
