from __future__ import annotations

import json
from pathlib import Path

import pytest
from castform_router.benchmax_workflow import REQUIRED_SCRIPTS
from castform_router.project import (
    create_training_project,
    load_project_spec,
)
from castform_router.workspace_cli import main


def _write_project(tmp_path: Path) -> Path:
    config = {
        "schema_version": "1",
        "auth_profiles": {
            "company": {
                "strategy": "github_app",
                "app_id_env": "CASTFORM_GITHUB_APP_ID",
                "private_key_env": "CASTFORM_GITHUB_PRIVATE_KEY",
                "installation_id_env": "CASTFORM_GITHUB_INSTALLATION_ID",
                "installation_token_env": "CASTFORM_GITHUB_INSTALLATION_TOKEN",
            }
        },
        "repositories": [
            {
                "repo": "acme/private-api",
                "auth_profile": "company",
                "revision": "main",
            }
        ],
        "pull_requests": {
            "limit_per_repo": 20,
            "eval_ratio": 0.2,
            "include_body": False,
            "exclude_labels": ["dependencies"],
        },
        "allowed_routes": [
            "claude-code/sonnet@anthropic",
            "codex/5.6-balanced@openai",
        ],
        "benchmark": {
            "tasks_per_repo": 2,
            "repetitions": 1,
            "average_run_cost_usd": 0.5,
            "execution": "customer_runner",
        },
    }
    path = tmp_path / "router-project.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_code_first_project_preserves_auth_references(tmp_path: Path) -> None:
    path = _write_project(tmp_path)
    spec = load_project_spec(path)
    result = create_training_project(
        spec,
        output_root=tmp_path / "runs",
    )

    workspace = Path(result["workspace_path"])
    manifest = json.loads((workspace / "manifest.json").read_text())

    assert manifest["repositories"][0]["auth"] == {
        "strategy": "github_app",
        "app_id_env": "CASTFORM_GITHUB_APP_ID",
        "private_key_env": "CASTFORM_GITHUB_PRIVATE_KEY",
        "installation_id_env": "CASTFORM_GITHUB_INSTALLATION_ID",
        "installation_token_env": "CASTFORM_GITHUB_INSTALLATION_TOKEN",
    }
    assert manifest["benchmark"]["planned_tasks"] == 2
    assert (workspace / "project.spec.json").exists()


def test_project_rejects_unknown_auth_profile(tmp_path: Path) -> None:
    path = _write_project(tmp_path)
    spec = json.loads(path.read_text())
    spec["repositories"][0]["auth_profile"] = "missing"
    path.write_text(json.dumps(spec))

    with pytest.raises(ValueError, match="unknown auth profile"):
        load_project_spec(path)


def test_project_defaults_to_castform_hosted(tmp_path: Path) -> None:
    path = _write_project(tmp_path)
    spec = json.loads(path.read_text())
    del spec["benchmark"]["execution"]
    path.write_text(json.dumps(spec))

    result = create_training_project(
        load_project_spec(path),
        output_root=tmp_path / "runs",
    )
    workspace = Path(result["workspace_path"])
    manifest = json.loads((workspace / "manifest.json").read_text())

    assert manifest["privacy_mode"] == "castform_hosted"


def test_cli_validates_and_creates_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_project(tmp_path)

    assert main(["validate", str(path)]) == 0
    assert "valid:" in capsys.readouterr().out

    output = tmp_path / "cli-runs"
    assert main(["create", str(path), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["workspace_path"]).is_dir()


def test_cli_prepare_writes_benchmax_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_project(tmp_path)
    workflow = tmp_path / "examples" / "model_router"
    workflow.mkdir(parents=True)
    for script in REQUIRED_SCRIPTS:
        (workflow / script).write_text("# fixture\n", encoding="utf-8")

    output = tmp_path / "prepared-runs"
    assert (
        main(
            [
                "prepare",
                str(path),
                "--output",
                str(output),
                "--workflow-dir",
                str(workflow),
                "--through",
                "gate",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    workspace = Path(payload["workspace_path"])

    assert payload["status"] == "ready_for_benchmax_mining"
    assert (
        workspace / "benchmax" / "model_router" / "workflow-plan.json"
    ).is_file()
    assert "materialization" not in payload
