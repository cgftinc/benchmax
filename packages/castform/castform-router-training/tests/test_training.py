from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from castform_router_training.dataset import build_dataset
from castform_router_training.project import load_project
from castform_router_training.repositories import mine_tasks


def write_project(path: Path, repo: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "repositories": [{"name": "demo", "url": str(repo), "revision": "main"}],
                "backends": [
                    {"name": "cheap", "model": "small", "provider": "local"},
                    {"name": "strong", "model": "large", "provider": "cloud"},
                ],
            }
        )
    )


def test_project_requires_unique_backends(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    write_project(path, tmp_path / "repo")
    value = json.loads(path.read_text())
    value["backends"][1]["name"] = "cheap"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="backend names must be unique"):
        load_project(path)


def test_dataset_uses_measured_success_rates(tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    write_project(project_path, tmp_path / "repo")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "demo:1", "task": "Fix parser"}) + "\n")
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "demo:1", "backend": "cheap", "success": True}),
                json.dumps({"task_id": "demo:1", "backend": "cheap", "success": False}),
                json.dumps({"task_id": "demo:1", "backend": "strong", "success": True}),
            ]
        )
        + "\n"
    )
    output = tmp_path / "train.jsonl"
    assert build_dataset(load_project(project_path), tasks, outcomes, output) == 1
    assistant = json.loads(json.loads(output.read_text())["messages"][2]["content"])
    assert assistant["predictions"] == [
        {"backend": "cheap", "success_probability": 0.5},
        {"backend": "strong", "success_probability": 1.0},
    ]


def test_mine_tasks_reads_local_git_history(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    (repo / "parser.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "parser.py"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "Fix parser",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    project_path = tmp_path / "project.json"
    write_project(project_path, repo)
    tasks_path = mine_tasks(load_project(project_path), tmp_path / "run", limit_per_repo=1)
    task = json.loads(tasks_path.read_text())
    assert task["repository"] == "demo"
    assert task["task"] == "Fix parser"
