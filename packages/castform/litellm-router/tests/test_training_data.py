from __future__ import annotations

import json
from pathlib import Path

from castform_router.training_data import format_benchmax_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_formatter_builds_full_route_matrix_without_prices(tmp_path: Path) -> None:
    routes = [
        {
            "route_id": "claude-code/sonnet@anthropic",
            "harness": "claude-code",
            "model": "sonnet",
            "provider": "anthropic",
            "harbor_model": "claude-sonnet-4-6",
            "estimated_cost_usd": 9.99,
        },
        {
            "route_id": "codex/5.6-balanced@openai",
            "harness": "codex",
            "model": "5.6-balanced",
            "provider": "openai",
            "harbor_model": "gpt-5.6-terra",
            "estimated_cost_usd": 1.23,
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidate_routes": routes}), encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    rows: list[dict[str, object]] = []
    for task_index in range(3):
        for route in routes:
            rows.append(
                {
                    "task_id": f"task-{task_index}",
                    "task_text": f"Fix task {task_index}.",
                    "repo": "acme/api",
                    "merged_at": f"2026-01-0{task_index + 1}",
                    "route": route["harbor_model"],
                    "harness": route["harness"],
                    "reward": 1 if task_index != 1 else 0,
                    "n_input_tokens": 100 + task_index,
                    "n_cache_tokens": 20,
                    "n_output_tokens": 30,
                    "cost_usd": 0.25
                    if route["harness"] == "claude-code"
                    else 0.1,
                }
            )
    _write_jsonl(dataset, rows)

    result = format_benchmax_dataset(
        dataset,
        manifest_path=manifest,
        output_dir=tmp_path / "router-data",
        eval_ratio=0.34,
    )

    assert result.complete_tasks == 3
    assert result.train_examples == 1
    assert result.eval_examples == 2
    train_row = json.loads(
        (tmp_path / "router-data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    serialized_request = json.dumps(train_row["request"])
    assert "estimated_cost_usd" not in serialized_request
    assert len(train_row["target"]["predictions"]) == 2
    assert (
        train_row["target"]["predictions"][0]["cache_read_token_band"]
        == "under_64k"
    )
    assert (
        train_row["target"]["predictions"][0]["success_probability"]
        == 0.6667
    )
    evidence = train_row["label_metadata"]["route_evidence"][0]
    assert evidence["attempts"] == 1
    assert evidence["mean_cache_read_tokens"] == 20
    costs = json.loads(
        (tmp_path / "router-data" / "route_costs.json").read_text(
            encoding="utf-8"
        )
    )
    assert costs["routes"]["claude-code/sonnet@anthropic"] == 0.25


def test_formatter_reads_model_router_tasks_and_splits_within_repo(
    tmp_path: Path,
) -> None:
    routes = [
        {
            "route_id": "codex/fast@openai",
            "harness": "codex",
            "model": "fast",
            "provider": "openai",
            "harbor_model": "gpt-fast",
        },
        {
            "route_id": "codex/strong@openai",
            "harness": "codex",
            "model": "strong",
            "provider": "openai",
            "harbor_model": "gpt-strong",
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidate_routes": routes}), encoding="utf-8")
    tasks_root = tmp_path / "tasks"
    rows: list[dict[str, object]] = []
    for repo in ("alpha", "beta"):
        for task_index in range(3):
            task_id = f"{repo}-{task_index}"
            task_dir = tasks_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "instruction.md").write_text(
                f"Fix {task_id}.",
                encoding="utf-8",
            )
            for route in routes:
                rows.append(
                    {
                        "task_id": task_id,
                        "task_dir": f"tasks/{task_id}",
                        "repo": repo,
                        "merged_at": f"2026-01-0{task_index + 1}",
                        "route": route["harbor_model"],
                        "harness": route["harness"],
                        "reward": float(task_index == 0),
                        "n_input_tokens": 100,
                        "n_cache_tokens": 20,
                        "n_output_tokens": 30,
                        "cost_usd": 0.1,
                    }
                )
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    dataset = dataset_dir / "outcomes.jsonl"
    _write_jsonl(dataset, rows)

    result = format_benchmax_dataset(
        dataset,
        manifest_path=manifest,
        output_dir=tmp_path / "router-data",
        eval_ratio=0.5,
        tasks_root=tasks_root,
        split_strategy="repo-temporal",
    )

    assert result.train_examples == 4
    assert result.eval_examples == 2
    train = [
        json.loads(line)
        for line in Path(result.train_path).read_text().splitlines()
    ]
    evaluation = [
        json.loads(line)
        for line in Path(result.eval_path).read_text().splitlines()
    ]
    assert {row["example_id"] for row in train} == {
        "alpha-0",
        "alpha-1",
        "beta-0",
        "beta-1",
    }
    assert {row["example_id"] for row in evaluation} == {
        "alpha-2",
        "beta-2",
    }
    assert train[0]["request"]["task"]["text"].startswith("Fix ")
