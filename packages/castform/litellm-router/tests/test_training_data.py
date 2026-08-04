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
        train_row["target"]["predictions"][0]["expected_cache_read_tokens"]
        == 20
    )
    costs = json.loads(
        (tmp_path / "router-data" / "route_costs.json").read_text(
            encoding="utf-8"
        )
    )
    assert costs["routes"]["claude-code/sonnet@anthropic"] == 0.25
