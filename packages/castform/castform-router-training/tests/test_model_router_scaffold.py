from __future__ import annotations

import json
from pathlib import Path

from castform_router_training.model_router_scaffold import (
    _provider_for_route,
    scaffold_model_router_sft,
)


def test_scaffold_model_router_sft_writes_reviewable_workspace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model-router"
    tasks = source / "tasks"
    datasets = source / "datasets"
    tasks.mkdir(parents=True)
    datasets.mkdir()
    models = ("gpt-fast", "claude-strong")
    rows: list[dict[str, object]] = []
    for repo in ("alpha", "beta"):
        for task_index in range(2):
            task_id = f"{repo}-{task_index}"
            task_dir = tasks / task_id
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text(
                f"Fix {task_id} without seeing the verifier.",
                encoding="utf-8",
            )
            for model in models:
                harness = "codex" if model.startswith("gpt-") else "claude-code"
                rows.append(
                    {
                        "task_id": task_id,
                        "task_dir": f"tasks/{task_id}",
                        "repo": repo,
                        "merged_at": f"2026-01-0{task_index + 1}",
                        "route": model,
                        "harness": harness,
                        "reward": float(model == "gpt-fast" and task_index == 0),
                        "n_input_tokens": 100,
                        "n_cache_tokens": 20,
                        "n_output_tokens": 30,
                        "cost_usd": 0.1,
                    }
                )
    (datasets / "outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "task_count": 4,
                "pipeline": {"repository": "castform-ai/model-router"},
                "traces": {"cutoff": "2026-08-05T00:00:00Z"},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "run"
    result = scaffold_model_router_sft(
        source,
        output_dir=output,
        models=models,
        eval_ratio=0.5,
    )

    assert result.complete_tasks == 4
    assert result.train_examples == 2
    assert result.eval_examples == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["benchmark"]["split_strategy"] == "repo-temporal"
    assert [route["provider"] for route in manifest["candidate_routes"]] == ["openai", "anthropic"]
    request = json.loads((output / "router" / "contract" / "request.example.json").read_text())
    response = json.loads((output / "router" / "contract" / "response.example.json").read_text())
    expected_route_ids = tuple(backend["name"] for backend in request["backends"])
    assert response["scorer_version"] == "qwen35-08b-sft-v2"
    predictions = response["predictions"]
    assert {prediction["backend"] for prediction in predictions} == set(expected_route_ids)
    assert {prediction["success_probability"] for prediction in predictions} == {
        0.3333,
        0.6667,
    }
    assert (output / "router" / "contract" / "response.schema.json").is_file()
    token_bands = json.loads((output / "router" / "contract" / "token_bands.json").read_text())
    assert token_bands["input"]["256k_1m"] == 524_288
    assert (output / "router" / "training_config.json").is_file()


def test_deepseek_provider_wins_over_claude_code_harness() -> None:
    assert (
        _provider_for_route(
            model="deepseek-v4-flash",
            harness="claude-code",
        )
        == "deepseek"
    )
