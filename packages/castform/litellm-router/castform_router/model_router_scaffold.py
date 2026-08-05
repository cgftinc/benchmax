"""Scaffold a Castform SFT workspace from the model-router corpus."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from castform_router.router_protocol import model_response_json_schema
from castform_router.token_bands import token_band_representatives
from castform_router.training_data import format_benchmax_dataset


@dataclass(frozen=True, slots=True)
class ModelRouterScaffoldSummary:
    workspace: str
    source_revision: str
    selected_models: tuple[str, ...]
    complete_tasks: int
    skipped_incomplete_tasks: int
    train_examples: int
    eval_examples: int
    manifest_path: str
    request_example_path: str
    response_example_path: str
    next_command: str


def scaffold_model_router_sft(
    source_repo: Path,
    *,
    output_dir: Path,
    models: tuple[str, ...],
    eval_ratio: float = 0.5,
) -> ModelRouterScaffoldSummary:
    """Create a reviewable SFT workspace without starting training."""

    source_repo = source_repo.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    dataset_path = source_repo / "datasets" / "outcomes.jsonl"
    tasks_root = source_repo / "tasks"
    dataset_metadata_path = source_repo / "dataset.json"
    if not dataset_path.is_file():
        raise ValueError(f"missing model-router outcomes: {dataset_path}")
    if not tasks_root.is_dir():
        raise ValueError(f"missing model-router tasks: {tasks_root}")
    if len(models) < 2:
        raise ValueError("at least two --model values are required")
    if len(set(models)) != len(models):
        raise ValueError("--model values must be unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output workspace is not empty: {output_dir}")

    rows = _read_jsonl(dataset_path)
    routes = [_route_for_model(model, rows) for model in models]
    route_ids = tuple(route["route_id"] for route in routes)
    source_revision = _git_revision(source_repo)
    dataset_metadata = (
        _read_object(dataset_metadata_path)
        if dataset_metadata_path.is_file()
        else {}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": "1",
        "kind": "castform-router-sft",
        "status": "scaffolded",
        "source": {
            "repository": dataset_metadata.get("pipeline", {}).get(
                "repository",
                "castform-ai/model-router",
            ),
            "revision": source_revision,
            "dataset": "datasets/outcomes.jsonl",
            "dataset_sha256": hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest(),
            "task_count": dataset_metadata.get("task_count"),
            "trace_cutoff": dataset_metadata.get("traces", {}).get("cutoff"),
        },
        "candidate_routes": routes,
        "benchmark": {
            "split_strategy": "repo-temporal",
            "eval_ratio": eval_ratio,
            "visible_task_fields": [
                "instruction.md",
                "repository",
                "candidate_routes",
            ],
            "hidden_task_fields": [
                "solution",
                "tests/overlay",
                "difficulty",
                "quality_score",
            ],
        },
        "objective": {
            "method": "supervised_fine_tuning",
            "prediction": (
                "per-route completion probability and expected token classes"
            ),
            "selection": "deterministic Castform cost/quality policy",
        },
    }
    _write_json(manifest_path, manifest)

    formatted = format_benchmax_dataset(
        dataset_path,
        manifest_path=manifest_path,
        output_dir=output_dir / "router" / "data",
        eval_ratio=eval_ratio,
        tasks_root=tasks_root,
        split_strategy="repo-temporal",
    )
    _write_json(
        output_dir / "router" / "training_config.json",
        {
            "schema_version": "1",
            "method": "supervised_fine_tuning",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "adapter": "lora",
            "assistant_only_loss": True,
            "success_target": {
                "method": "beta_posterior_mean",
                "alpha": 1.0,
                "beta": 1.0,
            },
            "token_target": "categorical_bands",
            "token_band_representatives_file": (
                "router/contract/token_bands.json"
            ),
            "epochs": 3,
            "learning_rate": 0.0002,
            "max_sequence_length": 8192,
            "seed": 42,
            "train_file": "router/data/train.jsonl",
            "eval_file": "router/data/eval.jsonl",
            "output_dir": "router/checkpoints/qwen35-08b-sft-v2",
        },
    )
    contract_dir = output_dir / "router" / "contract"
    train_example = _representative_example(
        _read_jsonl(Path(formatted.train_path))
    )
    request_example_path = contract_dir / "request.example.json"
    response_example_path = contract_dir / "response.example.json"
    _write_json(request_example_path, train_example["request"])
    _write_json(response_example_path, train_example["target"])
    _write_json(
        contract_dir / "response.schema.json",
        model_response_json_schema(expected_route_ids=route_ids),
    )
    _write_json(
        contract_dir / "token_bands.json",
        {
            "schema_version": "1",
            "unit": "tokens",
            "conversion": "deterministic_representative",
            "input": token_band_representatives("input"),
            "cache_read": token_band_representatives("cache_read"),
            "output": token_band_representatives("output"),
        },
    )
    (output_dir / "README.md").write_text(
        _workspace_readme(
            source_repo=source_repo,
            source_revision=source_revision,
            models=models,
            complete_tasks=formatted.complete_tasks,
            train_examples=formatted.train_examples,
            eval_examples=formatted.eval_examples,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    return ModelRouterScaffoldSummary(
        workspace=str(output_dir),
        source_revision=source_revision,
        selected_models=models,
        complete_tasks=formatted.complete_tasks,
        skipped_incomplete_tasks=formatted.skipped_incomplete_tasks,
        train_examples=formatted.train_examples,
        eval_examples=formatted.eval_examples,
        manifest_path=str(manifest_path),
        request_example_path=str(request_example_path),
        response_example_path=str(response_example_path),
        next_command=f"castform-router train-sft {output_dir}",
    )


def _route_for_model(
    model: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    matching = [row for row in rows if row.get("route") == model]
    if not matching:
        raise ValueError(f"model has no outcomes in the source dataset: {model}")
    harnesses = {str(row.get("harness") or "") for row in matching}
    if "" in harnesses or len(harnesses) != 1:
        raise ValueError(
            f"model must resolve to exactly one non-empty harness: {model}"
        )
    harness = next(iter(harnesses))
    provider = _provider_for_route(model=model, harness=harness)
    return {
        "route_id": f"{harness}/{model}@{provider}",
        "harness": harness,
        "model": model,
        "provider": provider,
        "harbor_model": model,
        "gateway_model": model,
    }


def _provider_for_route(*, model: str, harness: str) -> str:
    # Provider identity is a property of the served model, not the agent
    # harness. DeepSeek uses Claude Code through its Anthropic-compatible API.
    if model.startswith("deepseek-"):
        return "deepseek"
    if harness == "claude-code" or model.startswith("claude-"):
        return "anthropic"
    if harness == "codex" or model.startswith(("gpt-", "o")):
        return "openai"
    raise ValueError(
        f"cannot infer provider for model={model!r}, harness={harness!r}"
    )


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _representative_example(
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    for example in examples:
        predictions = example.get("target", {}).get("predictions", [])
        probabilities = {
            prediction.get("success_probability")
            for prediction in predictions
            if isinstance(prediction, dict)
        }
        if len(probabilities) > 1:
            return example
    return examples[0]


def _workspace_readme(
    *,
    source_repo: Path,
    source_revision: str,
    models: tuple[str, ...],
    complete_tasks: int,
    train_examples: int,
    eval_examples: int,
    output_dir: Path,
) -> str:
    model_lines = "\n".join(f"- `{model}`" for model in models)
    return f"""# Castform multi-model router SFT

Source: `{source_repo}` at `{source_revision}`.

The learned model sees a task plus the eligible routes and predicts each
route's completion probability and expected token usage. It does not select a
winner; Castform's deterministic policy combines these predictions with live
cost and availability.

## Selected models

{model_lines}

## Frozen split

- complete tasks: {complete_tasks}
- repo-temporal train examples: {train_examples}
- repo-temporal eval examples: {eval_examples}

Review `manifest.json`, `router/contract/request.example.json`, and
`router/contract/response.example.json` before training. The deterministic
band-to-token conversion is recorded in `router/contract/token_bands.json`.

```bash
uv sync --extra training
uv run castform-router train-sft {output_dir}
```

Training spends GPU resources. Do not start it until the dataset and run
configuration have been reviewed.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain JSON objects")
    return values


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summary_json(summary: ModelRouterScaffoldSummary) -> str:
    """Serialize a scaffold summary for the CLI."""

    return json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True)
