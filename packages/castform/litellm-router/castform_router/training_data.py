"""Convert Benchmax's audited outcome table into leak-safe Qwen SFT examples."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from castform_router.router_protocol import SCHEMA_VERSION, SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class FormatSummary:
    input_rows: int
    complete_tasks: int
    skipped_incomplete_tasks: int
    train_examples: int
    eval_examples: int
    train_path: str
    eval_path: str
    route_costs_path: str


def format_benchmax_dataset(
    dataset_path: Path,
    *,
    manifest_path: Path,
    output_dir: Path,
    eval_ratio: float = 0.2,
    held_out_repositories: Iterable[str] = (),
) -> FormatSummary:
    """Aggregate repeated task-route trials into full-matrix chat examples."""

    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")
    rows = _read_jsonl(dataset_path)
    manifest = _read_object(manifest_path)
    routes = manifest.get("candidate_routes")
    if not isinstance(routes, list) or len(routes) < 2:
        raise ValueError("manifest must contain at least two candidate_routes")
    route_ids = tuple(str(route["route_id"]) for route in routes)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id:
            grouped[task_id].append(row)

    task_records: list[dict[str, Any]] = []
    skipped = 0
    for task_id, task_rows in grouped.items():
        predictions: list[dict[str, Any]] = []
        for route in routes:
            matching = [
                row
                for row in task_rows
                if _row_matches_route(row, route)
            ]
            if not matching:
                break
            predictions.append(_aggregate_route(route, matching))
        if len(predictions) != len(route_ids):
            skipped += 1
            continue

        first = task_rows[0]
        repository = str(
            first.get("repo")
            or first.get("repository")
            or _repository_from_task_id(task_id)
        )
        task_records.append(
            {
                "task_id": task_id,
                "repository": repository,
                "merged_at": str(first.get("merged_at") or ""),
                "explicit_split": first.get("split"),
                "task_text": _task_text(first, dataset_path),
                "predictions": predictions,
            }
        )

    splits = _assign_splits(
        task_records,
        eval_ratio=eval_ratio,
        held_out_repositories=set(held_out_repositories),
    )
    candidates = [
        {
            "route_id": str(route["route_id"]),
            "harness": str(route["harness"]),
            "model": str(route["model"]),
            "provider": str(route["provider"]),
        }
        for route in routes
    ]
    examples = [
        _training_example(record, candidates, splits[record["task_id"]])
        for record in task_records
    ]
    train = [example for example in examples if example["split"] == "train"]
    evaluation = [example for example in examples if example["split"] == "eval"]
    if not train:
        raise ValueError("formatter produced no training examples")
    if not evaluation:
        raise ValueError("formatter produced no evaluation examples")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(eval_path, evaluation)
    route_costs_path = output_dir / "route_costs.json"
    _write_json(
        route_costs_path,
        {
            "schema_version": SCHEMA_VERSION,
            "source": "mean_cost_usd_on_train_split",
            "routes": {
                route_id: _mean_route_cost(
                    rows,
                    route,
                    train_task_ids={
                        example["example_id"] for example in train
                    },
                )
                for route_id, route in zip(route_ids, routes, strict=True)
            },
        },
    )
    return FormatSummary(
        input_rows=len(rows),
        complete_tasks=len(examples),
        skipped_incomplete_tasks=skipped,
        train_examples=len(train),
        eval_examples=len(evaluation),
        train_path=str(train_path),
        eval_path=str(eval_path),
        route_costs_path=str(route_costs_path),
    )


def _training_example(
    record: dict[str, Any],
    candidates: list[dict[str, str]],
    split: str,
) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "request_id": record["task_id"],
        "task": {
            "text": record["task_text"],
            "domain": "software_engineering",
        },
        # Repository-mined data has no reliable persona label. Keep this empty
        # until the same user context exists in both training and production.
        "user_context": {},
        "workspace_context": {
            "repository": record["repository"],
            "tools": ["repository", "tests"],
        },
        "candidate_routes": candidates,
    }
    target = {
        "schema_version": SCHEMA_VERSION,
        "router_model_version": "qwen35-08b-sft-v1",
        "predictions": record["predictions"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": record["task_id"],
        "split": split,
        "request": request,
        "target": target,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    request,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    target,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
    }


def _aggregate_route(
    route: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    success = [_number(row.get("reward"), "reward") for row in rows]
    return {
        "route_id": str(route["route_id"]),
        "success_probability": round(sum(success) / len(success), 4),
        "expected_input_tokens": _mean_int(
            row.get("n_input_tokens", row.get("input_tokens", 0))
            for row in rows
        ),
        "expected_cache_read_tokens": _mean_int(
            row.get("n_cache_tokens", row.get("cache_read_tokens", 0))
            for row in rows
        ),
        "expected_output_tokens": _mean_int(
            row.get("n_output_tokens", row.get("output_tokens", 0))
            for row in rows
        ),
    }


def _assign_splits(
    records: list[dict[str, Any]],
    *,
    eval_ratio: float,
    held_out_repositories: set[str],
) -> dict[str, str]:
    assigned: dict[str, str] = {}
    remaining: list[dict[str, Any]] = []
    for record in records:
        explicit = record.get("explicit_split")
        if explicit in {"test", "eval", "validation"}:
            assigned[record["task_id"]] = "eval"
        elif explicit == "train":
            assigned[record["task_id"]] = "train"
        elif record["repository"] in held_out_repositories:
            assigned[record["task_id"]] = "eval"
        else:
            remaining.append(record)

    remaining.sort(
        key=lambda record: (record["merged_at"], record["task_id"])
    )
    if remaining:
        eval_count = max(1, math.ceil(len(remaining) * eval_ratio))
        cutoff = max(1, len(remaining) - eval_count)
        for index, record in enumerate(remaining):
            assigned[record["task_id"]] = (
                "train" if index < cutoff else "eval"
            )
    return assigned


def _row_matches_route(row: dict[str, Any], route: dict[str, Any]) -> bool:
    row_route = str(row.get("route") or row.get("route_id") or "")
    row_model = str(row.get("model") or "")
    row_harness = str(row.get("harness") or "")
    route_id = str(route["route_id"])
    route_values = {
        route_id,
        str(route.get("model") or ""),
        str(route.get("harbor_model") or ""),
        _safe_name(route_id),
    }
    if row_route in route_values or row_model in route_values:
        expected_harness = str(route.get("harness") or "")
        return not row_harness or row_harness == expected_harness
    return False


def _task_text(row: dict[str, Any], dataset_path: Path) -> str:
    direct = row.get("task_text") or row.get("instruction")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    task_dir = row.get("task_dir")
    if isinstance(task_dir, str) and task_dir:
        candidate = Path(task_dir)
        if not candidate.is_absolute():
            candidate = dataset_path.parent / candidate
        for name in ("instruction.md", "task.md", "prompt.md"):
            path = candidate / name
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()
    raise ValueError(
        f"task {row.get('task_id')!r} has no task_text or readable instruction.md"
    )


def _repository_from_task_id(task_id: str) -> str:
    parts = task_id.split("-pr-", 1)
    return parts[0].replace("-", "/", 1) if parts else "unknown"


def _mean_int(values: Iterable[object]) -> int:
    normalized = [_number(value, "token count") for value in values]
    return round(sum(normalized) / len(normalized)) if normalized else 0


def _mean_route_cost(
    rows: list[dict[str, Any]],
    route: dict[str, Any],
    *,
    train_task_ids: set[str],
) -> float:
    costs = [
        _number(row["cost_usd"], "cost_usd")
        for row in rows
        if row.get("task_id") in train_task_ids
        and _row_matches_route(row, route)
        and row.get("cost_usd") is not None
    ]
    return round(sum(costs) / len(costs), 6) if costs else 0.0


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    if float(value) < 0:
        raise ValueError(f"{field} must be non-negative")
    return float(value)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-"
        for character in value.lower()
    ).strip("-")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"missing Benchmax dataset: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"Benchmax dataset is empty: {path}")
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing workspace manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
