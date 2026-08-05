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
from castform_router.token_bands import token_band_for_count

_BETA_PRIOR_ALPHA = 1.0
_BETA_PRIOR_BETA = 1.0


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
    tasks_root: Path | None = None,
    split_strategy: str = "global-temporal",
) -> FormatSummary:
    """Aggregate repeated task-route trials into full-matrix chat examples."""

    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")
    if split_strategy not in {"global-temporal", "repo-temporal"}:
        raise ValueError(
            "split_strategy must be 'global-temporal' or 'repo-temporal'"
        )
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
        route_evidence: list[dict[str, Any]] = []
        for route in routes:
            matching = [
                row
                for row in task_rows
                if _row_matches_route(row, route)
            ]
            if not matching:
                break
            prediction, evidence = _aggregate_route(route, matching)
            predictions.append(prediction)
            route_evidence.append(evidence)
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
                "task_text": _task_text(
                    first,
                    dataset_path,
                    tasks_root=tasks_root,
                ),
                "predictions": predictions,
                "route_evidence": route_evidence,
            }
        )

    splits = _assign_splits(
        task_records,
        eval_ratio=eval_ratio,
        held_out_repositories=set(held_out_repositories),
        split_strategy=split_strategy,
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
        "router_model_version": "qwen35-08b-sft-v2",
        "predictions": record["predictions"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": record["task_id"],
        "split": split,
        "request": request,
        "target": target,
        # Audit-only evidence is deliberately outside ``messages`` so the
        # model learns the smoothed targets without being asked to hallucinate
        # how many attempts were collected for a future task.
        "label_metadata": {
            "success_smoothing": {
                "method": "beta_posterior_mean",
                "alpha": _BETA_PRIOR_ALPHA,
                "beta": _BETA_PRIOR_BETA,
            },
            "route_evidence": record["route_evidence"],
        },
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    rewards = [_probability(row.get("reward"), "reward") for row in rows]
    attempts = len(rewards)
    reward_sum = sum(rewards)
    input_tokens = _mean_int(
        row.get("n_input_tokens", row.get("input_tokens", 0))
        for row in rows
    )
    cache_tokens = _mean_int(
        row.get("n_cache_tokens", row.get("cache_read_tokens", 0))
        for row in rows
    )
    output_tokens = _mean_int(
        row.get("n_output_tokens", row.get("output_tokens", 0))
        for row in rows
    )
    prediction = {
        "route_id": str(route["route_id"]),
        "success_probability": round(
            (reward_sum + _BETA_PRIOR_ALPHA)
            / (attempts + _BETA_PRIOR_ALPHA + _BETA_PRIOR_BETA),
            4,
        ),
        "input_token_band": token_band_for_count(input_tokens, "input"),
        "cache_read_token_band": token_band_for_count(
            cache_tokens,
            "cache_read",
        ),
        "output_token_band": token_band_for_count(output_tokens, "output"),
    }
    evidence = {
        "route_id": str(route["route_id"]),
        "attempts": attempts,
        "reward_sum": round(reward_sum, 4),
        "observed_success_rate": round(reward_sum / attempts, 4),
        "mean_input_tokens": input_tokens,
        "mean_cache_read_tokens": cache_tokens,
        "mean_output_tokens": output_tokens,
    }
    return prediction, evidence


def _assign_splits(
    records: list[dict[str, Any]],
    *,
    eval_ratio: float,
    held_out_repositories: set[str],
    split_strategy: str,
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

    if split_strategy == "repo-temporal":
        repositories = sorted({record["repository"] for record in remaining})
        for repository in repositories:
            group = sorted(
                (
                    record
                    for record in remaining
                    if record["repository"] == repository
                ),
                key=lambda record: (record["merged_at"], record["task_id"]),
            )
            _assign_temporal_group(
                group,
                assigned=assigned,
                eval_ratio=eval_ratio,
                prefer_train_remainder=True,
            )
    else:
        remaining.sort(
            key=lambda record: (record["merged_at"], record["task_id"])
        )
        _assign_temporal_group(
            remaining,
            assigned=assigned,
            eval_ratio=eval_ratio,
        )
    return assigned


def _assign_temporal_group(
    records: list[dict[str, Any]],
    *,
    assigned: dict[str, str],
    eval_ratio: float,
    prefer_train_remainder: bool = False,
) -> None:
    if not records:
        return
    if len(records) == 1:
        assigned[records[0]["task_id"]] = "train"
        return
    if prefer_train_remainder:
        cutoff = min(
            len(records) - 1,
            max(1, math.ceil(len(records) * (1 - eval_ratio))),
        )
    else:
        eval_count = min(
            len(records) - 1,
            max(1, math.ceil(len(records) * eval_ratio)),
        )
        cutoff = len(records) - eval_count
    for index, record in enumerate(records):
        assigned[record["task_id"]] = (
            "train" if index < cutoff else "eval"
        )


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


def _task_text(
    row: dict[str, Any],
    dataset_path: Path,
    *,
    tasks_root: Path | None,
) -> str:
    direct = row.get("task_text") or row.get("instruction")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    task_dir = row.get("task_dir")
    if isinstance(task_dir, str) and task_dir:
        relative = Path(task_dir)
        candidates: list[Path] = []
        if relative.is_absolute():
            candidates.append(relative)
        else:
            candidates.append(dataset_path.parent / relative)
            if tasks_root is not None:
                candidates.extend(
                    [
                        Path(tasks_root) / relative.name,
                        Path(tasks_root) / str(row.get("task_id") or ""),
                    ]
                )
        for candidate in candidates:
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


def _probability(value: object, field: str) -> float:
    normalized = _number(value, field)
    if normalized > 1:
        raise ValueError(f"{field} must be at most 1")
    return normalized


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
