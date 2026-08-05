"""Evaluate a served trained router and emit Benchmax-compatible picks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from castform_router.job_router import JobRouter
from castform_router.router_protocol import OpenAICompatibleRouteScorer
from castform_router.token_bands import (
    token_band_for_count,
    token_band_representative,
)
from castform_router.types import HarnessRoute, HarnessRouteRequest


def evaluate_trained_router(
    *,
    workspace: Path,
    base_url: str,
    model: str,
    api_key: str = "local",
    quality_threshold: float = 0.84,
) -> dict[str, Any]:
    """Run held-out requests, validate JSON, and write scoreboard picks."""

    manifest = _read_object(workspace / "manifest.json")
    costs = _read_object(workspace / "router" / "data" / "route_costs.json")
    examples = _read_jsonl(workspace / "router" / "data" / "eval.jsonl")
    route_specs = manifest.get("candidate_routes")
    if not isinstance(route_specs, list):
        raise ValueError("manifest candidate_routes must be an array")
    cost_table = costs.get("routes")
    if not isinstance(cost_table, dict):
        raise ValueError("route_costs.json routes must be an object")

    scorer = OpenAICompatibleRouteScorer(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=30,
    )
    routes = tuple(
        HarnessRoute(
            route_id=str(route["route_id"]),
            harness=str(route["harness"]),
            model=str(route["model"]),
            provider=str(route["provider"]),
            gateway_model=str(route.get("gateway_model") or route["model"]),
            estimated_cost_usd=float(cost_table.get(route["route_id"], 0)),
        )
        for route in route_specs
    )
    router = JobRouter(
        scorer=scorer,
        quality_threshold=quality_threshold,
        ttl_seconds=0,
    )
    specs_by_id = {str(route["route_id"]): route for route in route_specs}
    picks: list[dict[str, Any]] = []
    squared_errors: list[float] = []
    absolute_token_errors: list[float] = []
    token_band_results: dict[str, list[bool]] = {
        "input": [],
        "cache_read": [],
        "output": [],
    }
    for example in examples:
        request_payload = example["request"]
        task = request_payload["task"]
        decision = router.route(
            HarnessRouteRequest(
                request_id=str(request_payload["request_id"]),
                session_id=None,
                task_text=str(task["text"]),
                task_domain=str(task.get("domain") or "software_engineering"),
                user_context=dict(request_payload.get("user_context") or {}),
                workspace_context=dict(
                    request_payload.get("workspace_context") or {}
                ),
                candidate_routes=routes,
            )
        )
        selected_spec = specs_by_id[decision.selected_route.route_id]
        picks.append(
            {
                "task_id": str(example["example_id"]),
                "model": str(
                    selected_spec.get("harbor_model")
                    or selected_spec.get("model")
                ),
                "route_id": decision.selected_route.route_id,
                "reasoning": decision.reason,
                "router_cost_usd": 0.0,
            }
        )
        targets = {
            value["route_id"]: value
            for value in example["target"]["predictions"]
        }
        for prediction in decision.predictions:
            target = targets[prediction.route_id]
            squared_errors.append(
                (
                    prediction.success_probability
                    - float(target["success_probability"])
                )
                ** 2
            )
            absolute_token_errors.append(
                abs(
                    prediction.expected_total_tokens
                    - _target_total_tokens(target)
                )
            )
            if "input_token_band" in target:
                token_band_results["input"].append(
                    token_band_for_count(
                        prediction.expected_input_tokens,
                        "input",
                    )
                    == target["input_token_band"]
                )
                token_band_results["cache_read"].append(
                    token_band_for_count(
                        prediction.expected_cache_read_tokens,
                        "cache_read",
                    )
                    == target["cache_read_token_band"]
                )
                token_band_results["output"].append(
                    token_band_for_count(
                        prediction.expected_output_tokens,
                        "output",
                    )
                    == target["output_token_band"]
                )

    output_dir = workspace / "benchmax" / "model_router" / "router_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    picks_path = output_dir / "picks_trained.jsonl"
    _write_jsonl(picks_path, picks)
    report = {
        "schema_version": "1",
        "router_model_version": scorer.router_model_version,
        "quality_threshold": quality_threshold,
        "held_out_tasks": len(examples),
        "prediction_count": len(squared_errors),
        "brier_score": (
            round(sum(squared_errors) / len(squared_errors), 6)
            if squared_errors
            else None
        ),
        "mean_absolute_total_token_error": (
            round(sum(absolute_token_errors) / len(absolute_token_errors), 2)
            if absolute_token_errors
            else None
        ),
        "token_band_accuracy": _mean_bool(
            [
                result
                for results in token_band_results.values()
                for result in results
            ]
        ),
        "token_band_accuracy_by_class": {
            token_class: _mean_bool(results)
            for token_class, results in token_band_results.items()
        },
        "picks": str(picks_path),
        "next_command": (
            "python scoreboard.py dataset.jsonl --split test "
            f"--picks {picks_path}"
        ),
    }
    report_path = workspace / "router" / "reports" / "trained-evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(report_path)
    return report


def _target_total_tokens(target: dict[str, Any]) -> int:
    if "input_token_band" in target:
        return (
            token_band_representative(target["input_token_band"], "input")
            + token_band_representative(
                target["cache_read_token_band"],
                "cache_read",
            )
            + token_band_representative(
                target["output_token_band"],
                "output",
            )
        )
    return sum(
        int(target.get(field, 0))
        for field in (
            "expected_input_tokens",
            "expected_cache_read_tokens",
            "expected_output_tokens",
        )
    )


def _mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"missing required file: {path}") from error
    values = [json.loads(line) for line in lines if line.strip()]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain JSON objects")
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
