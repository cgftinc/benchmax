"""Build strict Qwen scorer examples from measured backend outcomes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from castform_router_training.project import Project
from castform_router_training.token_bands import token_band_for_count

SYSTEM_PROMPT = (
    "Score every candidate backend for its probability of successfully completing the task "
    "and its expected input, cache-read, and output token bands. Score each backend exactly "
    "once. Do not select a backend or use cost. Return only JSON matching the supplied schema."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL: {path}") from error
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"every row must be an object: {path}")
    return values


def build_dataset(
    project: Project, tasks_path: Path, outcomes_path: Path, output_path: Path
) -> int:
    """Join tasks with repeated rollout results and write chat-format SFT JSONL.

    Outcome rows have ``task_id``, ``backend``, boolean ``success``, and
    non-negative ``input_tokens``, ``cache_read_tokens``, and ``output_tokens``.
    Every retained task must have at least one measured rollout for every backend.
    """

    tasks = {str(row.get("task_id")): row for row in _read_jsonl(tasks_path)}
    backend_names = {backend.name for backend in project.backends}
    outcomes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_jsonl(outcomes_path):
        task_id, backend, success = row.get("task_id"), row.get("backend"), row.get("success")
        token_values = tuple(
            row.get(field) for field in ("input_tokens", "cache_read_tokens", "output_tokens")
        )
        if (
            task_id not in tasks
            or backend not in backend_names
            or not isinstance(success, bool)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in token_values
            )
        ):
            raise ValueError(f"invalid outcome row: {row}")
        outcomes[str(task_id)][str(backend)].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for task_id, task in tasks.items():
            measurements = outcomes.get(task_id, {})
            if set(measurements) != backend_names:
                continue
            request = {
                "request_id": task_id,
                "task": str(task.get("task") or ""),
                "backends": [
                    {"name": backend.name, "model": backend.model, "provider": backend.provider}
                    for backend in project.backends
                ],
            }
            predictions = []
            evidence = []
            for backend in project.backends:
                attempts = measurements[backend.name]
                successes = sum(bool(row["success"]) for row in attempts)
                means = {
                    field: round(sum(int(row[field]) for row in attempts) / len(attempts))
                    for field in (
                        "input_tokens",
                        "cache_read_tokens",
                        "output_tokens",
                    )
                }
                predictions.append(
                    {
                        "backend": backend.name,
                        "success_probability": round((successes + 1) / (len(attempts) + 2), 4),
                        "input_token_band": token_band_for_count(means["input_tokens"], "input"),
                        "cache_read_token_band": token_band_for_count(
                            means["cache_read_tokens"], "cache_read"
                        ),
                        "output_token_band": token_band_for_count(means["output_tokens"], "output"),
                    }
                )
                evidence.append(
                    {
                        "backend": backend.name,
                        "attempts": len(attempts),
                        "successes": successes,
                        **{f"mean_{key}": value for key, value in means.items()},
                    }
                )
            response = {
                "scorer_version": "training-target",
                "predictions": predictions,
            }
            example = {
                "label_metadata": {
                    "success_smoothing": {
                        "method": "beta_posterior_mean",
                        "alpha": 1,
                        "beta": 1,
                    },
                    "backend_evidence": evidence,
                },
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(request, sort_keys=True)},
                    {"role": "assistant", "content": json.dumps(response, sort_keys=True)},
                ],
            }
            stream.write(json.dumps(example, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError("no task has measured outcomes for every backend")
    return count
