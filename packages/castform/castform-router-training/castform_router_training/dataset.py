"""Build strict Qwen scorer examples from measured backend outcomes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from castform_router_training.project import Project

SYSTEM_PROMPT = (
    "Score every candidate backend for its probability of successfully completing the task. "
    "Score each backend exactly once. Do not select a backend or use cost. "
    "Return only JSON matching the supplied schema."
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

    Outcome rows have ``task_id``, ``backend``, and boolean ``success`` fields.
    Every retained task must have at least one measured rollout for every backend.
    """

    tasks = {str(row.get("task_id")): row for row in _read_jsonl(tasks_path)}
    backend_names = {backend.name for backend in project.backends}
    outcomes: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_jsonl(outcomes_path):
        task_id, backend, success = row.get("task_id"), row.get("backend"), row.get("success")
        if task_id not in tasks or backend not in backend_names or not isinstance(success, bool):
            raise ValueError(f"invalid outcome row: {row}")
        outcomes[str(task_id)][str(backend)].append(success)

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
            response = {
                "scorer_version": "training-target",
                "predictions": [
                    {
                        "backend": backend.name,
                        "success_probability": sum(measurements[backend.name])
                        / len(measurements[backend.name]),
                    }
                    for backend in project.backends
                ],
            }
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(request, sort_keys=True)},
                    {"role": "assistant", "content": json.dumps(response, sort_keys=True)},
                ]
            }
            stream.write(json.dumps(example, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError("no task has measured outcomes for every backend")
    return count
