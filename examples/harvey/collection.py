"""Collection and deterministic export helpers for Harvey AutoCompact SFT."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def enrich_records(
    trajectory: Mapping[str, Any],
    *,
    example_id: str,
    trace_id: str,
    rewards: Mapping[str, float],
    termination_reason: str,
) -> list[dict[str, Any]]:
    """Attach trial identity/outcome and validate call-level loss boundaries."""

    records: list[dict[str, Any]] = []
    raw_records = trajectory.get("sft_records", [])
    if not isinstance(raw_records, list):
        raise ValueError("trajectory.sft_records must be a list")
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("SFT record must be an object")
        record = json.loads(json.dumps(raw))
        prompts = record.get("prompt_messages")
        completions = record.get("completion_messages")
        if not isinstance(prompts, list) or not isinstance(completions, list):
            raise ValueError("SFT record requires prompt_messages and completion_messages")
        if len(completions) != 1 or completions[0].get("role") != "assistant":
            raise ValueError("SFT record must have exactly one assistant completion")
        if completions[0].get("step_loss_mask") != 1:
            raise ValueError("SFT completion must have step_loss_mask=1")
        if any(
            message.get("role") == "assistant" and message.get("step_loss_mask") != 0
            for message in prompts
        ):
            raise ValueError("every prompt assistant message must have step_loss_mask=0")
        record["id"] = f"{trace_id}:{record['id']}"
        record["example_id"] = example_id
        record["trace_id"] = trace_id
        task = dict(record.get("task") or {})
        task.update(
            {
                "terminal_rewards": {str(key): float(value) for key, value in rewards.items()},
                "termination_reason": termination_reason,
            }
        )
        record["task"] = task
        records.append(record)
    expected_categories = Counter(
        {
            "autocompact_trigger": 1,
            "autocompact_summary": 1,
            "autocompact_continuation": 1,
        }
    )
    categories_by_event: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for record in records:
        task = record.get("task") or {}
        event_id = task.get("compaction_event_id")
        if not isinstance(event_id, int):
            raise ValueError("SFT record requires an integer compaction_event_id")
        categories_by_event[event_id][str(record.get("category"))] += 1
    if any(categories != expected_categories for categories in categories_by_event.values()):
        raise ValueError("each compaction must export one trigger, summary, and continuation")
    return records


def aggregate_shards(
    output_dir: Path,
    *,
    split_seed: str = "harvey-autocompact-v1",
) -> dict[str, Any]:
    """Build task-disjoint default and passed-only train/eval JSONL files."""

    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for shard in sorted((output_dir / "shards").glob("*.jsonl")):
        rows.extend(_read_jsonl(shard))

    task_ids = sorted(
        {str(row["example_id"]) for row in rows},
        key=lambda value: _split_key(value, split_seed),
    )
    eval_count = max(1, round(len(task_ids) * 0.1)) if len(task_ids) > 1 else 0
    eval_ids = set(task_ids[:eval_count])
    rows.sort(key=lambda row: str(row["id"]))

    outputs = {
        "train": [row for row in rows if row["example_id"] not in eval_ids],
        "eval": [row for row in rows if row["example_id"] in eval_ids],
    }
    passed = {
        split: [row for row in split_rows if _is_passed(row)]
        for split, split_rows in outputs.items()
    }
    _write_jsonl_atomic(output_dir / "train.jsonl", outputs["train"])
    _write_jsonl_atomic(output_dir / "eval.jsonl", outputs["eval"])
    _write_jsonl_atomic(output_dir / "passed" / "train.jsonl", passed["train"])
    _write_jsonl_atomic(output_dir / "passed" / "eval.jsonl", passed["eval"])

    outcomes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "raw").glob("*/outcome.json"))
    ]
    records_by_trace = Counter(str(row["trace_id"]) for row in rows)
    filter_reasons = Counter()
    for outcome in outcomes:
        trace_id = str(outcome.get("trace_id", ""))
        if records_by_trace[trace_id]:
            filter_reasons["accepted"] += 1
        elif not outcome.get("rewards"):
            filter_reasons["unscorable"] += 1
        else:
            filter_reasons["no_complete_compaction"] += 1

    manifest = {
        "schema_version": 1,
        "split_seed": split_seed,
        "task_count": len(task_ids),
        "task_id_digest": hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest(),
        "eval_task_count": len(eval_ids),
        "outcome_count": len(outcomes),
        "record_count": len(rows),
        "passed_record_count": sum(len(value) for value in passed.values()),
        "counts_by_category": dict(Counter(str(row["category"]) for row in rows)),
        "counts_by_split": {key: len(value) for key, value in outputs.items()},
        "passed_counts_by_split": {key: len(value) for key, value in passed.items()},
        "termination_counts": dict(
            Counter(str(outcome.get("termination_reason", "unknown")) for outcome in outcomes)
        ),
        "filter_counts": dict(filter_reasons),
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


async def collect_trajectories(
    env: Any,
    *,
    output_dir: Path,
    model: str,
    base_url: str,
    model_auth: Any,
    max_examples: int | None,
    rollouts_per_task: int,
    max_concurrent_tasks: int,
    max_compactions: int,
    resume: bool,
) -> dict[str, Any]:
    """Collect the Harbor train split and materialize scored SFT shards."""

    from benchmax.envs import RolloutRequest

    if rollouts_per_task < 1 or max_concurrent_tasks < 1:
        raise ValueError("rollouts_per_task and max_concurrent_tasks must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = await env.create_dataset(
        "train",
        output_dir / "dataset-snapshot",
        max_examples=max_examples,
    )
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def collect_example(example: Any) -> None:
        async with semaphore:
            requests = []
            for rollout_index in range(rollouts_per_task):
                rollout_id = _rollout_id(str(example.id), rollout_index)
                shard = output_dir / "shards" / f"{rollout_id}.jsonl"
                if resume and shard.is_file():
                    try:
                        _read_jsonl(shard)
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        pass
                    else:
                        continue
                requests.append(
                    RolloutRequest(
                        rollout_id=rollout_id,
                        example=example,
                        model=model,
                        base_url=base_url,
                        model_auth=model_auth,
                        split="train",
                    )
                )
            if not requests:
                return
            outcomes = await env.run_group(requests)
            for request in requests:
                outcome = outcomes[request.rollout_id]
                rewards = dict(outcome.rewards)
                source = _trajectory_path(output_dir / "trials" / request.rollout_id)
                raw_dir = output_dir / "raw" / request.rollout_id
                raw_dir.mkdir(parents=True, exist_ok=True)
                outcome_payload = {
                    "example_id": str(example.id),
                    "trace_id": request.rollout_id,
                    "rewards": rewards,
                    "termination_reason": outcome.termination_reason,
                    "error": outcome.error,
                }
                _write_json_atomic(raw_dir / "outcome.json", outcome_payload)
                if source is None:
                    _write_jsonl_atomic(output_dir / "shards" / f"{request.rollout_id}.jsonl", [])
                    continue
                trajectory = json.loads(source.read_text(encoding="utf-8"))
                enriched = (
                    enrich_records(
                        trajectory,
                        example_id=str(example.id),
                        trace_id=request.rollout_id,
                        rewards=rewards,
                        termination_reason=outcome.termination_reason,
                    )
                    if rewards
                    else []
                )
                _write_json_atomic(raw_dir / "trajectory.json", trajectory)
                _write_jsonl_atomic(
                    output_dir / "shards" / f"{request.rollout_id}.jsonl",
                    enriched,
                )

    await asyncio.gather(*(collect_example(example) for example in dataset))
    manifest = aggregate_shards(output_dir)
    manifest["collection"] = {
        "model": model,
        "base_url": base_url,
        "dataset_size": len(dataset),
        "max_examples": max_examples,
        "rollouts_per_task": rollouts_per_task,
        "max_concurrent_tasks": max_concurrent_tasks,
        "max_compactions": max_compactions,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def _trajectory_path(trial_dir: Path) -> Path | None:
    matches = sorted(Path(trial_dir).rglob("autocompact-trajectory.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple AutoCompact trajectories found under {trial_dir}")
    return matches[0]


def _rollout_id(example_id: str, rollout_index: int) -> str:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:16]
    return f"harvey-{digest}-r{rollout_index}"


def _split_key(example_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{example_id}".encode()).hexdigest()


def _is_passed(row: Mapping[str, Any]) -> bool:
    rewards = row.get("task", {}).get("terminal_rewards", {})
    if "reward" in rewards:
        return float(rewards["reward"]) == 1.0
    return bool(rewards) and all(float(value) == 1.0 for value in rewards.values())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["aggregate_shards", "collect_trajectories", "enrich_records"]
