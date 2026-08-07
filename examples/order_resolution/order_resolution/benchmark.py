"""Frozen baseline matrix, metrics, and headroom decision."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import re
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from benchmax.envs import BaseRollout, Example, JsonRow, RolloutRequest, canonical_example_id
from castform import config
from castform.model_auth import CastformModelAuth, create_openai_client

from order_resolution import benchmark_spec as spec
from order_resolution.benchmark_spec import verify_predecessor
from order_resolution.branching import (
    NeonApi,
    RuntimeBranch,
    read_project_manifest,
    resolve_neon_api_key,
)
from order_resolution.command_codes import EnvelopeCode
from order_resolution.fixtures import check_v2_data, generate_v2_data
from order_resolution.hosting import (
    CANONICAL_RUNTIME_DEPENDENCIES,
    build_environment_bundle,
    inspect_environment_bundle,
)
from order_resolution.order_env import TOOLS, OrderResolutionEnv, world_id_for_rollout
from order_resolution.policy import render_system_contract, reply_tool_schema

SMALL_MODEL = "qwen3.5-4b"
FRONTIER_MODELS = ("gpt-5.6-sol", "grok-4.3")
TRAINING_GROUP_SIZE = 8
CONCURRENT_GROUPS = 16
STRESS_REPEATS = 3
FULL_REPORT_FILENAME = "report.html"
ORACLE_DEMO_IDS = (
    "train-cancel_item-execute-00",
    "train-replace_variant-deny-00",
)
#: Envelope codes that mark a tool call the model got wrong, including the v2
#: reply-contract rejection. No v1 transcript contains the newer code.
INVALID_RESULT_CODES = frozenset(str(code) for code in EnvelopeCode)
SCORABLE_TERMINATIONS = frozenset(
    {
        "finished",
        "context_exceeded",
        "output_exceeded",
        "max_turns_exceeded",
        "tool_budget_exceeded",
    }
)


class ObservedOrderResolutionEnv(OrderResolutionEnv):
    """Production environment loop with evaluation-only transcript capture."""

    def __init__(self, runtime_database_url: str) -> None:
        super().__init__(runtime_database_url)
        self._observations: dict[str, dict[str, Any]] = {}

    async def run_rollout(self, request: RolloutRequest[JsonRow]) -> BaseRollout:
        started = time.perf_counter()
        rollout = await super().run_rollout(request)
        self._observations[request.rollout_id] = {
            "latency_seconds": time.perf_counter() - started,
            "messages": json.loads(json.dumps(rollout.messages)),
        }
        return rollout

    def pop_observation(self, rollout_id: str) -> dict[str, Any]:
        try:
            return self._observations.pop(rollout_id)
        except KeyError as error:
            raise RuntimeError(f"missing observation for rollout {rollout_id!r}") from error


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any], forbidden: Sequence[str]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _assert_secret_safe(serialized, forbidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _assert_secret_safe(serialized: str, forbidden: Sequence[str]) -> None:
    if any(secret and secret in serialized for secret in forbidden):
        raise RuntimeError("refusing to persist a secret-bearing benchmark artifact")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path.name} must contain JSON objects")
    return rows


def load_oracle_demos(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {row["task_id"]: row for row in _load_jsonl(path)}
    missing = [task_id for task_id in ORACLE_DEMO_IDS if task_id not in by_id]
    if missing:
        raise RuntimeError(f"missing frozen oracle demos: {', '.join(missing)}")
    return tuple(by_id[task_id] for task_id in ORACLE_DEMO_IDS)  # type: ignore[return-value]


def build_two_shot_example(
    example: Example[JsonRow], demos: Sequence[Mapping[str, Any]]
) -> Example[JsonRow]:
    """Keep the production system prompt and prepend two frozen tool traces."""

    if len(demos) != 2:
        raise ValueError("two-shot prompting requires exactly two oracle demonstrations")
    target_messages = [dict(message) for message in example.payload["prompt_messages"]]
    system_messages = [message for message in target_messages if message.get("role") == "system"]
    if len(system_messages) != 1:
        raise ValueError("the production prompt must contain exactly one system message")
    messages: list[dict[str, Any]] = [system_messages[0]]
    for demo in demos:
        messages.extend(
            dict(message) for message in demo["prompt_messages"] if message.get("role") != "system"
        )
        messages.extend(dict(message) for message in demo["completion_messages"])
    versions = {
        demo.get("benchmark_id") for demo in demos
    } | {example.payload.get("benchmark_id")}
    if len(versions) != 1:
        raise ValueError("two-shot prompting cannot mix benchmark versions")
    messages.extend(message for message in target_messages if message.get("role") != "system")
    payload: JsonRow = {**example.payload, "prompt_messages": messages}
    # The augmented payload is a different example; recompute its canonical id
    # so a two-shot rollout can never be attributed to the production prompt.
    return Example(id=canonical_example_id(payload), payload=payload)


def freeze_task_selection(
    eval_examples: Sequence[Example[JsonRow]],
    train_examples: Sequence[Example[JsonRow]],
) -> dict[str, list[str]]:
    """Select all evaluation tasks and balanced fixed stress/probe/demo subsets."""

    eval_by_cell = _by_cell(eval_examples)
    train_by_cell = _by_cell(train_examples)
    if len(eval_by_cell) != 9 or len(train_by_cell) != 9:
        raise RuntimeError("the frozen task grid must contain exactly nine cells")
    if any(len(rows) != 10 for rows in eval_by_cell.values()):
        raise RuntimeError("each evaluation cell must contain exactly ten tasks")
    if any(len(rows) != 20 for rows in train_by_cell.values()):
        raise RuntimeError("each training cell must contain exactly twenty tasks")

    stress = [
        row.payload["task_id"] for cell in sorted(eval_by_cell) for row in eval_by_cell[cell][:3]
    ]
    probe = [
        row.payload["task_id"] for cell in sorted(train_by_cell) for row in train_by_cell[cell][:4]
    ]
    demo_cells = (
        "cancel_item-execute",
        "cancel_item-clarify",
        "change_address-execute",
        "change_address-deny",
        "replace_variant-execute",
        "replace_variant-clarify",
    )
    demos = [eval_by_cell[cell][0].payload["task_id"] for cell in demo_cells]
    return {
        "eval_task_ids": [example.payload["task_id"] for example in eval_examples],
        "stress_task_ids": stress,
        "signal_probe_task_ids": probe,
        "report_demo_task_ids": demos,
        "oracle_demo_task_ids": list(ORACLE_DEMO_IDS),
    }


def _by_cell(examples: Sequence[Example[JsonRow]]) -> dict[str, list[Example[JsonRow]]]:
    result: dict[str, list[Example[JsonRow]]] = defaultdict(list)
    for example in examples:
        result[str(example.payload["cell"])].append(example)
    return {
        cell: sorted(rows, key=lambda row: row.payload["task_id"]) for cell, rows in result.items()
    }


def _schema_head(example_root: Path) -> str:
    config_path = example_root / "alembic.ini"
    alembic_config = Config(str(config_path))
    alembic_config.set_main_option("script_location", str(example_root / "migrations"))
    head = ScriptDirectory.from_config(alembic_config).get_current_head()
    if head is None:
        raise RuntimeError("Alembic schema has no current head")
    return head


def resolve_model_catalog(base_url: str) -> list[str]:
    client = create_openai_client(
        model="catalog",
        base_url=base_url,
        auth=CastformModelAuth(),
        request_id="order-resolution-model-catalog",
    )
    try:
        return sorted(model.id for model in client.models.list().data)
    finally:
        client.close()


def _arm_specs(base_url: str) -> list[dict[str, str]]:
    return [
        {
            "id": "small_base",
            "model": SMALL_MODEL,
            "endpoint": base_url,
            "prompt": "production",
        },
        {
            "id": "small_two_shot",
            "model": SMALL_MODEL,
            "endpoint": base_url,
            "prompt": "two_oracle_examples",
        },
        {
            "id": "frontier_gpt",
            "model": FRONTIER_MODELS[0],
            "endpoint": base_url,
            "prompt": "production",
        },
        {
            "id": "frontier_grok",
            "model": FRONTIER_MODELS[1],
            "endpoint": base_url,
            "prompt": "production",
        },
    ]


def build_frozen_manifest(
    *,
    example_root: Path,
    branch: RuntimeBranch,
    eval_examples: Sequence[Example[JsonRow]],
    train_examples: Sequence[Example[JsonRow]],
    demos: Sequence[Mapping[str, Any]],
    available_models: Sequence[str],
    base_url: str,
) -> dict[str, Any]:
    selections = freeze_task_selection(eval_examples, train_examples)
    required_models = {SMALL_MODEL, *FRONTIER_MODELS}
    missing_models = sorted(required_models - set(available_models))
    if missing_models:
        raise RuntimeError(f"required models are unavailable: {', '.join(missing_models)}")
    two_shot_examples = [build_two_shot_example(example, demos) for example in eval_examples]
    return {
        "schema_version": 1,
        "status": "frozen",
        "frozen_at": _timestamp(),
        "usage_accounting": {
            "mode": "omitted",
            "reason": "user-approved; BenchMAX 0.2.3 does not expose exact provider usage",
        },
        "neon": {
            "project_id": branch.project_id,
            "parent_branch_id": branch.parent_branch_id,
            "branch_id": branch.branch_id,
            "branch_name": branch.branch_name,
            "endpoint_id": branch.endpoint_id,
            "expires_at": branch.expires_at,
        },
        "datasets": {
            **selections,
            "train_sha256": _sha256_file(example_root / "data" / "train.jsonl"),
            "eval_sha256": _sha256_file(example_root / "data" / "eval.jsonl"),
            "oracle_sha256": _sha256_file(example_root / "data" / "oracle_traces.jsonl"),
        },
        "environment": {
            "class": "order_resolution.order_env.OrderResolutionEnv",
            "schema_head": _schema_head(example_root),
            "tools_sha256": _sha256_json(TOOLS),
            "production_prompts_sha256": _sha256_json(
                [example.payload["prompt_messages"] for example in eval_examples]
            ),
            "two_shot_prompts_sha256": _sha256_json(
                [example.payload["prompt_messages"] for example in two_shot_examples]
            ),
            "model_call_defaults": {
                "max_turns": OrderResolutionEnv.max_turns,
                "max_tool_calls": OrderResolutionEnv.max_tool_calls,
                "temperature": "provider_default",
                "max_completion_tokens": "provider_default",
                "model_retries": 0,
            },
        },
        "execution": {
            "concurrent_groups": CONCURRENT_GROUPS,
            "baseline_group_size": 1,
            "training_group_size": TRAINING_GROUP_SIZE,
            "stress_repeats": STRESS_REPEATS,
        },
        "models": {
            "catalog_resolved_at": _timestamp(),
            "available_ids": list(available_models),
            "arms": _arm_specs(base_url),
        },
    }


async def run_frozen_matrix(
    *,
    env: ObservedOrderResolutionEnv,
    manifest: dict[str, Any],
    example_root: Path,
    raw_path: Path,
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    eval_dataset = await env.create_dataset("eval", example_root / "data")
    train_dataset = await env.create_dataset("train", example_root / "data")
    eval_examples = list(eval_dataset)
    train_examples = list(train_dataset)
    eval_by_id = {example.payload["task_id"]: example for example in eval_examples}
    train_by_id = {example.payload["task_id"]: example for example in train_examples}
    demos = load_oracle_demos(example_root / "data" / "oracle_traces.jsonl")
    auth = CastformModelAuth()
    semaphore = asyncio.Semaphore(int(manifest["execution"]["concurrent_groups"]))
    records: list[dict[str, Any]] = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_phase(
        *,
        phase: str,
        arm: Mapping[str, str],
        task_ids: Sequence[str],
        examples_by_id: Mapping[str, Example[JsonRow]],
        repetition: int,
        group_size: int,
    ) -> None:
        async def run_one(task_id: str) -> list[dict[str, Any]]:
            example = examples_by_id[task_id]
            if arm["prompt"] == "two_oracle_examples":
                example = build_two_shot_example(example, demos)
            group_id = f"{phase}-{arm['id']}-r{repetition}-{task_id}"
            requests = [
                RolloutRequest(
                    rollout_id=f"{group_id}-s{sample:02d}",
                    example=example,
                    model=arm["model"],
                    base_url=arm["endpoint"],
                    model_auth=auth,
                    split="train" if phase == "signal_probe" else "eval",
                )
                for sample in range(group_size)
            ]
            async with semaphore:
                outcomes = await env.run_group(requests)
            return [
                _record_for_rollout(
                    phase=phase,
                    arm=arm,
                    repetition=repetition,
                    group_id=group_id,
                    example=example,
                    rollout_id=request.rollout_id,
                    outcome=outcomes[request.rollout_id],
                    observation=env.pop_observation(request.rollout_id),
                )
                for request in requests
            ]

        phase_groups = await asyncio.gather(*(run_one(task_id) for task_id in task_ids))
        phase_records = sorted(
            (record for group in phase_groups for record in group),
            key=lambda record: record["rollout_id"],
        )
        with raw_path.open("a", encoding="utf-8") as output:
            for record in phase_records:
                line = _canonical_json(record) + "\n"
                _assert_secret_safe(line, forbidden)
                output.write(line)
            output.flush()
        records.extend(phase_records)
        print(
            f"baseline: phase={phase} arm={arm['id']} repetition={repetition} "
            f"rollouts={len(phase_records)}",
            flush=True,
        )

    raw_path.write_text("", encoding="utf-8")
    arms = list(manifest["models"]["arms"])
    eval_ids = manifest["datasets"]["eval_task_ids"]
    stress_ids = manifest["datasets"]["stress_task_ids"]
    probe_ids = manifest["datasets"]["signal_probe_task_ids"]
    for arm in arms:
        await run_phase(
            phase="full",
            arm=arm,
            task_ids=eval_ids,
            examples_by_id=eval_by_id,
            repetition=0,
            group_size=1,
        )
    for arm in arms:
        for repetition in range(1, STRESS_REPEATS + 1):
            await run_phase(
                phase="stress",
                arm=arm,
                task_ids=stress_ids,
                examples_by_id=eval_by_id,
                repetition=repetition,
                group_size=1,
            )
    await run_phase(
        phase="signal_probe",
        arm=next(arm for arm in arms if arm["id"] == "small_base"),
        task_ids=probe_ids,
        examples_by_id=train_by_id,
        repetition=0,
        group_size=TRAINING_GROUP_SIZE,
    )
    return records


def _record_for_rollout(
    *,
    phase: str,
    arm: Mapping[str, str],
    repetition: int,
    group_id: str,
    example: Example[JsonRow],
    rollout_id: str,
    outcome: Any,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    messages = list(observation["messages"])
    initial_message_count = len(example.payload["prompt_messages"])
    reply_count, predicted_disposition, tool_calls, invalid_tool_calls = transcript_facts(
        messages,
        initial_message_count=initial_message_count,
    )
    rewards = {key: float(value) for key, value in outcome.rewards.items()}
    return {
        "phase": phase,
        "arm": arm["id"],
        "model": arm["model"],
        "repetition": repetition,
        "group_id": group_id,
        "rollout_id": rollout_id,
        "task_id": example.payload["task_id"],
        "cell": example.payload["cell"],
        "action_family": example.payload["action_family"],
        "outcome_class": example.payload["outcome_class"],
        "expected_disposition": example.payload["expected_disposition"],
        "predicted_disposition": predicted_disposition,
        "task_success": float(rewards.get("task_success", 0.0)),
        "rewards": rewards,
        "termination_reason": outcome.termination_reason,
        "error_present": outcome.error is not None,
        "latency_seconds": round(float(observation["latency_seconds"]), 6),
        "initial_message_count": initial_message_count,
        "reply_call_count": reply_count,
        "tool_call_count": tool_calls,
        "invalid_tool_call_count": invalid_tool_calls,
        "messages": messages,
    }


def transcript_facts(
    messages: Sequence[Mapping[str, Any]], *, initial_message_count: int = 0
) -> tuple[int, str | None, int, int]:
    """Extract facts only from messages emitted during the live rollout."""

    if not 0 <= initial_message_count <= len(messages):
        raise ValueError("initial_message_count is outside the transcript")
    reply_count = 0
    predicted_disposition = None
    tool_calls = 0
    invalid_results = 0
    for message in messages[initial_message_count:]:
        for tool_call in message.get("tool_calls") or []:
            tool_calls += 1
            function = tool_call.get("function") or {}
            if function.get("name") == "reply_to_customer":
                reply_count += 1
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if isinstance(arguments.get("disposition"), str):
                    predicted_disposition = arguments["disposition"]
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            if content.startswith(("Unknown tool:", "Invalid JSON", "Tool arguments")):
                invalid_results += 1
            continue
        if isinstance(result, dict) and result.get("code") in INVALID_RESULT_CODES:
            invalid_results += 1
    return reply_count, predicted_disposition, tool_calls, invalid_results


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successes = sum(record["task_success"] == 1.0 for record in records)
    infra_failures = sum(
        bool(record["error_present"]) or record["termination_reason"] not in SCORABLE_TERMINATIONS
        for record in records
    )
    clarify = [
        record for record in records if record["expected_disposition"] == "needs_information"
    ]
    predicted_clarify = [
        record for record in records if record["predicted_disposition"] == "needs_information"
    ]
    clarify_tp = sum(
        record["expected_disposition"] == "needs_information" for record in predicted_clarify
    )
    deny = [record for record in records if record["expected_disposition"] == "cannot_complete"]
    tool_calls = sum(int(record["tool_call_count"]) for record in records)
    invalid_calls = sum(int(record["invalid_tool_call_count"]) for record in records)
    per_cell: dict[str, Any] = {}
    for cell in sorted({str(record["cell"]) for record in records}):
        cell_records = [record for record in records if record["cell"] == cell]
        cell_successes = sum(record["task_success"] == 1.0 for record in cell_records)
        per_cell[cell] = {
            "n": len(cell_records),
            "successes": cell_successes,
            "success_rate": cell_successes / len(cell_records),
            "wilson_95": wilson_interval(cell_successes, len(cell_records)),
        }
    latencies = sorted(float(record["latency_seconds"]) for record in records)
    return {
        "n": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "wilson_95": wilson_interval(successes, total),
        "infrastructure_failure_rate": infra_failures / total if total else 0.0,
        "required_state_accuracy": _reward_mean(records, "_required_state_fraction"),
        "forbidden_mutation_rate": _reward_mean(records, "_forbidden_mutation"),
        "policy_denial_accuracy": _success_mean(deny, "_correct_disposition"),
        "clarification_precision": clarify_tp / len(predicted_clarify)
        if predicted_clarify
        else 0.0,
        "clarification_recall": clarify_tp / len(clarify) if clarify else 0.0,
        "structured_reply_accuracy": _reward_mean(records, "_structured_reply_correct"),
        "terminal_reply_violation_rate": (
            sum(record["reply_call_count"] != 1 for record in records) / total if total else 0.0
        ),
        "valid_tool_call_rate": (tool_calls - invalid_calls) / tool_calls if tool_calls else 0.0,
        "invariant_failure_rate": _reward_mean(records, "_invariant_failure"),
        "mean_unnecessary_tool_calls": _reward_mean(records, "_unnecessary_tool_calls"),
        "model_attributable_failures": sum(
            record["task_success"] == 0.0
            and not record["error_present"]
            and record["termination_reason"] in SCORABLE_TERMINATIONS
            and float(record["rewards"].get("_invariant_failure", 0.0)) == 0.0
            for record in records
        ),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "per_cell": per_cell,
    }


def _reward_mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return (
        statistics.fmean(float(record["rewards"].get(key, 0.0)) for record in records)
        if records
        else 0.0
    )


def _success_mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return _reward_mean(records, key)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def build_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = [record for record in records if record["phase"] == "full"]
    arms = sorted({str(record["arm"]) for record in full})
    summaries = {
        arm: summarize_records([record for record in full if record["arm"] == arm]) for arm in arms
    }
    stress = _stress_summary([record for record in records if record["phase"] == "stress"])
    signal = _signal_summary([record for record in records if record["phase"] == "signal_probe"])
    decision = _headroom_decision(summaries, signal)
    return {
        "generated_at": _timestamp(),
        "arms": summaries,
        "stress": stress,
        "signal_probe": signal,
        "decision": decision,
    }


def _stress_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in sorted({str(record["arm"]) for record in records}):
        arm_records = [record for record in records if record["arm"] == arm]
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in arm_records:
            by_task[str(record["task_id"])].append(record)
        consistent = sum(
            len({record["task_success"] for record in task_records}) == 1
            for task_records in by_task.values()
        )
        result[arm] = {
            **summarize_records(arm_records),
            "tasks": len(by_task),
            "three_repeat_consistency": consistent / len(by_task) if by_task else 0.0,
        }
    return result


def _signal_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["group_id"])].append(record)
    mixed_groups = [
        group
        for group in groups.values()
        if {record["task_success"] for record in group} == {0.0, 1.0}
    ]
    mixed_families = sorted({str(group[0]["action_family"]) for group in mixed_groups})
    success_rate = (
        sum(record["task_success"] == 1.0 for record in records) / len(records) if records else 0.0
    )
    return {
        "groups": len(groups),
        "group_size": TRAINING_GROUP_SIZE,
        "mixed_groups": len(mixed_groups),
        "mixed_group_rate": len(mixed_groups) / len(groups) if groups else 0.0,
        "mixed_action_families": mixed_families,
        "sibling_success_rate": success_rate,
        "passes": (
            len(mixed_groups) >= math.ceil(0.25 * len(groups))
            and set(mixed_families) == {"cancel_item", "change_address", "replace_variant"}
            and 0.10 <= success_rate <= 0.90
        ),
    }


def _headroom_decision(
    summaries: Mapping[str, Mapping[str, Any]], signal: Mapping[str, Any]
) -> dict[str, Any]:
    base = summaries["small_base"]
    frontier_rate = max(
        float(summaries["frontier_gpt"]["success_rate"]),
        float(summaries["frontier_grok"]["success_rate"]),
    )
    infrastructure_rate = sum(
        float(summary["infrastructure_failure_rate"]) * int(summary["n"])
        for summary in summaries.values()
    ) / sum(int(summary["n"]) for summary in summaries.values())
    gates = {
        "infrastructure_below_2_percent": infrastructure_rate < 0.02,
        "frontier_at_least_70_percent": frontier_rate >= 0.70,
        "base_between_15_and_80_percent": 0.15 <= float(base["success_rate"]) <= 0.80,
        "frontier_base_gap_at_least_10_points": (
            frontier_rate - float(base["success_rate"]) >= 0.10
        ),
        "at_least_10_model_attributable_base_failures": int(base["model_attributable_failures"])
        >= 10,
        "signal_probe_passes": bool(signal["passes"]),
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        status = "go"
    elif not signal["passes"]:
        status = "no_rl_signal"
    elif float(base["success_rate"]) > 0.80:
        status = "harden"
    else:
        status = "repair"
    return {
        "status": status,
        "gates": gates,
        "failed_gates": failures,
        "base_success_rate": base["success_rate"],
        "strongest_frontier_success_rate": frontier_rate,
        "frontier_base_gap": frontier_rate - float(base["success_rate"]),
        "infrastructure_failure_rate": infrastructure_rate,
    }


def render_html_report(
    *,
    template_path: Path,
    output_path: Path,
    report: Mapping[str, Any],
    demo_task_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    arm_rows = "".join(
        "<tr>"
        f"<td>{html.escape(arm)}</td>"
        f"<td>{summary['successes']} / {summary['n']}</td>"
        f"<td>{summary['success_rate']:.1%}</td>"
        f"<td>{summary['infrastructure_failure_rate']:.1%}</td>"
        f"<td>{summary['forbidden_mutation_rate']:.1%}</td>"
        f"<td>{summary['structured_reply_accuracy']:.1%}</td>"
        "</tr>"
        for arm, summary in report["arms"].items()
    )
    demo_rows = "".join(_demo_row(task_id, records) for task_id in demo_task_ids)
    failed = report["decision"]["failed_gates"]
    decision_detail = "all frozen gates passed" if not failed else ", ".join(failed)
    rendered = (
        template_path.read_text(encoding="utf-8")
        .replace("{{generated_at}}", html.escape(str(report["generated_at"])))
        .replace("{{decision}}", html.escape(str(report["decision"]["status"])))
        .replace("{{decision_detail}}", html.escape(decision_detail))
        .replace("{{arm_rows}}", arm_rows)
        .replace("{{demo_rows}}", demo_rows)
        .replace("{{signal_json}}", html.escape(json.dumps(report["signal_probe"], indent=2)))
    )
    output_path.write_text(rendered, encoding="utf-8")


def refresh_report_artifacts(*, example_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Recompute derived transcript metrics without making model or database calls."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = manifest_path.parent / manifest["artifacts"]["raw_rollouts"]
    html_path = manifest_path.parent / manifest["artifacts"]["html_report"]
    eval_rows = {row["task_id"]: row for row in _load_jsonl(example_root / "data" / "eval.jsonl")}
    train_rows = {row["task_id"]: row for row in _load_jsonl(example_root / "data" / "train.jsonl")}
    demos = load_oracle_demos(example_root / "data" / "oracle_traces.jsonl")
    records = _load_jsonl(raw_path)
    for record in records:
        rows = train_rows if record["phase"] == "signal_probe" else eval_rows
        row = rows[record["task_id"]]
        example = Example(id="report-refresh", payload=row)
        if record["arm"] == "small_two_shot":
            example = build_two_shot_example(example, demos)
        initial_message_count = len(example.payload["prompt_messages"])
        reply_count, disposition, tool_calls, invalid_calls = transcript_facts(
            record["messages"],
            initial_message_count=initial_message_count,
        )
        record.update(
            {
                "initial_message_count": initial_message_count,
                "reply_call_count": reply_count,
                "predicted_disposition": disposition,
                "tool_call_count": tool_calls,
                "invalid_tool_call_count": invalid_calls,
            }
        )
    temporary = raw_path.with_suffix(f"{raw_path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(_canonical_json(record) + "\n")
    temporary.replace(raw_path)
    report = build_report(records)
    manifest["report"] = report
    manifest["derived_metrics_refreshed_at"] = _timestamp()
    _write_json(manifest_path, manifest, ())
    render_html_report(
        template_path=example_root / "templates" / "report.html",
        output_path=html_path,
        report=report,
        demo_task_ids=manifest["datasets"]["report_demo_task_ids"],
        records=records,
    )
    return manifest


def verify_report_artifacts(manifest_path: Path) -> dict[str, Any]:
    """Reconcile the frozen matrix, derived report, and rollout uniqueness."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = manifest_path.parent / manifest["artifacts"]["raw_rollouts"]
    records = _load_jsonl(raw_path)
    if manifest.get("status") != "complete":
        raise RuntimeError("benchmark manifest is not complete")
    if len(records) != manifest.get("rollout_count"):
        raise RuntimeError("raw rollout count does not match the manifest")
    rollout_ids = [record.get("rollout_id") for record in records]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise RuntimeError("a rollout/world identifier was reused")

    arms = [arm["id"] for arm in manifest["models"]["arms"]]
    eval_ids = set(manifest["datasets"]["eval_task_ids"])
    stress_ids = set(manifest["datasets"]["stress_task_ids"])
    probe_ids = set(manifest["datasets"]["signal_probe_task_ids"])
    required_record_fields = {
        "arm",
        "cell",
        "error_present",
        "initial_message_count",
        "latency_seconds",
        "messages",
        "phase",
        "repetition",
        "reply_call_count",
        "rewards",
        "rollout_id",
        "task_id",
        "task_success",
        "termination_reason",
        "tool_call_count",
    }
    for record in records:
        missing = required_record_fields - record.keys()
        if missing:
            raise RuntimeError(
                f"rollout {record.get('rollout_id', '<unknown>')} is missing metrics"
            )

    full = [record for record in records if record["phase"] == "full"]
    expected_full = {(arm, task_id) for arm in arms for task_id in eval_ids}
    actual_full = [(record["arm"], record["task_id"]) for record in full]
    if set(actual_full) != expected_full or len(actual_full) != len(expected_full):
        raise RuntimeError("full matrix does not contain every task exactly once per arm")

    stress = [record for record in records if record["phase"] == "stress"]
    expected_stress = {
        (arm, task_id, repetition)
        for arm in arms
        for task_id in stress_ids
        for repetition in range(1, STRESS_REPEATS + 1)
    }
    actual_stress = [(record["arm"], record["task_id"], record["repetition"]) for record in stress]
    if set(actual_stress) != expected_stress or len(actual_stress) != len(expected_stress):
        raise RuntimeError("stress matrix does not contain three repeats per frozen task and arm")

    probe = [record for record in records if record["phase"] == "signal_probe"]
    probe_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in probe:
        probe_groups[record["task_id"]].append(record)
    if set(probe_groups) != probe_ids or any(
        len(group) != TRAINING_GROUP_SIZE for group in probe_groups.values()
    ):
        raise RuntimeError("signal probe does not use every frozen task at the planned group size")

    unknown_phases = sorted(
        {record["phase"] for record in records} - {"full", "stress", "signal_probe"}
    )
    if unknown_phases:
        raise RuntimeError(f"raw rollouts contain unknown phases: {', '.join(unknown_phases)}")
    recomputed = build_report(records)
    for key in ("arms", "stress", "signal_probe", "decision"):
        if _canonical_json(recomputed[key]) != _canonical_json(manifest["report"][key]):
            raise RuntimeError(f"report section {key!r} does not reconcile with raw rollouts")
    decision = manifest["report"]["decision"]["status"]
    if decision not in {"go", "harden", "repair", "no_rl_signal"}:
        raise RuntimeError("report decision is not a recognized frozen outcome")
    if decision == "go" and not manifest["report"]["signal_probe"]["passes"]:
        raise RuntimeError("go decision violates the mixed-reward signal threshold")
    return {
        "rollouts": len(records),
        "full": len(full),
        "stress": len(stress),
        "signal_probe": len(probe),
        "unique_worlds": len(rollout_ids),
        "decision": decision,
    }


def _demo_row(task_id: str, records: Sequence[Mapping[str, Any]]) -> str:
    selected = [
        record for record in records if record["phase"] == "full" and record["task_id"] == task_id
    ]
    results = ", ".join(
        f"{record['arm']}={'pass' if record['task_success'] == 1.0 else 'fail'}"
        for record in sorted(selected, key=lambda record: str(record["arm"]))
    )
    cell = selected[0]["cell"] if selected else "missing"
    return (
        "<tr>"
        f"<td>{html.escape(task_id)}</td>"
        f"<td>{html.escape(str(cell))}</td>"
        f"<td>{html.escape(results)}</td>"
        "</tr>"
    )


def run_baseline(
    *,
    example_root: Path,
    neon_env: Path,
    neon_manifest_path: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    """Create a disposable child, freeze the matrix, run it, and clean up."""

    if output_manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite existing {output_manifest_path.name}")
    raw_path = output_manifest_path.with_suffix(".raw.jsonl")
    html_path = output_manifest_path.with_suffix(".html")
    api_key = resolve_neon_api_key(neon_env)
    project = read_project_manifest(neon_manifest_path)
    base_url = config.llm_url()
    branch = None
    manifest: dict[str, Any] = {}
    stage = "model catalog resolution"
    forbidden = [api_key]
    with NeonApi(api_key) as api:
        try:
            available_models = resolve_model_catalog(base_url)
            stage = "child branch creation"
            branch = api.create_runtime_branch(project, purpose="baseline")
            forbidden.extend([branch.admin_database_url, branch.runtime_database_url])
            stage = "dataset and contract freeze"
            freeze_env = ObservedOrderResolutionEnv(branch.runtime_database_url)
            try:
                eval_examples = list(
                    asyncio.run(freeze_env.create_dataset("eval", example_root / "data"))
                )
                train_examples = list(
                    asyncio.run(freeze_env.create_dataset("train", example_root / "data"))
                )
            finally:
                asyncio.run(freeze_env.aclose())
            demos = load_oracle_demos(example_root / "data" / "oracle_traces.jsonl")
            manifest = build_frozen_manifest(
                example_root=example_root,
                branch=branch,
                eval_examples=eval_examples,
                train_examples=train_examples,
                demos=demos,
                available_models=available_models,
                base_url=base_url,
            )
            manifest["artifacts"] = {
                "raw_rollouts": raw_path.name,
                "html_report": html_path.name,
            }
            _write_json(output_manifest_path, manifest, forbidden)
            stage = "frozen evaluation matrix"
            env = ObservedOrderResolutionEnv(branch.runtime_database_url)
            try:
                records = asyncio.run(
                    run_frozen_matrix(
                        env=env,
                        manifest=manifest,
                        example_root=example_root,
                        raw_path=raw_path,
                        forbidden=forbidden,
                    )
                )
            finally:
                asyncio.run(env.aclose())
            stage = "report generation"
            report = build_report(records)
            manifest["status"] = "complete"
            manifest["completed_at"] = _timestamp()
            manifest["rollout_count"] = len(records)
            manifest["report"] = report
            render_html_report(
                template_path=example_root / "templates" / "report.html",
                output_path=html_path,
                report=report,
                demo_task_ids=manifest["datasets"]["report_demo_task_ids"],
                records=records,
            )
        except BaseException as error:
            if manifest:
                manifest["status"] = (
                    "aborted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                )
                manifest["failed_at"] = _timestamp()
                manifest["failed_stage"] = stage
            if isinstance(error, Exception):
                raise RuntimeError(f"baseline failed during {stage}") from error
            raise
        finally:
            if branch is not None:
                api.delete_branch(project.project_id, branch.branch_id)
                if manifest:
                    manifest["neon"]["deleted"] = True
                    manifest["neon"]["deleted_at"] = _timestamp()
            if manifest:
                _write_json(output_manifest_path, manifest, forbidden)
    if manifest.get("status") != "complete":
        raise RuntimeError("baseline did not complete")
    return manifest


# ---------------------------------------------------------------------------
# order-resolution-v2: versioned, append-only execution
# ---------------------------------------------------------------------------

V2_SOURCE_MODULES = (
    "benchmark.py",
    "benchmark_spec.py",
    "branching.py",
    "command_codes.py",
    "contract.py",
    "database.py",
    "demo.py",
    "domain.py",
    "fixtures.py",
    "grading.py",
    "hosting.py",
    "order_env.py",
    "policy.py",
    "schema.py",
)

#: Source files whose bytes define the environment's observable behavior. The
#: ABI deliberately excludes constructor values and every DSN.
ABI_SOURCE_MODULES = (
    "command_codes.py",
    "domain.py",
    "fixtures.py",
    "grading.py",
    "hosting.py",
    "order_env.py",
    "policy.py",
    "schema.py",
)

DATASET_ROW_CONTRACT = (
    "action_family",
    "benchmark_id",
    "cell",
    "expected_disposition",
    "expected_reply",
    "fixture",
    "forbidden_state",
    "outcome_class",
    "prompt_messages",
    "required_state",
    "split",
    "task_id",
)


class BenchmarkStateError(RuntimeError):
    """A versioned artifact, authorization, or wave transition is not permitted."""


def _module_digests(example_root: Path, names: Sequence[str]) -> dict[str, str]:
    return {
        name: _sha256_file(example_root / "order_resolution" / name) for name in sorted(names)
    }


def environment_abi(example_root: Path) -> dict[str, Any]:
    """The secret-independent environment contract frozen in ``spec.json``.

    BenchMAX 0.2.3 serializes the branch-specific runtime DSN into the pickle,
    so an exact bundle digest cannot be stable across child branches. This ABI
    is what must not move; each wave records its own bundle digest separately.
    """

    return {
        "environment_class": "order_resolution.order_env.OrderResolutionEnv",
        "sources_sha256": _module_digests(example_root, ABI_SOURCE_MODULES),
        "tools": json.loads(_canonical_json(TOOLS)),
        "reply_schema": reply_tool_schema(),
        "system_contract_sha256": hashlib.sha256(render_system_contract().encode()).hexdigest(),
        "limits": {
            "max_turns": OrderResolutionEnv.max_turns,
            "max_tool_calls": OrderResolutionEnv.max_tool_calls,
        },
        "schema_head": _schema_head(example_root),
        "bundle_builder_sha256": _sha256_file(example_root / "order_resolution" / "hosting.py"),
        "pip_dependencies": list(CANONICAL_RUNTIME_DEPENDENCIES),
        "dataset_row_contract": list(DATASET_ROW_CONTRACT),
        "workspace_lock_sha256": _sha256_file(example_root.parents[1] / "uv.lock"),
    }


def environment_abi_sha256(example_root: Path) -> str:
    return _sha256_json(environment_abi(example_root))


def assert_bundle_matches_abi(
    inspection: Mapping[str, Any], *, example_root: Path, expected_abi_sha256: str
) -> None:
    """Prove an instantiated, secret-bearing bundle still matches the frozen ABI."""

    actual = environment_abi_sha256(example_root)
    if actual != expected_abi_sha256:
        raise BenchmarkStateError(
            f"environment ABI drifted: {actual} does not match frozen {expected_abi_sha256}"
        )
    if inspection.get("class") != "OrderResolutionEnv":
        raise BenchmarkStateError("bundled environment class does not match the frozen ABI")
    if list(inspection.get("pip_dependencies", ())) != list(CANONICAL_RUNTIME_DEPENDENCIES):
        raise BenchmarkStateError("bundled dependencies do not match the frozen ABI")
    if inspection.get("secret_boundary") != "ok":
        raise BenchmarkStateError("bundle did not pass its secret-boundary inspection")


def _create_exclusive(path: Path, payload: str, forbidden: Sequence[str]) -> None:
    """Write once. An existing path is an explicit failure, never an overwrite."""

    _assert_secret_safe(payload, forbidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise BenchmarkStateError(f"refusing to overwrite existing {path}") from error


def _allowed_artifact(example_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(example_root.resolve()).as_posix()
    except ValueError as error:
        raise BenchmarkStateError(f"{path} is outside the example root") from error
    if relative not in spec.ALLOWED_ARTIFACT_PATHS:
        raise BenchmarkStateError(f"{relative} is not a permitted v2 artifact path")
    return relative


def build_v2_spec(*, example_root: Path, predecessor_manifest: Path) -> dict[str, Any]:
    """Freeze experiment identity, sources, data, ABI, selections, and gates."""

    verify_predecessor(example_root, benchmark_id=spec.BENCHMARK_ID)
    if predecessor_manifest.resolve() != (example_root / spec.PREDECESSOR_MANIFEST_PATH).resolve():
        raise BenchmarkStateError("predecessor manifest must be the sealed v1 baseline")
    data_dir = example_root / spec.DATA_ROOT
    generated = generate_v2_data()
    check_v2_data(data_dir)
    train_rows = _load_jsonl(data_dir / "train.jsonl")
    eval_rows = _load_jsonl(data_dir / "eval.jsonl")
    demos = load_v2_oracle_demos(data_dir / "oracle_traces.jsonl")
    identity = spec.spec_identity()
    return {
        **identity,
        "frozen_at": _timestamp(),
        "repair_rationale": (
            "v1 hid the reply vocabulary, shipped two identical order lines, asked for more "
            "facts than its clarifications named, and taught an ungrounded first-item "
            "heuristic; v2 publishes the policy, makes every target visibly resolvable, and "
            "compiles its demonstrations from the live service"
        ),
        "sources_sha256": _module_digests(example_root, V2_SOURCE_MODULES),
        "datasets": {
            "train_sha256": _sha256_file(data_dir / "train.jsonl"),
            "eval_sha256": _sha256_file(data_dir / "eval.jsonl"),
            "oracle_sha256": _sha256_file(data_dir / "oracle_traces.jsonl"),
            "eval_frozen_sha256": (data_dir / "eval.sha256").read_text(encoding="utf-8").strip(),
            "catalog_generation_key_sha256": generated.catalog.generation_key_sha256,
            "catalog_content_sha256": generated.catalog.content_sha256,
        },
        "environment": {
            "abi_sha256": environment_abi_sha256(example_root),
            "abi": environment_abi(example_root),
        },
        "prompts": {
            "production_sha256": _sha256_json(
                [row["prompt_messages"] for row in eval_rows]
            ),
            "two_shot_sha256": _sha256_json(
                [
                    build_two_shot_example(Example(id=row["task_id"], payload=row), demos).payload[
                        "prompt_messages"
                    ]
                    for row in eval_rows
                ]
            ),
            "report_template_sha256": _sha256_file(example_root / "templates" / "report.html"),
        },
        "selections": v2_task_selection(train_rows, eval_rows),
        "artifact_creation_mode": "exclusive",
        "retry_state_machine": {
            "canary_attempts": list(spec.CANARY_ATTEMPT_ROOTS),
            "attempt_2_requires": "a sealed attempt-1 infrastructure_failure",
            "authorization": spec.CANARY_AUTHORIZATION_PATH,
            "post_result_change": "requires a separately audited v3 spec",
        },
    }


def v2_task_selection(
    train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Resolve every frozen wave selection from the generated rows."""

    train_by_cell = _by_cell_rows(train_rows)
    eval_by_cell = _by_cell_rows(eval_rows)
    if set(train_by_cell) != set(spec.CELLS) or set(eval_by_cell) != set(spec.CELLS):
        raise BenchmarkStateError("the v2 task grid must contain exactly nine cells")
    if any(len(rows) != spec.TRAIN_ROWS_PER_CELL for rows in train_by_cell.values()):
        raise BenchmarkStateError("each v2 training cell must be full")
    if any(len(rows) != spec.EVAL_ROWS_PER_CELL for rows in eval_by_cell.values()):
        raise BenchmarkStateError("each v2 evaluation cell must be full")

    def pick(by_cell: Mapping[str, list[Mapping[str, Any]]], indices: Sequence[int]) -> list[str]:
        return [
            by_cell[cell][index]["task_id"] for cell in spec.CELLS for index in sorted(indices)
        ]

    selection = {
        "canary_task_ids": pick(train_by_cell, spec.CANARY_INDICES),
        "eval_task_ids": [row["task_id"] for row in eval_rows],
        "stress_task_ids": pick(eval_by_cell, spec.STRESS_INDICES),
        "signal_probe_task_ids": pick(train_by_cell, spec.SIGNAL_PROBE_INDICES),
        "report_demo_task_ids": [
            eval_by_cell[cell][spec.REPORT_DEMO_INDEX]["task_id"]
            for cell in spec.REPORT_DEMO_CELLS
        ],
        "oracle_demo_task_ids": list(spec.oracle_demo_task_ids()),
    }
    reserved = set(selection["canary_task_ids"])
    for key in ("signal_probe_task_ids", "oracle_demo_task_ids"):
        if reserved & set(selection[key]):
            raise BenchmarkStateError(f"canary tasks overlap {key}")
    if len(selection["canary_task_ids"]) != spec.CANARY_TASKS_PER_ARM:
        raise BenchmarkStateError("canary selection is not 18 tasks")
    return selection


def _by_cell_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["cell"])].append(row)
    return {
        cell: sorted(items, key=lambda row: str(row["task_id"]))
        for cell, items in result.items()
    }


def load_v2_oracle_demos(path: Path) -> tuple[dict[str, Any], ...]:
    """The two live demonstrations frozen before the first model call."""

    by_id = {row["task_id"]: row for row in _load_jsonl(path)}
    demo_ids = spec.oracle_demo_task_ids()
    missing = [task_id for task_id in demo_ids if task_id not in by_id]
    if missing:
        raise BenchmarkStateError(f"missing frozen v2 oracle demos: {', '.join(missing)}")
    demos = tuple(by_id[task_id] for task_id in demo_ids)
    if any(demo.get("benchmark_id") != spec.BENCHMARK_ID for demo in demos):
        raise BenchmarkStateError("oracle demos are not all order-resolution-v2")
    if any(demo.get("reward") != 1.0 for demo in demos):
        raise BenchmarkStateError("a frozen oracle demo does not score 1.0")
    return demos


def freeze_v2_spec(
    *, example_root: Path, predecessor_manifest: Path, spec_path: Path
) -> dict[str, Any]:
    """Write ``spec.json`` exactly once, before any v2 model call."""

    _allowed_artifact(example_root, spec_path)
    payload = build_v2_spec(
        example_root=example_root, predecessor_manifest=predecessor_manifest
    )
    _create_exclusive(spec_path, json.dumps(payload, indent=2, sort_keys=True) + "\n", ())
    return payload


def read_spec(spec_path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != spec.SPEC_SCHEMA_VERSION:
        raise BenchmarkStateError("spec.json is not a schema-v2 specification")
    if payload.get("benchmark_id") != spec.BENCHMARK_ID:
        raise BenchmarkStateError("spec.json records another benchmark")
    return payload, _sha256_file(spec_path)


def v2_rollout_id(
    *,
    spec_sha256: str,
    wave: str,
    attempt: int | None,
    run_nonce: str,
    arm: str,
    repetition: int,
    task_id: str,
    sample: int,
) -> str:
    """A globally unique id; no v1 or sibling wave can collide with it."""

    payload = "|".join(
        (
            spec.BENCHMARK_ID,
            spec_sha256,
            wave,
            str(attempt if attempt is not None else "-"),
            run_nonce,
            arm,
            f"r{repetition}",
            task_id,
            f"s{sample:02d}",
        )
    )
    return f"{spec.BENCHMARK_ID}-{wave}-{hashlib.sha256(payload.encode()).hexdigest()[:40]}"


def _wave_paths(example_root: Path, manifest_path: Path, wave: str) -> dict[str, Path]:
    _allowed_artifact(example_root, manifest_path)
    if manifest_path.name != "manifest.json":
        raise BenchmarkStateError("a wave manifest must be named manifest.json")
    root = manifest_path.parent
    paths = {
        "manifest": manifest_path,
        "raw": root / "rollouts.raw.jsonl",
        "seal": root / "seal.json",
    }
    if wave == "full":
        paths["html"] = root / FULL_REPORT_FILENAME
    for path in paths.values():
        _allowed_artifact(example_root, path)
        if path.exists():
            raise BenchmarkStateError(f"refusing to start: {path} already exists")
    return paths


def _expected_wave_root(wave: str, attempt: int | None) -> str:
    if wave == "canary":
        if attempt not in range(1, spec.MAX_CANARY_ATTEMPTS + 1):
            raise BenchmarkStateError(f"canary attempt must be 1..{spec.MAX_CANARY_ATTEMPTS}")
        return spec.CANARY_ATTEMPT_ROOTS[attempt - 1]
    if wave != "full":
        raise BenchmarkStateError(f"unknown wave {wave!r}")
    if attempt is not None:
        raise BenchmarkStateError("the full wave has no attempts")
    return spec.FULL_ROOT


def _assert_wave_preconditions(
    *,
    example_root: Path,
    spec_sha256: str,
    wave: str,
    attempt: int | None,
    manifest_path: Path,
    authorization_path: Path,
    requires_infrastructure_failure: Path | None,
    requires_canary: Path | None,
) -> dict[str, Any] | None:
    """Enforce the frozen retry/authorization state machine before any model call."""

    expected_root = _expected_wave_root(wave, attempt)
    actual_root = manifest_path.parent.resolve().relative_to(example_root.resolve()).as_posix()
    if actual_root != expected_root:
        raise BenchmarkStateError(
            f"{wave} attempt {attempt} must write to {expected_root}, not {actual_root}"
        )
    if wave == "canary":
        if authorization_path.exists():
            raise BenchmarkStateError(
                "canary execution is closed: an authorization has already been created"
            )
        for index, root in enumerate(spec.CANARY_ATTEMPT_ROOTS, start=1):
            if index >= (attempt or 1):
                continue
            prior = example_root / root / "manifest.json"
            if not prior.exists():
                raise BenchmarkStateError(f"attempt {index} has not been run")
        if attempt == 1:
            if requires_infrastructure_failure is not None:
                raise BenchmarkStateError("attempt 1 never requires a prior failure")
            return None
        if requires_infrastructure_failure is None:
            raise BenchmarkStateError(
                "attempt 2 requires --requires-infrastructure-failure with attempt 1's manifest"
            )
        prior = json.loads(requires_infrastructure_failure.read_text(encoding="utf-8"))
        if prior.get("spec_sha256") != spec_sha256:
            raise BenchmarkStateError("attempt 1 was run against a different spec")
        if prior.get("status") != "infrastructure_failure":
            raise BenchmarkStateError(
                f"attempt 1 sealed {prior.get('status')!r}; only infrastructure_failure retries"
            )
        _verify_seal(requires_infrastructure_failure)
        return prior

    if requires_canary is None:
        raise BenchmarkStateError("the full wave requires --requires-canary authorization")
    authorization = json.loads(requires_canary.read_text(encoding="utf-8"))
    if authorization.get("benchmark_id") != spec.BENCHMARK_ID:
        raise BenchmarkStateError("authorization records another benchmark")
    if authorization.get("spec_sha256") != spec_sha256:
        raise BenchmarkStateError("authorization was issued against a different spec")
    if authorization.get("attempt_status") != "proceed":
        raise BenchmarkStateError("authorization does not record a passing canary")
    accepted = example_root / authorization["attempt_manifest_path"]
    if _sha256_file(accepted) != authorization["attempt_manifest_sha256"]:
        raise BenchmarkStateError("the accepted canary manifest no longer matches its seal")
    _verify_seal(accepted)
    return authorization


def _seal_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "sealed_at": _timestamp(),
        "sha256": {
            name: _sha256_file(path)
            for name, path in sorted(paths.items())
            if name != "seal" and path.exists()
        },
    }


def _verify_seal(manifest_path: Path) -> dict[str, Any]:
    seal_path = manifest_path.parent / "seal.json"
    if not seal_path.exists():
        raise BenchmarkStateError(f"{manifest_path.parent.name} has no seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name, digest in seal["sha256"].items():
        path = manifest_path.parent / _SEAL_FILENAMES[name]
        if not path.exists():
            raise BenchmarkStateError(f"sealed artifact {name} is missing")
        if _sha256_file(path) != digest:
            raise BenchmarkStateError(f"sealed artifact {name} was modified after sealing")
    return seal


_SEAL_FILENAMES = {
    "manifest": "manifest.json",
    "raw": "rollouts.raw.jsonl",
    "html": FULL_REPORT_FILENAME,
}


def evaluate_canary(
    records: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]
) -> dict[str, Any]:
    """Score the predeclared canary conditions and pick one terminal status.

    Ordering matters: infrastructure precedes product, and a product verdict is
    binding for v2 — only ``infrastructure_failure`` permits the second attempt.
    """

    by_arm = {
        arm.id: [record for record in records if record["arm"] == arm.id] for arm in spec.ARMS
    }
    successes = {
        arm: sum(row["task_success"] == 1.0 for row in rows) for arm, rows in by_arm.items()
    }
    infrastructure = sum(
        bool(record["error_present"]) or record["termination_reason"] not in SCORABLE_TERMINATIONS
        for record in records
    )
    invariants = sum(
        float(record["rewards"].get("_invariant_failure", 0.0)) > 0.0 for record in records
    )
    complete = len(records) == gates["expected_records"] and len(
        {record["rollout_id"] for record in records}
    ) == len(records)

    frontier_reports = {}
    for arm_id in spec.FRONTIER_ARM_IDS:
        rows = by_arm[arm_id]
        per_cell = defaultdict(int)
        for record in rows:
            per_cell[str(record["cell"])] += int(record["task_success"] == 1.0)
        frontier_reports[arm_id] = {
            "successes": successes[arm_id],
            "cells_covered": sum(
                count >= gates["frontier_min_successes_per_cell"] for count in per_cell.values()
            ),
            "forbidden_mutations": sum(
                float(record["rewards"].get("_forbidden_mutation", 0.0)) > 0.0 for record in rows
            ),
            "invariant_failures": sum(
                float(record["rewards"].get("_invariant_failure", 0.0)) > 0.0 for record in rows
            ),
        }
    qualifying = [
        arm_id
        for arm_id, report in frontier_reports.items()
        if report["successes"] >= gates["frontier_min_successes"]
        and report["cells_covered"] == len(spec.CELLS)
        and report["forbidden_mutations"] == 0
        and report["invariant_failures"] == 0
    ]
    base = successes[spec.BASE_ARM_ID]
    strongest = max(report["successes"] for report in frontier_reports.values())
    two_shot_classes = {
        str(record["outcome_class"])
        for record in by_arm[spec.TWO_SHOT_ARM_ID]
        if record["task_success"] == 1.0
    }
    two_shot_ok = two_shot_classes == set(spec.OUTCOME_CLASSES)

    gate_results = {
        "exact_record_count_and_unique_ids": complete,
        "zero_infrastructure_failures": infrastructure == 0,
        "zero_invariant_failures": invariants == 0,
        "one_frontier_meets_all_conditions": bool(qualifying),
        "base_within_band": gates["base_min_successes"] <= base <= gates["base_max_successes"],
        "frontier_base_gap": strongest - base >= gates["min_frontier_base_gap"],
        "two_shot_covers_every_outcome_class": two_shot_ok,
    }
    if not gate_results["zero_infrastructure_failures"]:
        status = "infrastructure_failure"
    elif not complete or invariants:
        status = "repair_again"
    elif base > gates["base_max_successes"]:
        status = "harden"
    elif not qualifying or not gate_results["frontier_base_gap"]:
        status = "no_headroom"
    elif base < gates["base_min_successes"]:
        status = "repair_again"
    elif not two_shot_ok:
        status = "repair_again"
    else:
        status = "proceed"
    return {
        "status": status,
        "gates": gate_results,
        "failed_gates": [name for name, passed in gate_results.items() if not passed],
        "successes": successes,
        "frontiers": frontier_reports,
        "qualifying_frontiers": qualifying,
        "infrastructure_failures": infrastructure,
        "invariant_failures": invariants,
        "two_shot_outcome_classes": sorted(two_shot_classes),
    }


def v2_decision(
    summaries: Mapping[str, Mapping[str, Any]], signal: Mapping[str, Any]
) -> dict[str, Any]:
    """The unchanged v1 thresholds, reported in the v2 decision vocabulary."""

    gates_spec = spec.FULL_GATES
    base = summaries[spec.BASE_ARM_ID]
    frontier_rate = max(float(summaries[arm]["success_rate"]) for arm in spec.FRONTIER_ARM_IDS)
    infrastructure_rate = sum(
        float(summary["infrastructure_failure_rate"]) * int(summary["n"])
        for summary in summaries.values()
    ) / sum(int(summary["n"]) for summary in summaries.values())
    base_rate = float(base["success_rate"])
    attributable = int(base["model_attributable_failures"])
    gates = {
        "infrastructure_below_2_percent": infrastructure_rate
        < gates_spec["max_infrastructure_failure_rate"],
        "frontier_at_least_70_percent": frontier_rate >= gates_spec["min_frontier_success_rate"],
        "base_between_15_and_80_percent": (
            gates_spec["min_base_success_rate"] <= base_rate <= gates_spec["max_base_success_rate"]
        ),
        "frontier_base_gap_at_least_10_points": (
            frontier_rate - base_rate >= gates_spec["min_frontier_base_gap"]
        ),
        "at_least_10_model_attributable_base_failures": (
            attributable >= gates_spec["min_model_attributable_base_failures"]
        ),
        "signal_probe_passes": bool(signal["passes"]),
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        status = "go"
    elif not gates["infrastructure_below_2_percent"] or not gates[
        "at_least_10_model_attributable_base_failures"
    ]:
        status = "repair_again"
    elif base_rate > gates_spec["max_base_success_rate"]:
        status = "harden"
    elif base_rate < gates_spec["min_base_success_rate"]:
        status = "repair_again"
    else:
        status = "no_headroom"
    return {
        "status": status,
        "gates": gates,
        "failed_gates": failures,
        "base_success_rate": base_rate,
        "strongest_frontier_success_rate": frontier_rate,
        "frontier_base_gap": frontier_rate - base_rate,
        "infrastructure_failure_rate": infrastructure_rate,
    }


def build_v2_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = [record for record in records if record["phase"] == "full"]
    summaries = {
        arm: summarize_records([record for record in full if record["arm"] == arm])
        for arm in sorted({str(record["arm"]) for record in full})
    }
    stress = _stress_summary([record for record in records if record["phase"] == "stress"])
    signal = _signal_summary([record for record in records if record["phase"] == "signal_probe"])
    return {
        "generated_at": _timestamp(),
        "arms": summaries,
        "stress": stress,
        "signal_probe": signal,
        "decision": v2_decision(summaries, signal),
    }


async def run_v2_matrix(
    *,
    env: ObservedOrderResolutionEnv,
    manifest: Mapping[str, Any],
    data_dir: Path,
    raw_path: Path,
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    """Execute one wave, appending each phase's records as it completes."""

    eval_examples = {
        example.payload["task_id"]: example
        for example in await env.create_dataset("eval", data_dir)
    }
    train_examples = {
        example.payload["task_id"]: example
        for example in await env.create_dataset("train", data_dir)
    }
    demos = load_v2_oracle_demos(data_dir / "oracle_traces.jsonl")
    auth = CastformModelAuth()
    execution = manifest["execution"]
    semaphore = asyncio.Semaphore(int(execution["concurrency"]))
    selections = manifest["selections"]
    records: list[dict[str, Any]] = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.touch()

    async def run_phase(
        *,
        phase: str,
        arm: Mapping[str, str],
        task_ids: Sequence[str],
        examples_by_id: Mapping[str, Example[JsonRow]],
        repetition: int,
        group_size: int,
        split: str,
    ) -> None:
        async def run_one(task_id: str) -> list[dict[str, Any]]:
            example = examples_by_id[task_id]
            if arm["prompt"] == spec.TWO_SHOT_PROMPT:
                example = build_two_shot_example(example, demos)
            requests = [
                RolloutRequest(
                    rollout_id=v2_rollout_id(
                        spec_sha256=manifest["spec_sha256"],
                        wave=manifest["wave"],
                        attempt=manifest["attempt"],
                        run_nonce=manifest["run_nonce"],
                        arm=arm["id"],
                        repetition=repetition,
                        task_id=task_id,
                        sample=sample,
                    ),
                    example=example,
                    model=arm["model"],
                    base_url=arm["endpoint"],
                    model_auth=auth,
                    split=split,
                )
                for sample in range(group_size)
            ]
            async with semaphore:
                outcomes = await env.run_group(requests)
            return [
                {
                    **_record_for_rollout(
                        phase=phase,
                        arm=arm,
                        repetition=repetition,
                        group_id=f"{phase}-{arm['id']}-r{repetition}-{task_id}",
                        example=example,
                        rollout_id=request.rollout_id,
                        outcome=outcomes[request.rollout_id],
                        observation=env.pop_observation(request.rollout_id),
                    ),
                    "benchmark_id": spec.BENCHMARK_ID,
                    "wave": manifest["wave"],
                    "attempt": manifest["attempt"],
                    "sample": sample,
                    "world_id": world_id_for_rollout(request.rollout_id),
                }
                for sample, request in enumerate(requests)
            ]

        groups = await asyncio.gather(*(run_one(task_id) for task_id in task_ids))
        phase_records = sorted(
            (record for group in groups for record in group),
            key=lambda record: record["rollout_id"],
        )
        with raw_path.open("a", encoding="utf-8") as output:
            for record in phase_records:
                line = _canonical_json(record) + "\n"
                _assert_secret_safe(line, forbidden)
                output.write(line)
            output.flush()
        records.extend(phase_records)
        print(
            f"{manifest['wave']}: phase={phase} arm={arm['id']} repetition={repetition} "
            f"rollouts={len(phase_records)}",
            flush=True,
        )

    arms = list(manifest["models"]["arms"])
    if manifest["wave"] == "canary":
        for arm in arms:
            await run_phase(
                phase="canary",
                arm=arm,
                task_ids=selections["canary_task_ids"],
                examples_by_id=train_examples,
                repetition=0,
                group_size=int(execution["group_size"]),
                split="train",
            )
        return records

    for arm in arms:
        await run_phase(
            phase="full",
            arm=arm,
            task_ids=selections["eval_task_ids"],
            examples_by_id=eval_examples,
            repetition=0,
            group_size=int(execution["group_size"]),
            split="eval",
        )
    for arm in arms:
        for repetition in range(1, int(execution["stress_repeats"]) + 1):
            await run_phase(
                phase="stress",
                arm=arm,
                task_ids=selections["stress_task_ids"],
                examples_by_id=eval_examples,
                repetition=repetition,
                group_size=int(execution["group_size"]),
                split="eval",
            )
    await run_phase(
        phase="signal_probe",
        arm=next(arm for arm in arms if arm["id"] == spec.BASE_ARM_ID),
        task_ids=selections["signal_probe_task_ids"],
        examples_by_id=train_examples,
        repetition=0,
        group_size=int(execution["training_group_size"]),
        split="train",
    )
    return records


def _v2_arm_specs(base_url: str) -> list[dict[str, str]]:
    return [
        {"id": arm.id, "model": arm.model, "endpoint": base_url, "prompt": arm.prompt}
        for arm in spec.ARMS
    ]


def _v2_execution(wave: str) -> dict[str, int]:
    if wave == "canary":
        return {
            "concurrency": spec.CANARY_CONCURRENCY,
            "group_size": spec.CANARY_GROUP_SIZE,
            "stress_repeats": 0,
            "training_group_size": 0,
        }
    return {
        "concurrency": spec.FULL_CONCURRENCY,
        "group_size": spec.FULL_GROUP_SIZE,
        "stress_repeats": spec.STRESS_REPEATS,
        "training_group_size": spec.TRAINING_GROUP_SIZE,
    }


def run_v2_wave(
    *,
    example_root: Path,
    spec_path: Path,
    manifest_path: Path,
    wave: str,
    attempt: int | None = None,
    authorization_path: Path,
    requires_infrastructure_failure: Path | None = None,
    requires_canary: Path | None = None,
    neon_env: Path,
    neon_manifest_path: Path,
    run_nonce: str,
) -> dict[str, Any]:
    """Create a fresh child, run one wave append-only, seal it, and tear down."""

    verify_predecessor(example_root, benchmark_id=spec.BENCHMARK_ID)
    frozen, spec_sha256 = read_spec(spec_path)
    paths = _wave_paths(example_root, manifest_path, wave)
    prior = _assert_wave_preconditions(
        example_root=example_root,
        spec_sha256=spec_sha256,
        wave=wave,
        attempt=attempt,
        manifest_path=manifest_path,
        authorization_path=authorization_path,
        requires_infrastructure_failure=requires_infrastructure_failure,
        requires_canary=requires_canary,
    )
    data_dir = example_root / spec.DATA_ROOT
    for name, key in (
        ("train.jsonl", "train_sha256"),
        ("eval.jsonl", "eval_sha256"),
        ("oracle_traces.jsonl", "oracle_sha256"),
    ):
        if _sha256_file(data_dir / name) != frozen["datasets"][key]:
            raise BenchmarkStateError(f"data/v2/{name} no longer matches the frozen spec")
    for name, digest in frozen["sources_sha256"].items():
        if _sha256_file(example_root / "order_resolution" / name) != digest:
            raise BenchmarkStateError(f"{name} changed after the spec was frozen")

    api_key = resolve_neon_api_key(neon_env)
    project = read_project_manifest(neon_manifest_path)
    base_url = config.llm_url()
    forbidden = [api_key]
    branch = None
    manifest: dict[str, Any] = {}
    stage = "model catalog resolution"
    with NeonApi(api_key) as api:
        try:
            available_models = resolve_model_catalog(base_url)
            missing = sorted(set(spec.REQUIRED_MODELS) - set(available_models))
            if missing:
                raise BenchmarkStateError(f"required models are unavailable: {', '.join(missing)}")
            stage = "child branch creation"
            branch = api.create_runtime_branch(project, purpose=f"v2-{wave}")
            forbidden.extend([branch.admin_database_url, branch.runtime_database_url])
            stage = "bundle and environment ABI verification"
            bundle = build_environment_bundle(branch.runtime_database_url)
            inspection = inspect_environment_bundle(
                bundle,
                runtime_database_url=branch.runtime_database_url,
                forbidden_secrets={
                    "admin_url": branch.admin_database_url,
                    "api_key": api_key,
                },
            )
            assert_bundle_matches_abi(
                inspection,
                example_root=example_root,
                expected_abi_sha256=frozen["environment"]["abi_sha256"],
            )
            manifest = {
                "schema_version": spec.SPEC_SCHEMA_VERSION,
                "benchmark_id": spec.BENCHMARK_ID,
                "spec_sha256": spec_sha256,
                "wave": wave,
                "attempt": attempt,
                "run_nonce": run_nonce,
                "status": "running",
                "started_at": _timestamp(),
                "usage_accounting": dict(spec.USAGE_ACCOUNTING),
                "neon": {
                    "project_id": branch.project_id,
                    "parent_branch_id": branch.parent_branch_id,
                    "branch_id": branch.branch_id,
                    "branch_name": branch.branch_name,
                    "endpoint_id": branch.endpoint_id,
                    "expires_at": branch.expires_at,
                },
                # Branch-specific by design: the pickle carries this child's DSN.
                "bundle": inspection,
                "environment_abi_sha256": frozen["environment"]["abi_sha256"],
                "datasets": dict(frozen["datasets"]),
                "selections": dict(frozen["selections"]),
                "execution": _v2_execution(wave),
                "models": {
                    "catalog_resolved_at": _timestamp(),
                    "available_ids": list(available_models),
                    "arms": _v2_arm_specs(base_url),
                },
                "gates": dict(spec.CANARY_GATES if wave == "canary" else spec.FULL_GATES),
                "artifacts": {
                    "raw_rollouts": paths["raw"].name,
                    "seal": paths["seal"].name,
                    **({"html_report": paths["html"].name} if wave == "full" else {}),
                },
            }
            if prior is not None:
                manifest["prior_attempt"] = {
                    "status": prior.get("status") or prior.get("attempt_status"),
                    "spec_sha256": prior.get("spec_sha256"),
                }
            stage = "wave execution"
            env = ObservedOrderResolutionEnv(branch.runtime_database_url)
            try:
                records = asyncio.run(
                    run_v2_matrix(
                        env=env,
                        manifest=manifest,
                        data_dir=data_dir,
                        raw_path=paths["raw"],
                        forbidden=forbidden,
                    )
                )
            finally:
                asyncio.run(env.aclose())
            stage = "gate evaluation"
            manifest["rollout_count"] = len(records)
            if wave == "canary":
                manifest["canary"] = evaluate_canary(records, manifest["gates"])
                manifest["status"] = manifest["canary"]["status"]
            else:
                report = build_v2_report(records)
                manifest["report"] = report
                manifest["status"] = "complete"
                render_html_report(
                    template_path=example_root / "templates" / "report.html",
                    output_path=paths["html"],
                    report=report,
                    demo_task_ids=manifest["selections"]["report_demo_task_ids"],
                    records=records,
                )
            manifest["completed_at"] = _timestamp()
        except BaseException as error:
            if manifest:
                manifest["status"] = (
                    "aborted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                )
                manifest["failed_at"] = _timestamp()
                manifest["failed_stage"] = stage
            if isinstance(error, Exception):
                raise RuntimeError(f"{wave} wave failed during {stage}") from error
            raise
        finally:
            # A failed model or gate never skips teardown.
            if branch is not None:
                api.delete_branch(project.project_id, branch.branch_id)
                if manifest:
                    manifest["neon"]["deleted"] = True
                    manifest["neon"]["deleted_at"] = _timestamp()
            if manifest:
                _create_exclusive(
                    paths["manifest"],
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    forbidden,
                )
                _create_exclusive(
                    paths["seal"],
                    json.dumps(_seal_payload(paths), indent=2, sort_keys=True) + "\n",
                    forbidden,
                )
    if wave == "canary" and manifest.get("status") == "proceed":
        _authorize_canary(
            example_root=example_root,
            authorization_path=authorization_path,
            manifest_path=paths["manifest"],
            spec_sha256=spec_sha256,
            attempt=attempt,
            prior=prior,
        )
    return manifest


def _authorize_canary(
    *,
    example_root: Path,
    authorization_path: Path,
    manifest_path: Path,
    spec_sha256: str,
    attempt: int | None,
    prior: Mapping[str, Any] | None,
) -> None:
    """Create the single exclusive authorization the full wave requires."""

    _allowed_artifact(example_root, authorization_path)
    payload: dict[str, Any] = {
        "benchmark_id": spec.BENCHMARK_ID,
        "spec_sha256": spec_sha256,
        "authorized_at": _timestamp(),
        "attempt": attempt,
        "attempt_status": "proceed",
        "attempt_manifest_path": manifest_path.resolve()
        .relative_to(example_root.resolve())
        .as_posix(),
        "attempt_manifest_sha256": _sha256_file(manifest_path),
        "attempt_seal_sha256": _sha256_file(manifest_path.parent / "seal.json"),
    }
    if attempt == spec.MAX_CANARY_ATTEMPTS:
        if prior is None or prior.get("status") != "infrastructure_failure":
            raise BenchmarkStateError(
                "attempt 2 can only be authorized alongside a sealed attempt-1 "
                "infrastructure_failure"
            )
        payload["prior_infrastructure_failure"] = {
            "status": prior["status"],
            "spec_sha256": prior["spec_sha256"],
        }
    _create_exclusive(
        authorization_path, json.dumps(payload, indent=2, sort_keys=True) + "\n", ()
    )


SECRET_PATTERNS = (
    re.compile(r"postgres(?:ql)?://[^\s\"']+"),
    re.compile(r"neon_api_key", re.IGNORECASE),
    re.compile(r"\bnapi_[A-Za-z0-9]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
)


def assert_no_secrets(text: str, *, label: str) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise BenchmarkStateError(f"{label} contains secret-like content")


def verify_v2_benchmark(
    manifest_path: Path,
    *,
    example_root: Path,
    require_status: str | None = None,
    require_decision: str | None = None,
) -> dict[str, Any]:
    """Reconcile one sealed wave against its spec, data, and raw rollouts.

    Integrity returns zero for any internally consistent terminal status. The
    executable product gates are ``--require-status`` and ``--require-decision``.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 1:
        # Retained compatibility path for the sealed v1 result.
        return verify_report_artifacts(manifest_path)
    if manifest.get("benchmark_id") != spec.BENCHMARK_ID:
        raise BenchmarkStateError("manifest records another benchmark")
    verify_predecessor(example_root, benchmark_id=spec.BENCHMARK_ID)

    spec_path = example_root / spec.SPEC_PATH
    frozen, spec_sha256 = read_spec(spec_path)
    if manifest["spec_sha256"] != spec_sha256:
        raise BenchmarkStateError("manifest does not reference the frozen spec")
    if manifest["environment_abi_sha256"] != frozen["environment"]["abi_sha256"]:
        raise BenchmarkStateError("manifest records a different environment ABI")
    seal = _verify_seal(manifest_path)

    wave = manifest["wave"]
    expected_root = _expected_wave_root(wave, manifest.get("attempt"))
    actual_root = manifest_path.parent.resolve().relative_to(example_root.resolve()).as_posix()
    if actual_root != expected_root:
        raise BenchmarkStateError(f"{wave} manifest is not at its frozen path")

    raw_path = manifest_path.parent / manifest["artifacts"]["raw_rollouts"]
    records = _load_jsonl(raw_path)
    if len(records) != manifest["rollout_count"]:
        raise BenchmarkStateError("raw rollout count does not match the manifest")
    rollout_ids = [record["rollout_id"] for record in records]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise BenchmarkStateError("a rollout identifier was reused")
    worlds_seen = {record["world_id"] for record in records}
    if len(worlds_seen) != len(records):
        raise BenchmarkStateError("a world identifier was reused")

    data_dir = example_root / spec.DATA_ROOT
    for name, key in (
        ("train.jsonl", "train_sha256"),
        ("eval.jsonl", "eval_sha256"),
        ("oracle_traces.jsonl", "oracle_sha256"),
    ):
        if _sha256_file(data_dir / name) != frozen["datasets"][key]:
            raise BenchmarkStateError(f"data/v2/{name} changed after the wave ran")
    rows = {
        row["task_id"]: row
        for path in ("train.jsonl", "eval.jsonl")
        for row in _load_jsonl(data_dir / path)
    }
    demos = load_v2_oracle_demos(data_dir / "oracle_traces.jsonl")
    demo_ids = set(frozen["selections"]["oracle_demo_task_ids"])

    arms = {arm["id"]: arm for arm in manifest["models"]["arms"]}
    expected_arms = {arm.id: arm for arm in spec.ARMS}
    for arm_id, arm in arms.items():
        if arm["model"] != expected_arms[arm_id].model or arm["prompt"] != (
            expected_arms[arm_id].prompt
        ):
            raise BenchmarkStateError(f"arm {arm_id} does not match the frozen mapping")

    execution = manifest["execution"]
    for record in records:
        task_id = record["task_id"]
        row = rows[task_id]
        for field in ("cell", "action_family", "outcome_class", "expected_disposition"):
            if record[field] != row[field]:
                raise BenchmarkStateError(f"{task_id}: {field} disagrees with the frozen data")
        if task_id in demo_ids and record["phase"] != "signal_probe":
            raise BenchmarkStateError(f"{task_id}: a frozen demo was also executed as a task")
        example = Example(id=task_id, payload=row)
        if arms[record["arm"]]["prompt"] == spec.TWO_SHOT_PROMPT:
            example = build_two_shot_example(example, demos)
        expected_initial = len(example.payload["prompt_messages"])
        if record["initial_message_count"] != expected_initial:
            raise BenchmarkStateError(f"{task_id}: transcript accounting excludes the wrong prefix")
        reply_count, disposition, tool_calls, invalid_calls = transcript_facts(
            record["messages"], initial_message_count=expected_initial
        )
        recomputed = {
            "reply_call_count": reply_count,
            "predicted_disposition": disposition,
            "tool_call_count": tool_calls,
            "invalid_tool_call_count": invalid_calls,
        }
        for key, value in recomputed.items():
            if record[key] != value:
                raise BenchmarkStateError(f"{task_id}: {key} disagrees with the transcript")
        expected_id = v2_rollout_id(
            spec_sha256=spec_sha256,
            wave=wave,
            attempt=manifest.get("attempt"),
            run_nonce=manifest["run_nonce"],
            arm=record["arm"],
            repetition=record["repetition"],
            task_id=task_id,
            sample=record["sample"],
        )
        if record["rollout_id"] != expected_id:
            raise BenchmarkStateError(f"{task_id}: rollout id is not derived from this wave")
        if record["world_id"] != world_id_for_rollout(record["rollout_id"]):
            raise BenchmarkStateError(f"{task_id}: world id is not derived from its rollout")

    selections = manifest["selections"]
    if wave == "canary":
        expected = {
            (arm_id, task_id) for arm_id in arms for task_id in selections["canary_task_ids"]
        }
        actual = [(record["arm"], record["task_id"]) for record in records]
        if set(actual) != expected or len(actual) != len(expected):
            raise BenchmarkStateError("canary does not contain every task exactly once per arm")
        recomputed_gates = evaluate_canary(records, manifest["gates"])
        if _canonical_json(recomputed_gates) != _canonical_json(manifest["canary"]):
            raise BenchmarkStateError("canary gate evaluation does not reconcile")
        status = manifest["status"]
        if status not in spec.CANARY_STATUSES:
            raise BenchmarkStateError(f"unrecognized canary status {status!r}")
        summary = {
            "wave": wave,
            "attempt": manifest.get("attempt"),
            "rollouts": len(records),
            "status": status,
            "failed_gates": recomputed_gates["failed_gates"],
        }
    else:
        _verify_full_membership(records, selections=selections, arms=arms, execution=execution)
        recomputed = build_v2_report(records)
        for key in ("arms", "stress", "signal_probe", "decision"):
            if _canonical_json(recomputed[key]) != _canonical_json(manifest["report"][key]):
                raise BenchmarkStateError(f"report section {key!r} does not reconcile")
        decision = manifest["report"]["decision"]["status"]
        if decision not in spec.FULL_DECISIONS:
            raise BenchmarkStateError(f"unrecognized full decision {decision!r}")
        if decision == "go" and not manifest["report"]["signal_probe"]["passes"]:
            raise BenchmarkStateError("go decision violates the mixed-reward signal threshold")
        summary = {
            "wave": wave,
            "rollouts": len(records),
            "status": manifest["status"],
            "decision": decision,
            "failed_gates": manifest["report"]["decision"]["failed_gates"],
        }

    if not manifest["neon"].get("deleted"):
        raise BenchmarkStateError("the disposable child branch was not recorded as deleted")
    assert_no_secrets(manifest_path.read_text(encoding="utf-8"), label=manifest_path.name)
    summary["sealed_artifacts"] = sorted(seal["sha256"])
    if require_status is not None and manifest["status"] != require_status:
        raise BenchmarkStateError(
            f"required status {require_status!r}, sealed status is {manifest['status']!r}"
        )
    if require_decision is not None:
        actual_decision = manifest.get("report", {}).get("decision", {}).get("status")
        if actual_decision != require_decision:
            raise BenchmarkStateError(
                f"required decision {require_decision!r}, sealed decision is {actual_decision!r}"
            )
    return summary


def _verify_full_membership(
    records: Sequence[Mapping[str, Any]],
    *,
    selections: Mapping[str, Any],
    arms: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    """Prove exact matrix membership using the manifest's own geometry."""

    full = [record for record in records if record["phase"] == "full"]
    expected_full = {
        (arm_id, task_id) for arm_id in arms for task_id in selections["eval_task_ids"]
    }
    actual_full = [(record["arm"], record["task_id"]) for record in full]
    if set(actual_full) != expected_full or len(actual_full) != len(expected_full):
        raise BenchmarkStateError("full matrix does not contain every task exactly once per arm")

    stress = [record for record in records if record["phase"] == "stress"]
    expected_stress = {
        (arm_id, task_id, repetition)
        for arm_id in arms
        for task_id in selections["stress_task_ids"]
        for repetition in range(1, int(execution["stress_repeats"]) + 1)
    }
    actual_stress = [
        (record["arm"], record["task_id"], record["repetition"]) for record in stress
    ]
    if set(actual_stress) != expected_stress or len(actual_stress) != len(expected_stress):
        raise BenchmarkStateError("stress matrix does not contain the manifest's repeats")

    probe_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["phase"] == "signal_probe":
            probe_groups[record["task_id"]].append(record)
    if set(probe_groups) != set(selections["signal_probe_task_ids"]):
        raise BenchmarkStateError("signal probe does not use every frozen task")
    if any(
        len(group) != int(execution["training_group_size"]) for group in probe_groups.values()
    ):
        raise BenchmarkStateError("signal probe does not use the manifest's group size")
    if {record["arm"] for group in probe_groups.values() for record in group} != {
        spec.BASE_ARM_ID
    }:
        raise BenchmarkStateError("signal probe must run only the small base arm")

    unknown = sorted({record["phase"] for record in records} - {"full", "stress", "signal_probe"})
    if unknown:
        raise BenchmarkStateError(f"raw rollouts contain unknown phases: {', '.join(unknown)}")


def check_v2_report(manifest_path: Path, *, example_root: Path) -> dict[str, Any]:
    """Re-render the report into a temporary file and compare bytes only."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("wave") != "full":
        raise BenchmarkStateError("only the full wave renders an HTML report")
    _verify_seal(manifest_path)
    raw_path = manifest_path.parent / manifest["artifacts"]["raw_rollouts"]
    html_path = manifest_path.parent / manifest["artifacts"]["html_report"]
    records = _load_jsonl(raw_path)
    report = build_v2_report(records)
    for key in ("arms", "stress", "signal_probe", "decision"):
        if _canonical_json(report[key]) != _canonical_json(manifest["report"][key]):
            raise BenchmarkStateError(f"report section {key!r} does not reconcile")
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "report.html"
        render_html_report(
            template_path=example_root / "templates" / "report.html",
            output_path=candidate,
            # generated_at is a timestamp; compare the accepted report's own value.
            report={**report, "generated_at": manifest["report"]["generated_at"]},
            demo_task_ids=manifest["selections"]["report_demo_task_ids"],
            records=records,
        )
        if candidate.read_bytes() != html_path.read_bytes():
            raise BenchmarkStateError("rendered report does not match the sealed HTML")
    return {"html_sha256": _sha256_file(html_path), "rollouts": len(records)}


__all__ = [
    "CONCURRENT_GROUPS",
    "FRONTIER_MODELS",
    "SMALL_MODEL",
    "STRESS_REPEATS",
    "TRAINING_GROUP_SIZE",
    "BenchmarkStateError",
    "assert_no_secrets",
    "build_v2_report",
    "check_v2_report",
    "evaluate_canary",
    "run_v2_wave",
    "v2_decision",
    "verify_v2_benchmark",
    "assert_bundle_matches_abi",
    "build_frozen_manifest",
    "build_report",
    "build_two_shot_example",
    "build_v2_spec",
    "environment_abi",
    "environment_abi_sha256",
    "freeze_task_selection",
    "freeze_v2_spec",
    "load_v2_oracle_demos",
    "read_spec",
    "refresh_report_artifacts",
    "run_baseline",
    "summarize_records",
    "transcript_facts",
    "v2_rollout_id",
    "v2_task_selection",
    "verify_report_artifacts",
    "wilson_interval",
]
