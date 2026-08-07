"""Redacted deterministic replay for the six frozen report cases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from benchmax.envs import BaseRollout

from order_resolution.database import Database
from order_resolution.fixtures import delete_operational_world
from order_resolution.grading import capture_world_snapshot
from order_resolution.order_env import TOOLS, OrderResolutionEnv, world_id_for_rollout
from order_resolution.schema import audit_events, command_receipts

_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_CREDENTIAL_URL = re.compile(r"postgres(?:ql)?://[^\s/:]+:[^\s@]+@", re.IGNORECASE)
_SENSITIVE_KEYS = frozenset(
    {
        "address",
        "city",
        "content",
        "country",
        "customer_email",
        "email",
        "line1",
        "line2",
        "name",
        "postal_code",
        "region",
    }
)
_IDENTIFIER_KEYS = frozenset(
    {
        "address_id",
        "case_id",
        "customer_id",
        "entity_id",
        "event_id",
        "message_id",
        "order_id",
        "order_item_id",
        "order_number",
        "receipt_id",
        "request_id",
        "rollout_id",
        "variant_id",
        "warehouse_id",
        "world_id",
    }
)
_KNOWN_TOOLS = frozenset(tool["function"]["name"] for tool in TOOLS)


def demo_arm(manifest: dict[str, Any]) -> str:
    """v1 replayed its two-shot arm; v2 replays whichever full arm scored best."""

    if manifest.get("schema_version") != 2:
        return "small_two_shot"
    arms = manifest["report"]["arms"]
    return max(sorted(arms), key=lambda name: float(arms[name]["success_rate"]))


async def replay_frozen_demos(
    *,
    database_url: str,
    data_dir: Path,
    baseline_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Replay the captured best-arm actions and emit a non-PII artifact."""

    manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    raw_path = baseline_manifest_path.parent / manifest["artifacts"]["raw_rollouts"]
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    selections = manifest.get("selections") or manifest["datasets"]
    selected_ids = selections["report_demo_task_ids"]
    arm = demo_arm(manifest)
    selected_records = {
        record["task_id"]: record
        for record in records
        if record["phase"] == "full"
        and record["arm"] == arm
        and record["task_id"] in selected_ids
    }
    if set(selected_records) != set(selected_ids):
        raise RuntimeError("captured rollouts are missing a frozen demo task")

    env = OrderResolutionEnv(database_url, retain_demo_worlds=True)
    query_database = Database(database_url)
    try:
        dataset = await env.create_dataset("eval", data_dir)
        examples = {example.payload["task_id"]: example for example in dataset}
        demos = []
        for task_id in selected_ids:
            record = selected_records[task_id]
            example = examples[task_id]
            replay_id = f"frozen-demo-{task_id}"
            world_id = world_id_for_rollout(replay_id)
            tool_timeline: list[dict[str, Any]] = []
            try:
                async with env.rollout_context(replay_id, example):
                    before = await query_database.read(
                        lambda connection: capture_world_snapshot(connection, world_id)
                    )
                    for name, arguments in _generated_tool_calls(record):
                        if name not in _KNOWN_TOOLS:
                            result: Any = {"ok": False, "code": "UNKNOWN_TOOL"}
                        elif arguments is None:
                            result = {"ok": False, "code": "INVALID_ARGUMENT"}
                        else:
                            result = await env.run_tool(replay_id, name, **arguments)
                        tool_timeline.append(
                            {
                                "tool": name,
                                "arguments": redact_value(arguments),
                                "result": redact_value(result),
                            }
                        )
                    rollout = BaseRollout(
                        rollout_id=replay_id,
                        termination_reason=record["termination_reason"],
                        messages=record["messages"],
                        example_args={
                            key: value
                            for key, value in example.payload.items()
                            if key != "prompt_messages"
                        },
                        split="eval",
                    )
                    rewards = await env.compute_reward(rollout)
                    after = await query_database.read(
                        lambda connection: capture_world_snapshot(connection, world_id)
                    )
                    receipts, audits = await query_database.read(
                        lambda connection: _timeline_rows(connection, world_id)
                    )
                if rewards["task_success"] != record["task_success"]:
                    raise RuntimeError(f"demo replay reward drifted for {task_id}")
                demos.append(
                    {
                        "task_id": task_id,
                        "cell": example.payload["cell"],
                        "customer_request": _redacted_request(example.payload),
                        "detailed_arm": "small_two_shot",
                        "all_arm_results": _arm_results(records, task_id),
                        "tool_timeline": tool_timeline,
                        "customer_reply": redact_value(after["reply"]),
                        "normalized_state_diff": redact_value(_diff(before, after)),
                        "command_receipts": redact_value(receipts),
                        "audit_timeline": redact_value(audits),
                        "reward": rewards,
                        "model": record["model"],
                        "latency_seconds": record["latency_seconds"],
                        "usage": "omitted by user instruction",
                    }
                )
            finally:
                await query_database.transaction(
                    lambda connection: delete_operational_world(connection, world_id)
                )
    finally:
        await env.aclose()
        await query_database.aclose()
    artifact = {
        "schema_version": 1,
        "baseline_manifest": baseline_manifest_path.name,
        "decision": manifest["report"]["decision"]["status"],
        "detailed_arm": "small_two_shot",
        "demos": demos,
    }
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    assert_redacted_artifact(serialized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def _generated_tool_calls(record: Mapping[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    calls: list[tuple[str, dict[str, Any] | None]] = []
    messages = record["messages"][int(record["initial_message_count"]) :]
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                arguments = None
            calls.append((name, arguments))
    return calls


async def _timeline_rows(
    connection, world_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receipts = (
        (
            await connection.execute(
                sa.select(
                    command_receipts.c.receipt_id,
                    command_receipts.c.command_name,
                    command_receipts.c.request_hash,
                    command_receipts.c.result,
                    command_receipts.c.created_at,
                )
                .where(command_receipts.c.world_id == world_id)
                .order_by(command_receipts.c.created_at, command_receipts.c.receipt_id)
            )
        )
        .mappings()
        .all()
    )
    audits = (
        (
            await connection.execute(
                sa.select(
                    audit_events.c.event_seq,
                    audit_events.c.actor,
                    audit_events.c.action,
                    audit_events.c.entity_type,
                    audit_events.c.entity_id,
                    audit_events.c.before_state,
                    audit_events.c.after_state,
                    audit_events.c.occurred_at,
                    audit_events.c.request_id,
                )
                .where(audit_events.c.world_id == world_id)
                .order_by(audit_events.c.event_seq)
            )
        )
        .mappings()
        .all()
    )
    return [_jsonable(dict(row)) for row in receipts], [_jsonable(dict(row)) for row in audits]


def _arm_results(records: Sequence[Mapping[str, Any]], task_id: str) -> dict[str, Any]:
    return {
        record["arm"]: {
            "task_success": record["task_success"],
            "structured_reply_correct": record["rewards"].get("_structured_reply_correct", 0.0),
            "forbidden_mutation": record["rewards"].get("_forbidden_mutation", 0.0),
        }
        for record in records
        if record["phase"] == "full" and record["task_id"] == task_id
    }


def _redacted_request(payload: Mapping[str, Any]) -> str:
    action = str(payload["action_family"]).replace("_", " ")
    outcome = str(payload["outcome_class"]).replace("_", " ")
    return f"customer requests {action}; scenario class: {outcome}; identifying details redacted"


def _diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in before:
                changes.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": child, "before": before[key], "after": None})
            else:
                changes.extend(_diff(before[key], after[key], child))
        return changes
    if before != after:
        return [{"path": path, "before": before, "after": after}]
    return []


def redact_value(value: Any, key: str | None = None) -> Any:
    if key in _SENSITIVE_KEYS:
        return "[redacted]"
    if key == "path" and isinstance(value, str):
        return ".".join(_public_mapping_key(part) for part in value.split("."))
    if key in _IDENTIFIER_KEYS and value is not None:
        return _pseudonym(value)
    if isinstance(value, Mapping):
        return {
            _public_mapping_key(str(item_key)): redact_value(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        if _EMAIL.search(value):
            return _EMAIL.sub("[redacted-email]", value)
        if _CREDENTIAL_URL.search(value):
            return "[redacted-credential-url]"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _public_mapping_key(value: str) -> str:
    if value.startswith(("OR-", "item-", "address-", "variant-", "warehouse-")):
        return _pseudonym(value)
    return value


def _pseudonym(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:10]
    return f"ref-{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def assert_redacted_artifact(serialized: str) -> None:
    if _CREDENTIAL_URL.search(serialized) or "NEON_API_KEY" in serialized:
        raise RuntimeError("demo artifact contains a credential-like value")
    if _EMAIL.search(serialized):
        raise RuntimeError("demo artifact contains an email-like value")


__all__ = ["assert_redacted_artifact", "redact_value", "replay_frozen_demos"]
