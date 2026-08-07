from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmax.envs import Example, canonical_example_id
from order_resolution.benchmark import (
    build_frozen_manifest,
    build_report,
    build_two_shot_example,
    freeze_task_selection,
    summarize_records,
    transcript_facts,
    wilson_interval,
)
from order_resolution.branching import RuntimeBranch


def _examples(split: str, count: int):
    return [
        Example(
            id=f"canonical-{split}-{cell}-{index}",
            payload={
                "task_id": f"{split}-{cell}-{index:02d}",
                "cell": cell,
                "action_family": cell.rsplit("-", 1)[0],
                "outcome_class": cell.rsplit("-", 1)[1],
                "expected_disposition": "completed",
                "prompt_messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "request"},
                ],
            },
        )
        for cell in (
            "cancel_item-execute",
            "cancel_item-clarify",
            "cancel_item-deny",
            "change_address-execute",
            "change_address-clarify",
            "change_address-deny",
            "replace_variant-execute",
            "replace_variant-clarify",
            "replace_variant-deny",
        )
        for index in range(count)
    ]


def _record(
    *,
    arm: str = "small_base",
    phase: str = "full",
    task_id: str = "eval-cancel_item-execute-00",
    success: float = 1.0,
    group_id: str = "group",
    action_family: str = "cancel_item",
) -> dict:
    return {
        "phase": phase,
        "arm": arm,
        "model": "model",
        "task_id": task_id,
        "group_id": group_id,
        "cell": f"{action_family}-execute",
        "action_family": action_family,
        "expected_disposition": "completed",
        "predicted_disposition": "completed",
        "task_success": success,
        "rewards": {
            "task_success": success,
            "_required_state_fraction": success,
            "_forbidden_mutation": 0.0,
            "_correct_disposition": success,
            "_structured_reply_correct": success,
            "_invariant_failure": 0.0,
            "_unnecessary_tool_calls": 0.0,
        },
        "termination_reason": "finished",
        "error_present": False,
        "latency_seconds": 1.0,
        "reply_call_count": 1,
        "tool_call_count": 2,
        "invalid_tool_call_count": 0,
    }


def test_frozen_task_selection_is_balanced_and_pre_result() -> None:
    selected = freeze_task_selection(_examples("eval", 10), _examples("train", 20))

    assert len(selected["eval_task_ids"]) == 90
    assert len(selected["stress_task_ids"]) == 27
    assert len(selected["signal_probe_task_ids"]) == 36
    assert len(selected["report_demo_task_ids"]) == 6
    assert len(set(selected["stress_task_ids"])) == 27
    assert len(set(selected["signal_probe_task_ids"])) == 36


def test_two_shot_prompt_keeps_one_system_and_target_identity() -> None:
    target = _examples("eval", 10)[0]
    demos = [
        {
            "prompt_messages": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": f"demo {index}"},
            ],
            "completion_messages": [{"role": "assistant", "content": f"answer {index}"}],
        }
        for index in range(2)
    ]

    prompted = build_two_shot_example(target, demos)

    # The augmented payload is a different example, so its canonical id moves.
    assert prompted.id != target.id
    assert prompted.id == canonical_example_id(prompted.payload)
    assert [message["role"] for message in prompted.payload["prompt_messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert prompted.payload["prompt_messages"][-1]["content"] == "request"


def test_frozen_manifest_records_nonsecret_contract() -> None:
    example_root = Path(__file__).parents[1]
    demos = [
        {
            "prompt_messages": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": f"demo {index}"},
            ],
            "completion_messages": [{"role": "assistant", "content": "done"}],
        }
        for index in range(2)
    ]
    branch = RuntimeBranch(
        project_id="project",
        parent_branch_id="parent",
        branch_id="child",
        branch_name="baseline-child",
        endpoint_id="endpoint",
        expires_at="2026-08-07T00:00:00Z",
        database_name="order_resolution",
        runtime_role_name="runtime",
        admin_database_url="admin-secret",
        runtime_database_url="runtime-secret",
    )

    manifest = build_frozen_manifest(
        example_root=example_root,
        branch=branch,
        eval_examples=_examples("eval", 10),
        train_examples=_examples("train", 20),
        demos=demos,
        available_models=["qwen3.5-4b", "gpt-5.6-sol", "grok-4.3"],
        base_url="https://llm.example.test/v1",
    )

    assert manifest["environment"]["schema_head"] == "cc6287f220ec"
    assert manifest["execution"]["training_group_size"] == 8
    assert len(manifest["datasets"]["eval_task_ids"]) == 90
    assert "admin-secret" not in json.dumps(manifest)
    assert "runtime-secret" not in json.dumps(manifest)


def test_summary_and_wilson_interval() -> None:
    records = [_record(success=1.0), _record(success=0.0, task_id="failed")]
    summary = summarize_records(records)

    assert summary["success_rate"] == 0.5
    assert summary["valid_tool_call_rate"] == 1.0
    assert summary["model_attributable_failures"] == 1
    low, high = wilson_interval(1, 2)
    assert low < 0.5 < high


def test_transcript_metrics_exclude_two_shot_demonstrations() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "reply_to_customer",
                        "arguments": '{"disposition":"cannot_complete"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"ok":true}'},
        {"role": "user", "content": "live task"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "reply_to_customer",
                        "arguments": '{"disposition":"completed"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"ok":true}'},
    ]

    facts = transcript_facts(messages, initial_message_count=4)

    assert facts == (1, "completed", 1, 0)


def test_report_applies_all_headroom_and_signal_gates() -> None:
    records: list[dict] = []
    rates = {
        "small_base": 0.5,
        "small_two_shot": 0.6,
        "frontier_gpt": 0.8,
        "frontier_grok": 0.7,
    }
    for arm, rate in rates.items():
        for index in range(20):
            records.append(
                _record(
                    arm=arm,
                    task_id=f"{arm}-{index}",
                    success=float(index < round(rate * 20)),
                )
            )
    for family in ("cancel_item", "change_address", "replace_variant"):
        for group_index in range(3):
            for sample in range(8):
                records.append(
                    _record(
                        phase="signal_probe",
                        task_id=f"{family}-{group_index}",
                        group_id=f"{family}-{group_index}",
                        action_family=family,
                        success=float(sample % 2 == 0),
                    )
                )

    report = build_report(records)

    assert report["signal_probe"]["passes"] is True
    assert report["decision"]["status"] == "go"


def test_frozen_eval_file_remains_valid_jsonl() -> None:
    path = Path(__file__).parents[1] / "data" / "eval.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 90


@pytest.mark.parametrize(("successes", "total"), [(0, 0), (0, 10), (10, 10)])
def test_wilson_bounds(successes: int, total: int) -> None:
    low, high = wilson_interval(successes, total)
    assert 0.0 <= low <= high <= 1.0
