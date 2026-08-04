from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from collection import aggregate_shards, enrich_records
from harness.autocompact import SUMMARY_HEADER, run_autocompact_agent


def assistant(content: str | None = None, *, call_id: str | None = None, tool: str | None = None):
    message = {"role": "assistant", "content": content}
    if call_id and tool:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": "{}"},
            }
        ]
    return message


class FakeAdapter:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append((json.loads(json.dumps(messages)), json.loads(json.dumps(tools))))
        message = next(self.responses)
        return SimpleNamespace(
            message=message,
            text=message.get("content") or "",
            input_tokens=10,
            output_tokens=2,
        )

    def make_tool_result_messages(self, results):
        return [
            {"role": "tool", "tool_call_id": call_id, "content": result}
            for call_id, result in results
        ]

    def make_system_message(self, content):
        return {"role": "system", "content": content}

    def make_user_message(self, content):
        return {"role": "user", "content": content}


class FakeExecutor:
    def __init__(self):
        self.executed = []

    def execute(self, tool_name, arguments):
        self.executed.append((tool_name, arguments))
        return f"result:{tool_name}"

    def get_metrics(self):
        return {"tool_calls": len(self.executed)}


class RepairingJudge:
    input_tokens = 3
    output_tokens = 4

    def __init__(self):
        self.action_count = 0
        self.continuation_context = None

    def review_action(self, visible_context, candidate, **kwargs):
        del visible_context, candidate, kwargs
        self.action_count += 1
        if self.action_count == 2:
            return {"decision": "replace_with_compact", "reason_code": "phase_complete"}
        return {"decision": "keep", "reason_code": "useful_action"}

    def review_summary(self, visible_context, candidate):
        assert any("result:read" in str(message.get("content")) for message in visible_context)
        assert candidate == f"{SUMMARY_HEADER}\nMissing the target."
        return {
            "decision": "repair",
            "corrected_summary": f"{SUMMARY_HEADER}\nTarget: agreement.pdf. Next: draft.",
        }

    def review_continuation(self, visible_context, candidate):
        self.continuation_context = json.loads(json.dumps(visible_context))
        assert candidate["tool_calls"][0]["function"]["name"] == "read"
        return {
            "decision": "repair",
            "corrected_message": assistant(call_id="edit-1", tool="edit"),
        }


def test_one_compaction_emits_three_call_level_records(tmp_path: Path) -> None:
    adapter = FakeAdapter(
        [
            assistant(call_id="read-1", tool="read"),
            assistant(call_id="discarded", tool="read"),
            assistant(f"{SUMMARY_HEADER}\nMissing the target."),
            assistant(call_id="repeat", tool="read"),
            assistant("done"),
        ]
    )
    executor = FakeExecutor()
    judge = RepairingJudge()
    trajectory_path = tmp_path / "trajectory.json"

    result = run_autocompact_agent(
        adapter=adapter,
        judge=judge,
        system_prompt="system",
        user_prompt="Review the agreement",
        tool_executor=executor,
        tools=[{"name": "read", "description": "read", "parameters": {"type": "object"}},
               {"name": "edit", "description": "edit", "parameters": {"type": "object"}}],
        max_turns=10,
        max_compactions=2,
        trajectory_path=trajectory_path,
    )

    trajectory = json.loads(trajectory_path.read_text())
    records = trajectory["sft_records"]
    assert result["finished_cleanly"] is True
    assert result["sft_record_count"] == 3
    assert [record["category"] for record in records] == [
        "autocompact_trigger",
        "autocompact_summary",
        "autocompact_continuation",
    ]
    assert [name for name, _ in executor.executed] == ["read", "edit"]
    assert "discarded" not in json.dumps(judge.continuation_context)
    assert "result:read" in json.dumps(judge.continuation_context)
    assert "Target: agreement.pdf" in json.dumps(judge.continuation_context)
    for record in records:
        assert record["completion_messages"][0]["step_loss_mask"] == 1
        assert all(
            message.get("step_loss_mask") == 0
            for message in record["prompt_messages"]
            if message["role"] == "assistant"
        )


def test_enrichment_and_aggregation_keep_triplets_task_disjoint(tmp_path: Path) -> None:
    base_record = {
        "id": "compact-0-trigger",
        "category": "autocompact_trigger",
        "source_kind": "judge_guided_on_policy",
        "prompt_messages": [{"role": "assistant", "content": "old", "step_loss_mask": 0}],
        "completion_messages": [{"role": "assistant", "content": "new", "step_loss_mask": 1}],
        "tools": [],
        "task": {"compaction_event_id": 0},
    }
    trajectory = {"sft_records": []}
    for suffix, category in (
        ("trigger", "autocompact_trigger"),
        ("summary", "autocompact_summary"),
        ("continuation", "autocompact_continuation"),
    ):
        row = json.loads(json.dumps(base_record))
        row["id"] = f"compact-0-{suffix}"
        row["category"] = category
        trajectory["sft_records"].append(row)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for index in range(12):
        rows = enrich_records(
            trajectory,
            example_id=f"task-{index}",
            trace_id=f"trace-{index}",
            rewards={"reward": float(index % 2)},
            termination_reason="finished",
        )
        (shard_dir / f"trace-{index}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    manifest = aggregate_shards(tmp_path)
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (tmp_path / "eval.jsonl").read_text().splitlines()]

    assert manifest["record_count"] == 36
    assert set(row["example_id"] for row in train).isdisjoint(
        row["example_id"] for row in evaluation
    )
    assert all(
        sum(row["example_id"] == task_id for row in train + evaluation) == 3
        for task_id in {row["example_id"] for row in train + evaluation}
    )
    assert manifest["passed_record_count"] == 18
