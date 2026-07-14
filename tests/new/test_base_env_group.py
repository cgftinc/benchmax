from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from openai import InternalServerError

from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    Dataset,
    DatasetSplit,
    Example,
    FrozenDataset,
    JsonRow,
    RewardMap,
    RolloutRequest,
    Tool,
)
from benchmax.envs.base.openai_types import Messages
from tests.new.fakes.model_server import LocalModelServer, completion_response


class _MathEnv(BaseEnv):
    def __init__(
        self,
        *,
        max_turns: int = 1,
        max_tool_calls: int | None = None,
    ) -> None:
        super().__init__(max_turns=max_turns, max_tool_calls=max_tool_calls)
        self.reward_calls: list[tuple[str, Messages, Mapping[str, Any], str]] = []

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[JsonRow]:
        return FrozenDataset([])

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> RewardMap:
        self.reward_calls.append(
            (
                rollout.rollout_id,
                rollout.messages,
                rollout.example_args,
                rollout.termination_reason,
            )
        )
        answer = str(rollout.messages[-1].get("content", ""))
        expected = str(rollout.example_args["answer"])
        return {
            "correctness": float(
                rollout.termination_reason == "finished" and answer == expected
            )
        }


class _ToolMathEnv(_MathEnv):
    def __init__(self, *, max_tool_calls: int) -> None:
        super().__init__(max_turns=2, max_tool_calls=max_tool_calls)
        self.tool_calls: list[tuple[str, str, Mapping[str, Any]]] = []

    async def list_tools(self) -> list[Tool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two integers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "left": {"type": "integer"},
                            "right": {"type": "integer"},
                        },
                        "required": ["left", "right"],
                    },
                },
            }
        ]

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> Any:
        self.tool_calls.append((rollout_id, tool_name, tool_args))
        return {"value": tool_args["left"] * tool_args["right"]}


def _requests(
    server: LocalModelServer,
    example: Example[JsonRow],
    rollout_ids: Sequence[str],
) -> list[RolloutRequest[JsonRow]]:
    return [
        RolloutRequest(
            rollout_id=rollout_id,
            example=example,
            model="test-model",
            base_url=server.base_url(rollout_id),
            api_key=f"session-key-{rollout_id}",
        )
        for rollout_id in rollout_ids
    ]


async def test_base_env_group_runs_end_to_end_through_distinct_http_endpoints() -> None:
    rollout_ids = ("rollout-1", "rollout-2", "rollout-3")
    example = Example(
        id="math-42",
        payload={
            "prompt_messages": [{"role": "user", "content": "What is 6 times 7?"}],
            "answer": "42",
        },
    )
    env = _MathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(content="42"),
        ),
        concurrent_calls=len(rollout_ids),
    ) as server:
        requests = _requests(server, example, rollout_ids)
        outcomes = await env.run_group(requests)

    assert set(outcomes) == set(rollout_ids)
    assert all(outcome.rewards == {"correctness": 1.0} for outcome in outcomes.values())
    assert {request.session_id for request in server.requests} == set(rollout_ids)
    assert {request.path for request in server.requests} == {
        f"/sessions/{rollout_id}/v1/chat/completions" for rollout_id in rollout_ids
    }
    assert {
        request.session_id: request.authorization for request in server.requests
    } == {rollout_id: f"Bearer session-key-{rollout_id}" for rollout_id in rollout_ids}
    assert all(
        request.body["messages"] == [{"role": "user", "content": "What is 6 times 7?"}]
        for request in server.requests
    )
    assert {call[0] for call in env.reward_calls} == set(rollout_ids)
    assert all(call[2] == {"answer": "42"} for call in env.reward_calls)


async def test_context_exceeded_is_still_scored() -> None:
    example = Example(
        id="math-context",
        payload={
            "prompt_messages": [{"role": "user", "content": "Keep writing"}],
            "answer": "done",
        },
    )
    env = _MathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(content="partial", finish_reason="length"),
        )
    ) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    assert outcomes["rollout-1"].termination_reason == "context_exceeded"
    assert outcomes["rollout-1"].rewards == {"correctness": 0.0}
    assert env.reward_calls[0][3] == "context_exceeded"


async def test_model_infrastructure_failure_is_not_scored_or_retried() -> None:
    example = Example(
        id="math-infra",
        payload={
            "prompt_messages": [{"role": "user", "content": "hello"}],
            "answer": "hello",
        },
    )
    env = _MathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            500,
            {
                "error": {
                    "message": "model unavailable",
                    "type": "server_error",
                    "code": "unavailable",
                }
            },
        )
    ) as server:
        with pytest.raises(InternalServerError):
            await env.run_group(_requests(server, example, ["rollout-1"]))

    assert len(server.requests) == 1
    assert env.reward_calls == []


async def test_base_env_dispatches_an_advertised_tool_and_continues() -> None:
    def respond(session_id, call_index, body):
        if call_index == 0:
            return 200, completion_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "arguments": '{"left":6,"right":7}',
                        },
                    },
                ],
            )

        assert body["messages"] == [
            {"role": "user", "content": "What is 6 × 7?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "arguments": '{"left":6,"right":7}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"value":42}',
            },
        ]
        return 200, completion_response(content="42")

    example = Example(
        id="math-tool",
        payload={
            "prompt_messages": [{"role": "user", "content": "What is 6 × 7?"}],
            "answer": "42",
        },
    )
    env = _ToolMathEnv(max_tool_calls=1)

    with LocalModelServer(respond) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    assert outcomes["rollout-1"].rewards == {"correctness": 1.0}
    assert env.tool_calls == [("rollout-1", "multiply", {"left": 6, "right": 7})]


async def test_model_authored_tool_errors_are_returned_to_the_model() -> None:
    tool_calls = [
        {
            "id": "unknown-call",
            "type": "function",
            "function": {"name": "divide", "arguments": '{"left":6,"right":3}'},
        },
        {
            "id": "invalid-json-call",
            "type": "function",
            "function": {"name": "multiply", "arguments": "{invalid"},
        },
        {
            "id": "non-object-call",
            "type": "function",
            "function": {"name": "multiply", "arguments": "[6,7]"},
        },
    ]

    def respond(session_id, call_index, body):
        if call_index == 0:
            return 200, completion_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=tool_calls,
            )

        tool_messages = body["messages"][-3:]
        assert [message["tool_call_id"] for message in tool_messages] == [
            "unknown-call",
            "invalid-json-call",
            "non-object-call",
        ]
        assert tool_messages[0]["content"] == "Unknown tool: divide"
        assert tool_messages[1]["content"].startswith("Invalid JSON tool arguments:")
        assert tool_messages[2]["content"] == "Tool arguments must be a JSON object."
        return 200, completion_response(content="42")

    example = Example(
        id="math-tool-errors",
        payload={
            "prompt_messages": [{"role": "user", "content": "Use the tools"}],
            "answer": "42",
        },
    )
    env = _ToolMathEnv(max_tool_calls=len(tool_calls))

    with LocalModelServer(respond) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    assert outcomes["rollout-1"].rewards == {"correctness": 1.0}
    assert env.tool_calls == []


async def test_tool_budget_termination_is_scored_without_executing_tools() -> None:
    example = Example(
        id="math-tool-budget",
        payload={
            "prompt_messages": [{"role": "user", "content": "What is 6 × 7?"}],
            "answer": "42",
        },
    )
    env = _ToolMathEnv(max_tool_calls=0)

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "arguments": '{"left":6,"right":7}',
                        },
                    }
                ],
            ),
        )
    ) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    outcome = outcomes["rollout-1"]
    assert outcome.termination_reason == "tool_budget_exceeded"
    assert outcome.rewards == {"correctness": 0.0}
    assert len(server.requests) == 1
    assert env.tool_calls == []


async def test_base_env_group_only_reward_receives_complete_rollouts() -> None:
    class RelativeMathEnv(BaseEnv):
        def __init__(self) -> None:
            super().__init__(max_turns=1)
            self.group_rollouts: list[BaseRollout] = []

        async def create_dataset(self, split, base_dir):
            return FrozenDataset([])

        async def compute_group_rewards(
            self,
            rollouts: Sequence[BaseRollout],
        ) -> Mapping[str, RewardMap]:
            self.group_rollouts = list(rollouts)
            contents = [
                rollout.messages[-1].get("content") for rollout in self.group_rollouts
            ]
            assert all(isinstance(content, str) for content in contents)
            values = [
                float(content) for content in contents if isinstance(content, str)
            ]
            maximum = max(values)
            return {
                rollout.rollout_id: {"relative": float(value == maximum)}
                for rollout, value in zip(rollouts, values, strict=True)
            }

    example = Example(
        id="math-relative",
        payload={
            "prompt_messages": [{"role": "user", "content": "Give a number"}],
            "answer": "unused",
        },
    )
    env = RelativeMathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(content=session_id.removeprefix("rollout-")),
        ),
        concurrent_calls=3,
    ) as server:
        outcomes = await env.run_group(
            _requests(server, example, ["rollout-1", "rollout-2", "rollout-3"])
        )

    assert len(env.group_rollouts) == 3
    assert all(rollout.rewards is None for rollout in env.group_rollouts)
    assert outcomes["rollout-1"].rewards == {"relative": 0.0}
    assert outcomes["rollout-2"].rewards == {"relative": 0.0}
    assert outcomes["rollout-3"].rewards == {"relative": 1.0}


async def test_base_env_merges_individual_and_group_reward_dimensions() -> None:
    class CombinedMathEnv(_MathEnv):
        async def compute_group_rewards(
            self,
            rollouts: Sequence[BaseRollout],
        ) -> Mapping[str, RewardMap]:
            assert all(rollout.rewards == {"correctness": 1.0} for rollout in rollouts)
            return {
                rollout.rollout_id: {"group_rank": float(index)}
                for index, rollout in enumerate(rollouts)
            }

    example = Example(
        id="math-combined",
        payload={
            "prompt_messages": [{"role": "user", "content": "Answer"}],
            "answer": "42",
        },
    )
    env = CombinedMathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(content="42"),
        ),
        concurrent_calls=2,
    ) as server:
        outcomes = await env.run_group(
            _requests(server, example, ["rollout-1", "rollout-2"])
        )

    assert outcomes["rollout-1"].rewards == {
        "correctness": 1.0,
        "group_rank": 0.0,
    }
    assert outcomes["rollout-2"].rewards == {
        "correctness": 1.0,
        "group_rank": 1.0,
    }


async def test_base_env_rejects_duplicate_individual_and_group_reward_keys() -> None:
    class ConflictingMathEnv(_MathEnv):
        async def compute_group_rewards(
            self,
            rollouts: Sequence[BaseRollout],
        ) -> Mapping[str, RewardMap]:
            return {rollout.rollout_id: {"correctness": 0.5} for rollout in rollouts}

    example = Example(
        id="math-conflict",
        payload={
            "prompt_messages": [{"role": "user", "content": "Answer"}],
            "answer": "42",
        },
    )
    env = ConflictingMathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(content="42"),
        )
    ) as server:
        with pytest.raises(ValueError, match="duplicate keys.*correctness"):
            await env.run_group(_requests(server, example, ["rollout-1"]))
