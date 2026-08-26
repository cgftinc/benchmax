from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from benchmax.auth import InjectedAuth, StaticBearerAuth
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    Dataset,
    DatasetSplit,
    Example,
    JsonRow,
    RewardMap,
    RolloutRequest,
    Tool,
)
from benchmax.envs.base.openai_types import Messages
from benchmax.rewards import JudgeError
from tests.unit.fakes.model_server import LocalModelServer, completion_response


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
        return Dataset([])

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
            "correctness": float(rollout.termination_reason == "finished" and answer == expected)
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


class _PartialToolMathEnv(_ToolMathEnv):
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
        final_message = rollout.messages[-1]
        tool_result = json.loads(str(final_message.get("content", "")))
        return {
            "correctness": float(
                rollout.termination_reason == "context_exceeded"
                and final_message.get("role") == "tool"
                and str(tool_result["value"]) == str(rollout.example_args["answer"])
            )
        }


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
            model_auth=StaticBearerAuth(f"session-key-{rollout_id}"),
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
    assert {request.session_id: request.authorization for request in server.requests} == {
        rollout_id: f"Bearer session-key-{rollout_id}" for rollout_id in rollout_ids
    }
    assert all(
        request.body["messages"] == [{"role": "user", "content": "What is 6 times 7?"}]
        for request in server.requests
    )
    assert {call[0] for call in env.reward_calls} == set(rollout_ids)
    assert all(call[2] == {"answer": "42"} for call in env.reward_calls)


async def test_output_exceeded_scores_the_partial_transcript() -> None:
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

    assert outcomes["rollout-1"].termination_reason == "output_exceeded"
    assert outcomes["rollout-1"].rewards == {"correctness": 0.0}
    assert len(env.reward_calls) == 1
    assert env.reward_calls[0][0] == "rollout-1"
    assert env.reward_calls[0][1][-1] == {
        "role": "assistant",
        "content": "partial",
    }
    assert env.reward_calls[0][3] == "output_exceeded"


async def test_gateway_context_exhaustion_scores_the_partial_transcript() -> None:
    example = Example(
        id="math-context",
        payload={
            "prompt_messages": [{"role": "user", "content": "Keep working"}],
            "answer": "42",
        },
    )
    env = _PartialToolMathEnv(max_tool_calls=1)

    def respond(session_id, call_index, body):
        if call_index == 0:
            return 200, completion_response(
                content=None,
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
            )
        return 400, {
            "error": {
                "message": (
                    "This model's maximum context length is 32 tokens for this "
                    "gateway session; no tokens remain for generation."
                ),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_budget_exceeded",
            }
        }

    with LocalModelServer(respond) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    outcome = outcomes["rollout-1"]
    assert outcome.termination_reason == "context_exceeded"
    assert outcome.rewards == {"correctness": 1.0}
    assert len(server.requests) == 2
    assert env.tool_calls == [("rollout-1", "multiply", {"left": 6, "right": 7})]
    assert len(env.reward_calls) == 1
    assert env.reward_calls[0][0] == "rollout-1"
    assert env.reward_calls[0][2:] == ({"answer": "42"}, "context_exceeded")
    assert env.reward_calls[0][1][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"value":42}',
    }


async def test_unrelated_bad_request_is_a_logged_model_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    example = Example(
        id="math-bad-request",
        payload={
            "prompt_messages": [{"role": "user", "content": "hello"}],
            "answer": "hello",
        },
    )
    env = _MathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            400,
            {
                "error": {
                    "message": "malformed messages",
                    "type": "invalid_request_error",
                    "code": "invalid_messages",
                }
            },
        )
    ) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    assert outcomes["rollout-1"].termination_reason == "model_error"
    assert outcomes["rollout-1"].rewards == {}
    assert env.reward_calls == []
    assert "malformed messages" in caplog.text


async def test_model_infrastructure_failure_is_empty_logged_and_not_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    assert len(server.requests) == 1
    assert outcomes["rollout-1"].termination_reason == "model_error"
    assert outcomes["rollout-1"].rewards == {}
    assert env.reward_calls == []
    assert "model unavailable" in caplog.text


async def test_empty_model_response_has_no_rewards_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    example = Example(
        id="math-empty-response",
        payload={
            "prompt_messages": [{"role": "user", "content": "hello"}],
            "answer": "hello",
        },
    )
    env = _MathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            {
                "id": "completion-empty",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [],
            },
        )
    ) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    assert outcomes["rollout-1"].termination_reason == "model_error"
    assert outcomes["rollout-1"].rewards == {}
    assert env.reward_calls == []
    assert "response contained no choices" in caplog.text


async def test_unbound_runtime_auth_is_a_model_failure_without_cancelling_sibling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    example = Example(
        id="math-auth",
        payload={
            "prompt_messages": [{"role": "user", "content": "answer"}],
            "answer": "42",
        },
    )
    env = _MathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (200, completion_response(content="42"))
    ) as server:
        requests = [
            RolloutRequest(
                rollout_id="rollout-1",
                example=example,
                model="test-model",
                base_url=server.base_url("rollout-1"),
                model_auth=InjectedAuth("missing"),
            ),
            RolloutRequest(
                rollout_id="rollout-2",
                example=example,
                model="test-model",
                base_url=server.base_url("rollout-2"),
                model_auth=StaticBearerAuth("session-key-rollout-2"),
            ),
        ]
        outcomes = await env.run_group(requests)

    assert outcomes["rollout-1"].termination_reason == "model_error"
    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-2"].termination_reason == "finished"
    assert outcomes["rollout-2"].rewards == {"correctness": 1.0}
    assert [request.session_id for request in server.requests] == ["rollout-2"]
    assert "No runtime model-auth provider" in caplog.text


async def test_failed_sibling_is_empty_and_excluded_from_group_scoring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RelativeMathEnv(_MathEnv):
        def __init__(self) -> None:
            super().__init__()
            self.group_ids: list[str] = []

        async def compute_group_rewards(self, rollouts):
            self.group_ids = [rollout.rollout_id for rollout in rollouts]
            return {rollout.rollout_id: {"relative": 1.0} for rollout in rollouts}

    example = Example(
        id="math-siblings",
        payload={
            "prompt_messages": [{"role": "user", "content": "answer"}],
            "answer": "42",
        },
    )
    env = RelativeMathEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (
            (500, {"error": {"message": "worker crashed", "type": "server_error"}})
            if session_id == "rollout-1"
            else (200, completion_response(content="42"))
        ),
        concurrent_calls=2,
    ) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1", "rollout-2"]))

    assert outcomes["rollout-1"].termination_reason == "model_error"
    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-2"].termination_reason == "finished"
    assert outcomes["rollout-2"].rewards == {
        "correctness": 1.0,
        "relative": 1.0,
    }
    assert env.group_ids == ["rollout-2"]
    assert "worker crashed" in caplog.text


async def test_judge_failure_empties_only_that_sibling_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class JudgedMathEnv(_MathEnv):
        async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
            if rollout.rollout_id == "rollout-1":
                raise JudgeError("judge unavailable")
            return await super().compute_reward(rollout)

    example = Example(
        id="math-judge",
        payload={
            "prompt_messages": [{"role": "user", "content": "answer"}],
            "answer": "42",
        },
    )
    env = JudgedMathEnv()
    with LocalModelServer(
        lambda session_id, call_index, body: (200, completion_response(content="42")),
        concurrent_calls=2,
    ) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1", "rollout-2"]))

    assert outcomes["rollout-1"].termination_reason == "judge_error"
    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-2"].rewards == {"correctness": 1.0}
    assert "judge unavailable" in caplog.text


async def test_base_env_dispatches_an_advertised_tool_and_continues() -> None:
    def respond(session_id, call_index, body):
        if call_index == 0:
            return 200, completion_response(
                content=None,
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
                "content": None,
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


async def test_base_env_resolves_tools_per_example_without_schema_leakage() -> None:
    class MixedTaskEnv(_ToolMathEnv):
        def __init__(self) -> None:
            super().__init__(max_tool_calls=1)
            self.resolved_row_kinds: list[str] = []

        async def list_tools_for_rollout(
            self,
            init_rollout_args: Mapping[str, Any] | None = None,
        ) -> list[Tool]:
            row_kind = None if init_rollout_args is None else init_rollout_args.get("row_kind")
            if not isinstance(row_kind, str):
                raise ValueError("row_kind is required")
            self.resolved_row_kinds.append(row_kind)
            if row_kind == "instruction_following":
                return []
            if row_kind == "ui":
                return await self.list_tools()
            raise ValueError(f"unsupported row_kind: {row_kind}")

    examples = {
        row_kind: Example(
            id=f"mixed-{row_kind}",
            payload={
                "prompt_messages": [{"role": "user", "content": "answer"}],
                "answer": "42",
                "row_kind": row_kind,
            },
        )
        for row_kind in ("ui", "instruction_following")
    }
    env = MixedTaskEnv()

    with LocalModelServer(
        lambda session_id, call_index, body: (200, completion_response(content="42")),
    ) as server:
        outcomes = {}
        for row_kind, example in examples.items():
            group = await env.run_group(
                [
                    RolloutRequest(
                        rollout_id=row_kind,
                        example=example,
                        model="test-model",
                        base_url=server.base_url(row_kind),
                        model_auth=StaticBearerAuth(f"session-key-{row_kind}"),
                    )
                ]
            )
            outcomes.update(group)

    request_bodies = {request.session_id: request.body for request in server.requests}
    assert request_bodies["ui"]["tools"][0]["function"]["name"] == "multiply"
    assert "tools" not in request_bodies["instruction_following"]
    assert sorted(env.resolved_row_kinds) == ["instruction_following", "ui"]
    assert all(outcome.rewards == {"correctness": 1.0} for outcome in outcomes.values())


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
                content=None,
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


async def test_unexpected_tool_failure_is_empty_logged_and_does_not_cancel_sibling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingToolEnv(_ToolMathEnv):
        async def run_tool(
            self,
            rollout_id: str,
            tool_name: str,
            **tool_args: Any,
        ) -> Any:
            if rollout_id == "rollout-1":
                raise RuntimeError("tool backend crashed")
            return await super().run_tool(rollout_id, tool_name, **tool_args)

    def respond(session_id, call_index, body):
        if call_index == 0:
            return 200, completion_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": f"call-{session_id}",
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "arguments": '{"left":6,"right":7}',
                        },
                    }
                ],
            )
        return 200, completion_response(content="42")

    example = Example(
        id="math-tool-failure",
        payload={
            "prompt_messages": [{"role": "user", "content": "What is 6 × 7?"}],
            "answer": "42",
        },
    )
    env = FailingToolEnv(max_tool_calls=1)

    with LocalModelServer(respond) as server:
        outcomes = await env.run_group(_requests(server, example, ["rollout-1", "rollout-2"]))

    assert outcomes["rollout-1"].termination_reason == "tool_error"
    assert outcomes["rollout-1"].rewards == {}
    assert outcomes["rollout-2"].termination_reason == "finished"
    assert outcomes["rollout-2"].rewards == {"correctness": 1.0}
    assert "tool backend crashed" in caplog.text


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
    assert [call[3] for call in env.reward_calls] == ["tool_budget_exceeded"]


@pytest.mark.parametrize(
    ("max_turns", "max_tool_calls", "termination_reason", "executed_tool_calls"),
    [
        (1, 1, "max_turns_exceeded", 1),
        (2, 0, "tool_budget_exceeded", 0),
    ],
)
async def test_budget_exhaustion_runs_individual_and_group_scoring(
    max_turns: int,
    max_tool_calls: int,
    termination_reason: str,
    executed_tool_calls: int,
) -> None:
    class BudgetScoringEnv(_ToolMathEnv):
        def __init__(self) -> None:
            super().__init__(max_tool_calls=max_tool_calls)
            self.max_turns = max_turns
            self.group_termination_reasons: list[str] = []

        async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
            self.reward_calls.append(
                (
                    rollout.rollout_id,
                    rollout.messages,
                    rollout.example_args,
                    rollout.termination_reason,
                )
            )
            return {"partial": 0.75}

        async def compute_group_rewards(
            self,
            rollouts: Sequence[BaseRollout],
        ) -> Mapping[str, RewardMap]:
            self.group_termination_reasons = [rollout.termination_reason for rollout in rollouts]
            return {rollout.rollout_id: {"group": 0.25} for rollout in rollouts}

    example = Example(
        id="math-budget-exhaustion",
        payload={
            "prompt_messages": [{"role": "user", "content": "What is 6 × 7?"}],
            "answer": "42",
        },
    )
    env = BudgetScoringEnv()

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
    assert outcome.termination_reason == termination_reason
    assert outcome.rewards == {"partial": 0.75, "group": 0.25}
    assert [call[3] for call in env.reward_calls] == [termination_reason]
    assert env.group_termination_reasons == [termination_reason]
    assert len(env.tool_calls) == executed_tool_calls


async def test_base_env_group_only_reward_receives_complete_rollouts() -> None:
    class RelativeMathEnv(BaseEnv):
        def __init__(self) -> None:
            super().__init__(max_turns=1)
            self.group_rollouts: list[BaseRollout] = []

        async def create_dataset(self, split, base_dir):
            return Dataset([])

        async def compute_group_rewards(
            self,
            rollouts: Sequence[BaseRollout],
        ) -> Mapping[str, RewardMap]:
            self.group_rollouts = list(rollouts)
            contents = [rollout.messages[-1].get("content") for rollout in self.group_rollouts]
            assert all(isinstance(content, str) for content in contents)
            values = [float(content) for content in contents if isinstance(content, str)]
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
        outcomes = await env.run_group(_requests(server, example, ["rollout-1", "rollout-2"]))

    assert outcomes["rollout-1"].rewards == {
        "correctness": 1.0,
        "group_rank": 0.0,
    }
    assert outcomes["rollout-2"].rewards == {
        "correctness": 1.0,
        "group_rank": 1.0,
    }


async def test_duplicate_individual_and_group_components_settle_the_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        outcomes = await env.run_group(_requests(server, example, ["rollout-1"]))

    outcome = outcomes["rollout-1"]
    assert outcome.termination_reason == "group_reward_error"
    assert outcome.rewards == {}
    assert "duplicate keys" in caplog.text


async def test_compute_reward_defect_keeps_a_budget_stop_label(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same precedence as HarborEnv: a budget stop outranks a scoring error."""

    class BrokenBudgetRewardEnv(_ToolMathEnv):
        def __init__(self) -> None:
            super().__init__(max_tool_calls=1)
            self.max_turns = 1

        async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
            raise KeyError("missing ground_truth column")

    example = Example(
        id="math-budget-broken-reward",
        payload={
            "prompt_messages": [{"role": "user", "content": "What is 6 × 7?"}],
            "answer": "42",
        },
    )

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(
                content=None,
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
        outcomes = await BrokenBudgetRewardEnv().run_group(
            _requests(server, example, ["rollout-1"])
        )

    outcome = outcomes["rollout-1"]
    assert outcome.termination_reason == "max_turns_exceeded"
    assert outcome.rewards == {}
    assert outcome.error == "KeyError: 'missing ground_truth column'"
    assert "missing ground_truth column" in caplog.text


async def test_compute_reward_defect_settles_the_rollout_as_reward_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenRewardEnv(_MathEnv):
        async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
            raise KeyError("missing ground_truth column")

    example = Example(
        id="math-broken-reward",
        payload={
            "prompt_messages": [{"role": "user", "content": "Answer"}],
            "answer": "42",
        },
    )

    with LocalModelServer(
        lambda session_id, call_index, body: (
            200,
            completion_response(content="42"),
        )
    ) as server:
        outcomes = await BrokenRewardEnv().run_group(_requests(server, example, ["rollout-1"]))

    outcome = outcomes["rollout-1"]
    assert outcome.termination_reason == "reward_error"
    assert outcome.rewards == {}
    assert "missing ground_truth column" in caplog.text


async def test_request_split_reaches_reward_hooks_and_must_be_uniform() -> None:
    """Envs may branch eval execution on ``rollout.split``; groups stay one-split."""

    class _SplitAwareEnv(_MathEnv):
        def __init__(self) -> None:
            super().__init__()
            self.seen_splits: list[str] = []

        async def compute_reward(self, rollout: BaseRollout) -> RewardMap:
            self.seen_splits.append(rollout.split)
            return await super().compute_reward(rollout)

    env = _SplitAwareEnv()
    example = Example(
        id="split-example",
        payload={"prompt_messages": [{"role": "user", "content": "2*2?"}], "answer": "4"},
    )

    def respond(session_id: str, call_index: int, body: dict[str, Any]):
        return 200, completion_response(content="4")

    with LocalModelServer(respond, concurrent_calls=2) as server:
        outcomes = await env.run_group(
            [
                RolloutRequest(
                    rollout_id=rollout_id,
                    example=example,
                    model="test-model",
                    base_url=server.base_url(rollout_id),
                    model_auth=StaticBearerAuth("k"),
                    split="eval",
                )
                for rollout_id in ("a", "b")
            ]
        )
    assert env.seen_splits == ["eval", "eval"]
    assert all(o.termination_reason == "finished" for o in outcomes.values())

    with LocalModelServer(respond, concurrent_calls=2) as server:
        mixed = [
            RolloutRequest(
                rollout_id="a",
                example=example,
                model="test-model",
                base_url=server.base_url("a"),
                model_auth=StaticBearerAuth("k"),
                split="train",
            ),
            RolloutRequest(
                rollout_id="b",
                example=example,
                model="test-model",
                base_url=server.base_url("b"),
                model_auth=StaticBearerAuth("k"),
                split="eval",
            ),
        ]
        with pytest.raises(ValueError, match="one split"):
            await env.run_group(mixed)


async def test_tool_content_parts_pass_through_to_the_transcript() -> None:
    """Rich tool results (image + text parts) must not be stringified."""

    parts = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
        {"type": "text", "text": "zoomed"},
    ]

    class _RichToolEnv(_ToolMathEnv):
        async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
            return parts

    env = _RichToolEnv(max_tool_calls=1)
    example = Example(
        id="rich-tool",
        payload={
            "prompt_messages": [{"role": "user", "content": "6*7?"}],
            "answer": "42",
        },
    )

    def respond(session_id: str, call_index: int, body: dict[str, Any]):
        if call_index == 0:
            return 200, completion_response(
                content=None,
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
            )
        # The tool message the model sees next turn must carry the parts.
        tool_turn = body["messages"][-1]
        assert tool_turn["role"] == "tool"
        assert tool_turn["content"] == parts
        return 200, completion_response(content="42")

    with LocalModelServer(respond, concurrent_calls=1) as server:
        outcomes = await env.run_group(_requests(server, example, ["solo"]))

    assert outcomes["solo"].termination_reason == "finished"
    assert outcomes["solo"].rewards == {"correctness": 1.0}
