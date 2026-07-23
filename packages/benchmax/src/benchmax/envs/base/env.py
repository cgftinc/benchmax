"""Default OpenAI-compatible environment loop for custom Benchmax environments."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI, BadRequestError, OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionToolMessageParam,
)

from benchmax.envs.base.dataset import JsonRow
from benchmax.envs.base.openai_types import Message, Messages, Tool
from benchmax.envs.dataset import Dataset
from benchmax.envs.environment import Environment
from benchmax.envs.shared_types import (
    DatasetSplit,
    Example,
    RewardMap,
    RolloutAttempt,
    RolloutFailure,
    RolloutRequest,
)
from benchmax.auth import ModelRequestContext

__all__ = ["BaseEnv", "BaseRollout"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseRollout(RolloutAttempt):
    """Completed BaseEnv chat loop available to individual and group scoring."""

    messages: Messages
    example_args: Mapping[str, Any]
    # Which run context requested the rollout; reward hooks may branch on it.
    split: DatasetSplit = "train"


class BaseEnv(Environment[JsonRow, BaseRollout], ABC):
    """Reusable chat-completion loop with optional function tools.

    Example payloads are JSON objects containing ``prompt_messages``. All other
    fields are opaque environment data available to reward hooks. Most envs
    implement ``create_dataset`` and ``compute_reward``; tools, group rewards,
    and rollout resources are optional hooks.
    """

    max_turns: int | None = None
    max_tool_calls: int | None = None

    def __init__(
        self,
        *,
        max_turns: int | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        super().__init__()
        if max_turns is not None:
            self.max_turns = max_turns
        if max_tool_calls is not None:
            self.max_tool_calls = max_tool_calls
        self._validate_limits()

    @abstractmethod
    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[JsonRow]:
        """Create a finite dataset whose payloads contain ``prompt_messages``."""

    async def list_tools(self) -> list[Tool]:
        """Return OpenAI-compatible tools. No tools is the default."""

        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        """Execute one model-requested tool."""

        raise ValueError(f"Unknown tool: {tool_name}")

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> RewardMap | None:
        """Optionally score one completed rollout.

        Raise :class:`RolloutFailure` for expected runtime failures to settle
        this rollout under that reason. Any other exception settles it as
        ``reward_error``; a hook defect never crashes the run.
        """

        return None

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> Mapping[str, RewardMap] | None:
        """Optionally contribute rewards that depend on the complete group."""

        return None

    @asynccontextmanager
    async def rollout_context(
        self,
        rollout_id: str,
        example: Example[JsonRow],
    ) -> AsyncGenerator[None]:
        """Optionally acquire and release per-rollout resources."""

        yield

    async def aclose(self) -> None:
        """Optionally close environment-owned shared resources."""

    async def run_rollout(
        self,
        request: RolloutRequest[JsonRow],
    ) -> BaseRollout:
        """Run and optionally score one common chat/tool attempt."""

        self._validate_limits()
        if self.max_turns is None:
            raise ValueError(
                f"{self.__class__.__name__} must configure max_turns before rollout"
            )

        messages, example_args = _prepare_example(request.example.payload)

        async with self.rollout_context(request.rollout_id, request.example):
            tools = await self.list_tools()
            async with _model_client(request) as client:
                tool_calls_used = 0

                for _ in range(self.max_turns):
                    try:
                        completion = await _create_chat_completion(
                            client=client,
                            request=request,
                            model=request.model,
                            messages=messages,
                            tools=tools,
                        )
                    except BadRequestError as error:
                        if error.code == "context_budget_exceeded":
                            termination_reason = "context_exceeded"
                        else:
                            termination_reason = "model_error"
                        logger.warning(
                            "base.rollout.model_failed rollout_id=%s "
                            "termination_reason=%s",
                            request.rollout_id,
                            termination_reason,
                            exc_info=True,
                        )
                        break
                    except OpenAIError:
                        termination_reason = "model_error"
                        logger.exception(
                            "base.rollout.model_failed rollout_id=%s "
                            "termination_reason=%s",
                            request.rollout_id,
                            termination_reason,
                        )
                        break
                    except Exception:
                        # Authentication is resolved inside this exact runtime
                        # boundary and custom providers need not raise an
                        # OpenAI exception. Keep response-processing and env
                        # programming errors outside this catch.
                        termination_reason = "model_error"
                        logger.exception(
                            "base.rollout.model_failed rollout_id=%s "
                            "termination_reason=%s",
                            request.rollout_id,
                            termination_reason,
                        )
                        break
                    if not completion.choices:
                        termination_reason = "model_error"
                        logger.error(
                            "base.rollout.model_failed rollout_id=%s "
                            "termination_reason=%s: response contained no choices",
                            request.rollout_id,
                            termination_reason,
                        )
                        break
                    assistant_turn = completion.choices[0]
                    tool_calls = _function_tool_calls(assistant_turn.message)
                    messages.append(
                        _to_assistant_message(assistant_turn.message, tool_calls)
                    )

                    if not tool_calls:
                        termination_reason = _termination_reason(
                            assistant_turn.finish_reason
                        )
                        break

                    if (
                        self.max_tool_calls is not None
                        and tool_calls_used + len(tool_calls) > self.max_tool_calls
                    ):
                        termination_reason = "tool_budget_exceeded"
                        break

                    tool_calls_used += len(tool_calls)
                    try:
                        tool_messages = await self._execute_tool_calls(
                            request.rollout_id,
                            tool_calls,
                            tools,
                        )
                    except RolloutFailure as failure:
                        termination_reason = failure.termination_reason
                        logger.error(
                            "base.rollout.tool_failed rollout_id=%s "
                            "termination_reason=%s: %s",
                            request.rollout_id,
                            termination_reason,
                            failure,
                            exc_info=(type(failure), failure, failure.__traceback__),
                        )
                        break
                    except Exception:
                        termination_reason = "tool_error"
                        logger.exception(
                            "base.rollout.tool_failed rollout_id=%s "
                            "termination_reason=%s",
                            request.rollout_id,
                            termination_reason,
                        )
                        break
                    messages.extend(tool_messages)
                else:
                    termination_reason = "max_turns_exceeded"

                rollout = BaseRollout(
                    rollout_id=request.rollout_id,
                    termination_reason=termination_reason,
                    messages=messages,
                    example_args=example_args,
                    split=request.split,
                )
                if termination_reason not in ("finished", "context_exceeded"):
                    return replace(
                        rollout,
                        rewards={key: 0.0 for key in self.reward_keys},
                    )
                try:
                    rewards = await self.compute_reward(rollout)
                except RolloutFailure as failure:
                    logger.error(
                        "base.rollout.reward_failed rollout_id=%s "
                        "termination_reason=%s: %s",
                        request.rollout_id,
                        failure.termination_reason,
                        failure,
                        exc_info=(type(failure), failure, failure.__traceback__),
                    )
                    return replace(
                        rollout,
                        termination_reason=failure.termination_reason,
                        rewards={key: 0.0 for key in self.reward_keys},
                    )
                except Exception as error:
                    # compute_reward is user code: a defect settles this one
                    # rollout with a labeled zero outcome instead of crashing
                    # the run; siblings and the group hook are unaffected.
                    logger.error(
                        "base.rollout.reward_failed rollout_id=%s "
                        "termination_reason=reward_error: %s",
                        request.rollout_id,
                        error,
                        exc_info=(type(error), error, error.__traceback__),
                    )
                    return replace(
                        rollout,
                        termination_reason="reward_error",
                        rewards={key: 0.0 for key in self.reward_keys},
                    )
                return replace(rollout, rewards=rewards)

    async def _execute_tool_calls(
        self,
        rollout_id: str,
        tool_calls: Sequence[ChatCompletionMessageFunctionToolCall],
        tools: Sequence[Tool],
    ) -> list[Message]:
        """Execute function calls and return their transcript messages."""

        advertised_tools = {
            name
            for tool in tools
            if isinstance(tool, Mapping)
            and isinstance((function := tool.get("function")), Mapping)
            and isinstance((name := function.get("name")), str)
        }
        tool_messages: list[Message] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name not in advertised_tools:
                tool_messages.append(
                    _tool_message(tool_call.id, f"Unknown tool: {tool_name}")
                )
                continue

            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_messages.append(
                    _tool_message(
                        tool_call.id,
                        f"Invalid JSON tool arguments: {exc.msg}",
                    )
                )
                continue
            if not isinstance(tool_args, dict):
                tool_messages.append(
                    _tool_message(
                        tool_call.id,
                        "Tool arguments must be a JSON object.",
                    )
                )
                continue

            result = await self.run_tool(rollout_id, tool_name, **tool_args)
            if _is_tool_content_parts(result):
                # Rich tool results (e.g. an image crop) pass through as
                # OpenAI content parts; VL chat templates render them with
                # real vision tokens inside the tool response.
                tool_messages.append(_tool_message(tool_call.id, list(result)))
            else:
                tool_messages.append(
                    _tool_message(tool_call.id, _serialize_tool_result(result))
                )

        return tool_messages

    def _validate_limits(self) -> None:
        """Reject invalid turn and tool-call limits."""

        if self.max_turns is not None and (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer")
        if self.max_tool_calls is not None and (
            isinstance(self.max_tool_calls, bool)
            or not isinstance(self.max_tool_calls, int)
            or self.max_tool_calls < 0
        ):
            raise ValueError("max_tool_calls must be a non-negative integer")


def _prepare_example(payload: JsonRow) -> tuple[Messages, Mapping[str, Any]]:
    """Split prompt messages from opaque reward arguments."""

    if not isinstance(payload, Mapping):
        raise TypeError("BaseEnv example payload must be a JSON object")
    messages = _coerce_messages(payload.get("prompt_messages"))
    example_args = {
        key: value for key, value in payload.items() if key != "prompt_messages"
    }
    return messages, example_args


def _coerce_messages(value: object) -> Messages:
    """Validate and copy the initial OpenAI-compatible messages."""

    if not isinstance(value, list) or not value:
        raise ValueError(
            "BaseEnv example payload requires a non-empty prompt_messages list"
        )
    messages: Messages = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError(f"prompt_messages[{index}] must be a JSON object")
        if not isinstance(message.get("role"), str):
            raise TypeError(f"prompt_messages[{index}].role must be a string")
        messages.append(cast(Message, dict(message)))
    return messages


@asynccontextmanager
async def _model_client(
    request: RolloutRequest[JsonRow],
) -> AsyncGenerator[AsyncOpenAI]:
    """Create and close the rollout's OpenAI-compatible client."""

    client = AsyncOpenAI(
        # The real Authorization header is resolved immediately before every
        # request. The SDK requires a constructor value even when callers use
        # per-request headers.
        api_key="benchmax-runtime-auth",
        base_url=request.base_url,
        max_retries=0,
    )
    try:
        yield client
    finally:
        await client.close()


async def _create_chat_completion(
    *,
    client: AsyncOpenAI,
    request: RolloutRequest[JsonRow],
    model: str,
    messages: Messages,
    tools: Sequence[Tool],
) -> ChatCompletion:
    """Request one assistant turn, including tools when available."""

    auth_headers = await request.model_auth.headers_for_request(
        ModelRequestContext(
            base_url=request.base_url,
            model=model,
            rollout_id=request.rollout_id,
        )
    )

    if tools:
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            extra_headers=dict(auth_headers),
        )
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers=dict(auth_headers),
    )


def _function_tool_calls(
    message: ChatCompletionMessage,
) -> list[ChatCompletionMessageFunctionToolCall]:
    """Return function calls and reject unsupported tool-call types."""

    tool_calls: list[ChatCompletionMessageFunctionToolCall] = []
    for tool_call in message.tool_calls or []:
        if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
            raise TypeError("BaseEnv supports function tool calls only")
        tool_calls.append(tool_call)
    return tool_calls


def _to_assistant_message(
    message: ChatCompletionMessage,
    tool_calls: Sequence[ChatCompletionMessageFunctionToolCall],
) -> Message:
    """Convert a model response into a reusable transcript message."""

    assistant_message: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": message.content,
    }
    if tool_calls:
        call_params: list[ChatCompletionMessageFunctionToolCallParam] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]
        assistant_message["tool_calls"] = call_params
    return assistant_message


def _tool_message(
    tool_call_id: str,
    content: str | list[Any],
) -> Message:
    """Build the transcript message for one tool result."""

    message: ChatCompletionToolMessageParam = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    return message


def _termination_reason(finish_reason: str | None) -> str:
    """Map an OpenAI finish reason to Benchmax tracking metadata."""

    if finish_reason == "length":
        return "output_exceeded"
    if finish_reason == "stop":
        return "finished"
    if finish_reason is None:
        return "model_error"
    return str(finish_reason)


_TOOL_CONTENT_PART_TYPES = frozenset({"text", "image_url"})


def _is_tool_content_parts(result: object) -> bool:
    """Whether a tool result is an explicit OpenAI content-part list."""

    return (
        isinstance(result, (list, tuple))
        and len(result) > 0
        and all(
            isinstance(part, Mapping)
            and part.get("type") in _TOOL_CONTENT_PART_TYPES
            for part in result
        )
    )


def _serialize_tool_result(result: object) -> str:
    """Serialize a tool result for the chat transcript."""

    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (Mapping, list, tuple, int, float, bool)):
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return str(result)
