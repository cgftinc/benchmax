"""BenchMax environment for supervised fine-tuning demonstrations."""

from __future__ import annotations

import json
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Messages, ToolDefinition


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _parse_optional_json_field(value: Any) -> Any:
    if value is None:
        return None
    return _parse_json_field(value)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"value": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": arguments}


def _normalize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if "function" in call:
            fn = call.get("function") or {}
            normalized.append(
                {
                    "id": call.get("id", ""),
                    "type": call.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": _normalize_tool_call_arguments(
                            fn.get("arguments")
                        ),
                    },
                }
            )
            continue
        normalized.append(
            {
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": _normalize_tool_call_arguments(call.get("arguments")),
                },
            }
        )
    return normalized


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in message.items():
        if value is None:
            continue
        if key == "tool_calls":
            if isinstance(value, list) and value:
                out[key] = _normalize_tool_calls(value)
            continue
        out[key] = value
    return out


def _as_message_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    value = _parse_json_field(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(
            f"{field_name} must be a message dict or list, got {type(value)}"
        )
    return [_clean_message(message) for message in value]


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "prompt_messages" in row:
        return _as_message_list(row["prompt_messages"], field_name="prompt_messages")
    if "messages" in row:
        return _as_message_list(row["messages"], field_name="messages")
    if "prompt" in row:
        return _as_message_list(
            [{"role": "user", "content": row["prompt"]}], field_name="prompt"
        )
    raise KeyError("SFT rows must include prompt_messages, messages, or prompt")


def _completion_source(row: dict[str, Any]) -> Any:
    if "completion_messages" in row:
        return row["completion_messages"]
    if "ground_truth" in row:
        return row["ground_truth"]
    task = _parse_optional_json_field(row.get("task"))
    if isinstance(task, dict):
        if "completion_messages" in task:
            return task["completion_messages"]
        if "completion" in task:
            return task["completion"]
    return None


def _completion_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    completion = _completion_source(row)
    if completion is None:
        raise ValueError(
            "SFT rows must include completion_messages, ground_truth, or task.completion_messages"
        )
    return _as_message_list(completion, field_name="completion_messages")


def _task(row: dict[str, Any]) -> dict[str, Any]:
    task: dict[str, Any] = {}
    existing_task = _parse_optional_json_field(row.get("task"))
    if isinstance(existing_task, dict):
        task.update(existing_task)

    for key, value in row.items():
        if key in {
            "prompt_messages",
            "messages",
            "prompt",
            "completion_messages",
            "ground_truth",
            "task",
        }:
            continue
        if key == "init_rollout_args":
            continue
        task.setdefault(key, value)

    task["completion_messages"] = _completion_messages(row)
    if "ground_truth" in row:
        task.setdefault("ground_truth", row["ground_truth"])
    return task


class SftDemonstrationEnv(BaseEnv):
    """BenchMax env wrapper for rows that already contain assistant targets.

    This env exists to make SFT use the same env-bundle packaging contract as
    RL. It does not provide tools or rewards because pure SFT trains from the
    demonstration completion stored in ``Example.task["completion_messages"]``.
    """

    @classmethod
    def dataset_preprocess(cls, row: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return make_example(
            prompt_messages=_prompt_messages(row),
            task=_task(row),
            init_rollout_args=_parse_optional_json_field(row.get("init_rollout_args")),
            system_prompt=cls.system_prompt or None,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        _ = rollout_id, tool_args
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tools; got {tool_name!r}"
        )

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        _ = rollout_id, messages, task, kwargs
        raise NotImplementedError(
            f"{self.__class__.__name__} is for SFT demonstrations, not reward scoring"
        )
