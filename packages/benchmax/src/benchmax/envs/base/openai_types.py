"""OpenAI-compatible runtime shapes used by :class:`BaseEnv`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

    Message: TypeAlias = ChatCompletionMessageParam
    Tool: TypeAlias = ChatCompletionToolParam
else:
    Message: TypeAlias = dict[str, Any]
    Tool: TypeAlias = dict[str, Any]

Messages: TypeAlias = list[Message]

__all__ = ["Message", "Messages", "Tool"]
