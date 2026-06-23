import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, NotRequired, Optional, TypedDict


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class _ToolCallFunction(TypedDict):
    """The ``function`` payload inside a tool-call dict (OpenAI nested format)."""

    name: str
    arguments: str


class _ToolCallDict(TypedDict, total=False):
    """Serialized tool-call dict (OpenAI nested format).

    This is the canonical dict shape produced by :meth:`ToolCall.to_dict`
    and stored in :class:`ChatMessage`, matching what the trainer emits::

        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}}

    At deserialization boundaries (trace import, dataset loading) the flat
    format (``{name, arguments, id}``) is also accepted and normalized by
    :func:`~benchmax.traces.adapter.normalize_message`.
    """

    id: str
    type: str
    function: _ToolCallFunction


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation within a message.

    ``arguments`` is stored as a raw JSON string (matching OpenAI format)
    to guarantee JSON-serializability and preserve the original format.

    This is the canonical in-memory representation.  Use :meth:`to_dict`
    to serialize to a :class:`_ToolCallDict` for Arrow / JSONL storage.
    """

    name: str
    arguments: str = "{}"
    id: str | None = None

    def to_dict(self) -> _ToolCallDict:
        """Serialize to OpenAI nested-format :class:`_ToolCallDict`."""
        return _ToolCallDict(
            id=self.id or "",
            type="function",
            function=_ToolCallFunction(name=self.name, arguments=self.arguments),
        )

    def arguments_dict(self) -> dict[str, Any]:
        """Parse *arguments* back to a dict.  Returns ``{}`` on failure."""
        try:
            val = json.loads(self.arguments)
            return val if isinstance(val, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


class _ChatMessageRequired(TypedDict):
    role: str  # "system" | "user" | "assistant" | "tool"


class ChatMessage(_ChatMessageRequired, total=False):
    """A single message in a chat-message list.

    ``role`` is always required.  All other fields are optional because
    not every role uses every field (e.g. system messages have no
    ``tool_calls``).
    """

    content: str
    tool_calls: List[_ToolCallDict]
    tool_call_id: str
    name: str


Messages = List[ChatMessage]


# ---------------------------------------------------------------------------
# Dataset example
# ---------------------------------------------------------------------------


class Example(TypedDict):
    """A dataset example after preprocessing.

    ``id`` is a SHA-256 hash over ``(prompt_messages, task)`` computed by
    :func:`benchmax.envs.example_id.canonical_example_id`. Equal seeds + tasks
    across calls (and across Python/TypeScript) produce equal ids.

    ``prompt_messages`` is the full prompt as a chat-message list. Includes the
    system message if the env has one (rendered with tool definitions when
    tools are present).

    ``task`` carries per-example reward-side data: ground truth, scoring config,
    or anything else the env's ``compute_reward`` needs to grade the rollout.
    Must be JSON-serializable. May be ``None`` if the env grades without
    per-example data.

    ``init_rollout_args`` carries trainer-runtime context passed to
    ``init_rollout`` (e.g. workspace_path). Not part of the example identity.
    """

    id: str
    prompt_messages: Messages
    task: NotRequired[Optional[Dict[str, Any]]]
    init_rollout_args: NotRequired[Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class PolicyConfig:
    """Policy endpoint/config for envs that own the full rollout loop."""

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    rollout_func: Any | None = None
    generation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Trainer-ready artifact produced by one completed rollout attempt."""

    rollout_id: str
    example_id: str
    prompt_messages: Messages
    messages: Messages
    task: Optional[Dict[str, Any]]
    prompt_ids: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    logprobs: List[float]
    rewards: Dict[str, float]
    prompt_mask: List[int] | None = None
    truncated: bool = False
    workspace_path: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sample_dict(self) -> Dict[str, Any]:
        """Return the sample dict consumed by the current trainer."""
        sample = {
            "rollout_id": self.rollout_id,
            "example_id": self.example_id,
            "prompt_messages": self.prompt_messages,
            "messages": self.messages,
            "task": self.task,
            "prompt_ids": self.prompt_ids,
            "prompt_mask": self.prompt_mask or [1] * len(self.prompt_ids),
            "completion_ids": self.completion_ids,
            "completion_mask": self.completion_mask,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "truncated": self.truncated,
            **self.metadata,
        }
        if self.workspace_path is not None:
            sample["workspace_path"] = self.workspace_path
        return sample


# ---------------------------------------------------------------------------
# Tool definitions (env interface)
# ---------------------------------------------------------------------------


@dataclass
class ToolDefinition:
    """Definition of a tool's interface"""

    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
