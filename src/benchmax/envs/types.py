from dataclasses import dataclass
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

Messages = List[Dict[str, Any]]


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


@dataclass
class ToolDefinition:
    """Definition of a tool's interface"""

    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
