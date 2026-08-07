"""Row schema validation for the `benchmax-sft-v1` format.

One row is a closed JSON object: required ``messages``, optional ``tools``,
optional ``metadata``. Within-row validation order is fixed so equivalent rows
always yield an identical issue sequence: row shape, tool definitions, the
message sequence (with tool-call/result linking interleaved in message order),
the tools-required-for-calls check, metadata, then the trainable-turn check.

The validator never raises on malformed input; every problem is reported
through the ``emit`` callback as ``(location, message)``.
"""

from __future__ import annotations

from collections.abc import Callable

from benchmax.sft.strict_json import (
    StrictJsonError,
    canonical_json_bytes,
    check_json_tree,
    parse_strict_json,
)

__all__ = [
    "MAX_MESSAGES",
    "MAX_METADATA_BYTES",
    "MAX_ROW_BYTES",
    "MAX_TOOLS",
    "RESERVED_METADATA_PREFIX",
    "validate_row",
]

MAX_MESSAGES = 1024
MAX_TOOLS = 128
MAX_ROW_BYTES = 1_048_576
MAX_METADATA_BYTES = 65_536
RESERVED_METADATA_PREFIX = "_castform_"

_ROW_KEYS = frozenset({"messages", "tools", "metadata"})
_ROLES = ("assistant", "system", "tool", "user")

Emit = Callable[[str, str], None]


def _unknown_keys(mapping: dict[str, object], allowed: frozenset[str]) -> list[str]:
    return sorted(key for key in mapping if key not in allowed)


def _is_nonempty_str(value: object) -> bool:
    return type(value) is str and value != ""


def validate_row(row: dict[str, object], emit: Emit) -> None:
    """Validate one already tree-checked row object, reporting via ``emit``."""

    unknown = _unknown_keys(row, _ROW_KEYS)
    if unknown:
        emit("$", f"unknown top-level key(s): {', '.join(unknown)}")

    defined_tool_names = _validate_tools(row, emit)
    any_tool_calls, trainable_turns = _validate_messages(row, defined_tool_names, emit)

    if any_tool_calls:
        tools = row.get("tools")
        if type(tools) is not list or not tools:
            emit("$.tools", "tools must be a non-empty list when any message makes a tool call")

    _validate_metadata(row, emit)

    if "messages" in row and type(row["messages"]) is list and row["messages"]:
        if trainable_turns == 0:
            emit(
                "$.messages",
                "at least one assistant message with weight 1 (or omitted weight) is required",
            )


def _validate_tools(row: dict[str, object], emit: Emit) -> set[str] | None:
    """Validate ``tools`` definitions; return defined names, or None if unusable."""

    if "tools" not in row:
        return None
    tools = row["tools"]
    if type(tools) is not list:
        emit("$.tools", "tools must be a list")
        return None
    if not tools:
        # No definitions to resolve against; the tools-required-for-calls
        # check reports the real problem when a call exists.
        return None
    if len(tools) > MAX_TOOLS:
        emit("$.tools", f"tools has {len(tools)} entries; the maximum is {MAX_TOOLS}")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        location = f"$.tools[{index}]"
        if type(tool) is not dict:
            emit(location, "tool definition must be a JSON object")
            continue
        unknown = _unknown_keys(tool, frozenset({"type", "function"}))
        if unknown:
            emit(location, f"unknown key(s): {', '.join(unknown)}")
        if tool.get("type") != "function":
            emit(f"{location}.type", 'tool definition type must be the literal "function"')
        if "function" not in tool:
            emit(location, "tool definition requires a function object")
            continue
        function = tool["function"]
        if type(function) is not dict:
            emit(f"{location}.function", "function must be a JSON object")
            continue
        unknown = _unknown_keys(function, frozenset({"name", "parameters", "description"}))
        if unknown:
            emit(f"{location}.function", f"unknown key(s): {', '.join(unknown)}")
        name = function.get("name")
        if not _is_nonempty_str(name):
            emit(f"{location}.function.name", "name must be a non-empty string")
        elif name in names:
            emit(f"{location}.function.name", f"duplicate tool definition for function {name!r}")
        else:
            names.add(name)
        if "parameters" not in function:
            emit(f"{location}.function", "parameters is required and must be a JSON object")
        elif type(function["parameters"]) is not dict:
            emit(f"{location}.function.parameters", "parameters must be a JSON object")
        else:
            declared = function["parameters"].get("type")
            if "type" in function["parameters"] and declared != "object":
                emit(
                    f"{location}.function.parameters.type",
                    'parameters type, when declared, must be "object"',
                )
        if "description" in function and type(function["description"]) is not str:
            emit(f"{location}.function.description", "description must be a string")
    return names


def _validate_messages(
    row: dict[str, object],
    defined_tool_names: set[str] | None,
    emit: Emit,
) -> tuple[bool, int]:
    """Validate the message sequence; return (any_tool_calls, trainable_turns)."""

    if "messages" not in row:
        emit("$", "messages is required")
        return False, 0
    messages = row["messages"]
    if type(messages) is not list:
        emit("$.messages", "messages must be a list")
        return False, 0
    if not messages:
        emit("$.messages", "messages must not be empty")
        return False, 0
    if len(messages) > MAX_MESSAGES:
        emit("$.messages", f"messages has {len(messages)} entries; the maximum is {MAX_MESSAGES}")

    linker = _ToolCallLinker(emit)
    any_tool_calls = False
    trainable_turns = 0
    for index, message in enumerate(messages):
        location = f"$.messages[{index}]"
        if type(message) is not dict:
            linker.close_window()
            emit(location, "message must be a JSON object")
            continue
        role = message.get("role")
        if role not in _ROLES:
            linker.close_window()
            if "role" not in message:
                emit(location, "role is required")
            else:
                emit(
                    f"{location}.role",
                    "role must be one of " + ", ".join(f'"{name}"' for name in _ROLES),
                )
            continue
        if role == "assistant":
            linker.close_window()
            called, trainable = _validate_assistant(
                message, location, defined_tool_names, linker, emit
            )
            any_tool_calls = any_tool_calls or called
            trainable_turns += trainable
        elif role == "tool":
            _validate_tool_result(message, location, linker, emit)
        else:
            linker.close_window()
            _validate_system_or_user(message, location, emit)
    linker.close_window()
    return any_tool_calls, trainable_turns


def _validate_system_or_user(message: dict[str, object], location: str, emit: Emit) -> None:
    unknown = _unknown_keys(message, frozenset({"role", "content"}))
    if unknown:
        detail = f"unknown key(s): {', '.join(unknown)}"
        if "weight" in unknown:
            detail += " (weight is only supported on assistant messages)"
        emit(location, detail)
    if "content" not in message or type(message["content"]) is not str:
        emit(f"{location}.content", "content must be a string")
    elif message["content"] == "":
        emit(f"{location}.content", "content must not be empty")


def _validate_assistant(
    message: dict[str, object],
    location: str,
    defined_tool_names: set[str] | None,
    linker: _ToolCallLinker,
    emit: Emit,
) -> tuple[bool, int]:
    """Validate one assistant message; return (makes_tool_calls, trainable 0/1)."""

    unknown = _unknown_keys(message, frozenset({"role", "content", "tool_calls", "weight"}))
    if unknown:
        emit(location, f"unknown key(s): {', '.join(unknown)}")

    content = message.get("content")
    if "content" in message and content is not None and type(content) is not str:
        emit(f"{location}.content", "content must be a string or null")
        content = None

    weight_valid = True
    if "weight" in message:
        weight = message["weight"]
        if type(weight) is not int or weight not in (0, 1):
            emit(f"{location}.weight", "weight must be the integer 0 or 1")
            weight_valid = False

    call_ids: list[str] = []
    makes_calls = False
    if "tool_calls" in message:
        tool_calls = message["tool_calls"]
        if type(tool_calls) is not list or not tool_calls:
            emit(f"{location}.tool_calls", "tool_calls must be a non-empty list when present")
        else:
            makes_calls = True
            for index, call in enumerate(tool_calls):
                call_id = _validate_tool_call(
                    call, f"{location}.tool_calls[{index}]", defined_tool_names, linker, emit
                )
                if call_id is not None:
                    call_ids.append(call_id)

    if not _is_nonempty_str(content) and not makes_calls:
        emit(location, "assistant message needs non-empty content or at least one tool call")

    linker.open_window(location, call_ids)
    trainable = (
        1
        if weight_valid
        and message.get("weight", 1) != 0
        and (_is_nonempty_str(content) or makes_calls)
        else 0
    )
    return makes_calls, trainable


def _validate_tool_call(
    call: object,
    location: str,
    defined_tool_names: set[str] | None,
    linker: _ToolCallLinker,
    emit: Emit,
) -> str | None:
    """Validate one assistant tool call; return its id when usable for linking."""

    if type(call) is not dict:
        emit(location, "tool call must be a JSON object")
        return None
    unknown = _unknown_keys(call, frozenset({"id", "type", "function"}))
    if unknown:
        emit(location, f"unknown key(s): {', '.join(unknown)}")
    if call.get("type") != "function":
        emit(f"{location}.type", 'tool call type must be the literal "function"')

    call_id = call.get("id")
    usable_id: str | None = None
    if not _is_nonempty_str(call_id):
        emit(f"{location}.id", "id must be a non-empty string")
    elif call_id in linker.declared_ids:
        emit(f"{location}.id", f"duplicate tool call id {call_id!r}")
    else:
        linker.declared_ids.add(call_id)
        usable_id = call_id

    if "function" not in call:
        emit(location, "tool call requires a function object")
        return usable_id
    function = call["function"]
    if type(function) is not dict:
        emit(f"{location}.function", "function must be a JSON object")
        return usable_id
    unknown = _unknown_keys(function, frozenset({"name", "arguments"}))
    if unknown:
        emit(f"{location}.function", f"unknown key(s): {', '.join(unknown)}")
    name = function.get("name")
    if not _is_nonempty_str(name):
        emit(f"{location}.function.name", "name must be a non-empty string")
    elif defined_tool_names is not None and name not in defined_tool_names:
        emit(f"{location}.function.name", f"tool call references undefined function {name!r}")

    arguments_location = f"{location}.function.arguments"
    arguments = function.get("arguments")
    if type(arguments) is not str:
        emit(arguments_location, "arguments must be a JSON-encoded string")
        return usable_id
    try:
        decoded = parse_strict_json(arguments)
    except StrictJsonError as error:
        emit(arguments_location, f"arguments must decode as JSON: {error}")
        return usable_id
    if type(decoded) is not dict:
        emit(arguments_location, "arguments must decode as a JSON object")
        return usable_id
    check_json_tree(decoded, arguments_location, emit)
    return usable_id


def _validate_tool_result(
    message: dict[str, object],
    location: str,
    linker: _ToolCallLinker,
    emit: Emit,
) -> None:
    unknown = _unknown_keys(message, frozenset({"role", "content", "tool_call_id"}))
    if unknown:
        emit(location, f"unknown key(s): {', '.join(unknown)}")
    if "content" not in message or type(message["content"]) is not str:
        emit(f"{location}.content", "content must be a string")
    tool_call_id = message.get("tool_call_id")
    if not _is_nonempty_str(tool_call_id):
        emit(f"{location}.tool_call_id", "tool_call_id must be a non-empty string")
        return
    linker.resolve(location, tool_call_id)


class _ToolCallLinker:
    """Track the open tool-call window while walking one row's messages.

    An assistant message with tool calls opens a window listing its call ids in
    declaration order. Only ``tool`` results may follow while the window is
    open; each must match the next expected id. Any other message (or the end
    of the row) closes the window, reporting ids still awaiting results.
    """

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self.declared_ids: set[str] = set()
        self._resolved_ids: set[str] = set()
        self._window: list[str] = []
        self._window_location = ""

    def open_window(self, location: str, call_ids: list[str]) -> None:
        self.close_window()
        self._window = list(call_ids)
        self._window_location = location

    def close_window(self) -> None:
        if self._window:
            missing = ", ".join(repr(call_id) for call_id in self._window)
            self._emit(
                f"{self._window_location}.tool_calls",
                f"tool call id(s) {missing} have no tool result before the next non-tool message",
            )
        self._window = []

    def resolve(self, location: str, tool_call_id: str) -> None:
        target = f"{location}.tool_call_id"
        if self._window and tool_call_id == self._window[0]:
            self._window.pop(0)
            self._resolved_ids.add(tool_call_id)
        elif tool_call_id in self._window:
            self._emit(target, f"tool result for id {tool_call_id!r} is out of declaration order")
            self._window.remove(tool_call_id)
            self._resolved_ids.add(tool_call_id)
        elif tool_call_id in self._resolved_ids:
            self._emit(target, f"duplicate tool result for id {tool_call_id!r}")
        elif tool_call_id in self.declared_ids:
            self._emit(
                target,
                f"tool result for id {tool_call_id!r} appears after its tool-call window closed",
            )
        else:
            self._emit(target, f"tool result references unknown tool_call_id {tool_call_id!r}")


def _validate_metadata(row: dict[str, object], emit: Emit) -> None:
    if "metadata" not in row:
        return
    metadata = row["metadata"]
    if type(metadata) is not dict:
        emit("$.metadata", "metadata must be a JSON object")
        return
    for key in metadata:
        if type(key) is str and key.startswith(RESERVED_METADATA_PREFIX):
            emit(
                "$.metadata",
                f"metadata key {key!r} uses the reserved {RESERVED_METADATA_PREFIX!r} prefix",
            )
        # The training runtime stores the row's tool definitions under
        # metadata["tools"]; a user value there would bypass preflight and
        # reach the chat template unvalidated.
        if key == "tools":
            emit(
                "$.metadata",
                'metadata key "tools" is reserved for the training runtime; '
                "declare tools at the top level of the row",
            )
    try:
        size = len(canonical_json_bytes(metadata))
    except (TypeError, ValueError):
        return  # Unencodable subtrees are already reported by the tree check.
    if size > MAX_METADATA_BYTES:
        emit(
            "$.metadata",
            f"metadata canonical size is {size} bytes; the maximum is {MAX_METADATA_BYTES}",
        )
