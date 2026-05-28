"""TraceAdapter protocol and normalized trace data models."""

from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass
class TraceCredentials:
    """Base credentials with redacted repr to prevent key leakage in logs."""

    api_key: str

    def __repr__(self) -> str:
        redacted = self.api_key[:2] + "****" if len(self.api_key) > 2 else "****"
        return f"{self.__class__.__name__}(api_key={redacted})"

    def to_headers(self) -> dict[str, str]:
        """Provider-specific auth headers. Override in subclasses."""
        return {"Authorization": f"Bearer {self.api_key}"}


@dataclass(repr=False)
class BraintrustCredentials(TraceCredentials):
    """Braintrust API credentials."""

    pass


@dataclass(repr=False)
class LangfuseCredentials(TraceCredentials):
    """Langfuse credentials with SSRF-validated host.

    Langfuse uses Basic auth (base64(public_key:secret_key)), not Bearer.
    """

    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"

    def __post_init__(self) -> None:
        validate_provider_url(self.host)

    def to_headers(self) -> dict[str, str]:
        """Langfuse uses Basic auth: base64(public_key:secret_key)."""
        import base64

        token = base64.b64encode(f"{self.api_key}:{self.secret_key}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def __repr__(self) -> str:
        rpk = self.api_key[:2] + "****" if len(self.api_key) > 2 else "****"
        rsk = self.secret_key[:2] + "****" if len(self.secret_key) > 2 else "****"
        return f"LangfuseCredentials(api_key={rpk}, secret_key={rsk})"


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal"}


def validate_provider_url(url: str) -> None:
    """Validate that *url* is HTTPS and does not resolve to a private IP.

    Raises ``ValueError`` on violation.  Applied to self-hosted provider URLs
    (Langfuse, OTel).  Not needed for fixed-URL providers (Braintrust).
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Provider URL must use HTTPS, got: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid provider URL: {url}")
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked metadata endpoint: {hostname}")
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(hostname))
    except (socket.gaierror, ValueError):
        # DNS resolution failed at validation time.  The host might be
        # temporarily unreachable or not yet provisioned.  We allow this
        # through — the actual HTTP request will fail with a clear error.
        # Note: this creates a small TOCTOU window where DNS rebinding
        # could bypass validation, but practical risk is low in Modal.
        return
    if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local:
        raise ValueError(f"Provider URL resolves to private/reserved IP: {addr}")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation within an assistant message.

    ``arguments`` is stored as a raw JSON string (matching OpenAI format)
    to guarantee JSON-serializability and preserve the original format.
    """

    name: str
    arguments: str = "{}"
    id: str | None = None

    def arguments_dict(self) -> dict[str, Any]:
        """Parse *arguments* back to a dict.  Returns ``{}`` on failure."""
        try:
            val = json.loads(self.arguments)
            return val if isinstance(val, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass(frozen=True)
class TraceMessage:
    """Single message in a normalised conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role="tool"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict with structured tool_calls.

        Used by the web-app wizard preview, trace rendering, and
        intermediate pipeline storage. All fields are always present
        with type-safe defaults (empty list / empty string, never None).
        """
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        d["tool_calls"] = (
            [{"name": tc.name, "arguments": tc.arguments, "id": tc.id or ""} for tc in self.tool_calls]
            if self.tool_calls
            else []
        )
        d["tool_call_id"] = self.tool_call_id or ""
        d["name"] = self.name or ""
        return d



@dataclass(frozen=True)
class NormalizedTrace:
    """Provider-agnostic trace with extracted conversation."""

    id: str
    messages: list[TraceMessage]
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    errors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON transport (e.g. Modal ↔ wizard)."""
        d: dict[str, Any] = {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "scores": self.scores,
            "metadata": self.metadata,
        }
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        if self.errors:
            d["errors"] = self.errors
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedTrace:
        """Reconstruct from a dict produced by ``to_dict``."""
        messages = [_message_from_dict(m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            messages=messages,
            scores=data.get("scores", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
            errors=data.get("errors"),
        )


@dataclass
class TraceProject:
    """A project/workspace from the provider."""

    id: str
    name: str


@dataclass
class DetectedTool:
    """A tool auto-detected from trace messages."""

    name: str
    call_count: int
    sample_args: list[dict[str, Any]] = field(default_factory=list)
    param_keys: set[str] = field(default_factory=set)


@dataclass
class DetectedTools:
    """Tools auto-detected from trace messages."""

    tools: list[DetectedTool] = field(default_factory=list)


@dataclass
class DetectedSystemPrompt:
    """System prompt detected across traces."""

    prompt: str
    count: int
    total_traces: int
    variants: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TraceAdapter(Protocol):
    """Protocol for trace provider backends.

    Credentials are passed to ``__init__`` (provider-specific), not to
    individual methods.  This matches the ``ChunkSource`` pattern used
    by corpus backends.

    Implementations:
        BraintrustTraceAdapter
        LangfuseTraceAdapter
        OTelTraceAdapter
    """

    def connect(self) -> dict[str, Any]:
        """Validate credentials and return connection info."""
        ...

    def list_projects(self) -> list[TraceProject]:
        """List available projects/workspaces."""
        ...

    def count_traces(self, project_id: str) -> int:
        """Return the total number of traces for a project."""
        ...

    def fetch_traces(
        self,
        project_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[NormalizedTrace], str | None]:
        """Fetch and normalise traces.  Returns ``(traces, next_cursor)``."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message_from_dict(d: dict[str, Any]) -> TraceMessage:
    tool_calls = None
    if "tool_calls" in d and d["tool_calls"]:
        tool_calls = [
            ToolCall(
                name=tc.get("name", ""),
                arguments=tc.get("arguments", "{}"),
                id=tc.get("id"),
            )
            for tc in d["tool_calls"]
        ]
    return TraceMessage(
        role=d["role"],
        content=d.get("content", ""),
        tool_calls=tool_calls,
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
    )


# ---------------------------------------------------------------------------
# Message format normalizers
# ---------------------------------------------------------------------------


def _ensure_json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _is_structured_content(msg: dict[str, Any]) -> bool:
    """True if ``content`` is a list of typed blocks (openclaw / Anthropic style)."""
    content = msg.get("content")
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and "type" in content[0]
    )


def normalize_structured_message(msg: dict[str, Any]) -> TraceMessage:
    """Normalize a message with structured content blocks to ``TraceMessage``.

    Handles the format used by openclaw and Anthropic-style agents where
    ``content`` is a list of typed blocks::

        {type: "text", text: "..."}
        {type: "toolCall", name: "...", arguments: "...", id: "..."}
        {type: "toolResult", ...}

    And ``role`` may be ``"toolResult"`` (normalized to ``"tool"``).

    Any provider adapter can call this when it encounters structured-content
    messages — it's not tied to a specific provider.
    """
    role = msg.get("role", "")
    content = msg.get("content", "")

    if role == "toolResult":
        role = "tool"

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "toolCall":
                tool_calls.append(ToolCall(
                    name=block.get("name", ""),
                    arguments=_ensure_json_string(block.get("arguments", "{}")),
                    id=block.get("id"),
                ))
        content = "\n".join(text_parts) if text_parts else ""
    else:
        content = str(content) if content else ""

    return TraceMessage(
        role=role,
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        tool_call_id=msg.get("toolCallId") or msg.get("tool_call_id"),
        name=msg.get("toolName") or msg.get("name"),
    )


def normalize_message(msg: dict[str, Any]) -> TraceMessage:
    """Auto-detect message format and normalize to ``TraceMessage``.

    Handles both structured-content (openclaw / Anthropic) and flat
    (OpenAI) message formats. Provider adapters can call this instead
    of implementing format detection themselves.
    """
    if _is_structured_content(msg):
        return normalize_structured_message(msg)
    return _message_from_dict(msg)
