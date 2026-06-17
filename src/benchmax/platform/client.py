"""Trainer API clients for storage uploads and job management."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import textwrap
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from benchmax import config

from .credentials import TokenProvider, resolve_token_provider
from .exceptions import (
    AuthenticationError,
    JobLaunchError,
    RolloutError,
    RolloutNotFound,
    RolloutServerError,
    RolloutStreamError,
    TrainerError,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types import ModuleType

    from benchmax.envs.base_env import BaseEnv


@dataclass(frozen=True)
class LaunchArgSpec:
    """One tunable arg exposed by the platform's GET /train/launch-args endpoint.

    Mirrors the API response shape — see core/platform-service/src/lib/trainer-args.ts.
    """

    name: str
    label: str
    type: str  # "string" | "number" | "integer" | "boolean"
    required: bool
    description: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    warn_above: float | None = None
    # tuple (not list) so the frozen dataclass actually stays immutable —
    # list would still be mutable in place.
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExampleValidation:
    """Per-example outcome from RolloutClient.validate_examples.

    ``rewards`` carries the rollout's final reward components — the per-rollout
    final rewards for a per-example entry, and the **mean** of the group
    rollouts' final rewards for the group entry (index ``-1``). None when the
    rollout failed or the server reported no reward values. Surfacing these is
    what lets ``castform validate`` show numbers, not just pass/fail.
    """

    index: int
    ok: bool
    error: str | None = None
    rewards: dict[str, float] | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate result from RolloutClient.validate_examples.

    Bool-castable: ``if not validate_examples(...)`` keeps working for the
    common "did everything pass" pattern. The ``examples`` field exposes
    per-example detail for richer reporting.
    """

    examples: list[ExampleValidation]
    # Outcome of the compute_group_reward contract check, run on the real
    # smoke-rollout transcripts. None when the env has no group reward, the
    # env class wasn't supplied, or the check was skipped (deps not installed
    # locally — it runs on the trainer instead). Its index is -1.
    group_reward: ExampleValidation | None = None

    @property
    def ok(self) -> bool:
        rollouts_ok = all(ex.ok for ex in self.examples)
        group_ok = self.group_reward is None or self.group_reward.ok
        return rollouts_ok and group_ok

    def __bool__(self) -> bool:
        return self.ok


# MIME type mappings for common file extensions
_MIME_TYPES = {
    ".jsonl": "application/jsonl",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".pkl": "application/octet-stream",
    ".pickle": "application/octet-stream",
}


def _get_mime_type(path: Path) -> str:
    """Get MIME type from file extension."""
    suffix = path.suffix.lower()
    mime_type = _MIME_TYPES.get(suffix)
    if mime_type is None:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: {', '.join(_MIME_TYPES.keys())}"
        )
    return mime_type


def _file_hash(content: bytes, length: int = 8) -> str:
    """Compute a short hash of file content."""
    return hashlib.sha256(content).hexdigest()[:length]


def _mean_rewards(reward_dicts: list[Any]) -> dict[str, float] | None:
    """Mean of each reward component across rollouts. None if no numeric values.

    Bools are excluded (``bool`` is an ``int`` subclass but not a reward).
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for rewards in reward_dicts:
        if not isinstance(rewards, dict):
            continue
        for key, value in rewards.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return {key: sums[key] / counts[key] for key in sums}


class _BearerAuth(httpx.Auth):
    """Resolve the platform bearer per request via ``token_provider``.

    Built once but called on every request, so the auth header is never frozen
    at construction — a rotating/expiring device or act-as token is picked up
    each call (the "token expires mid-run" bug ``credentials.py`` warns about).
    """

    def __init__(self, token_provider: TokenProvider) -> None:
        self._token_provider = token_provider

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self._token_provider()}"
        yield request


@dataclass
class StorageClient:
    """Client for uploading files to storage via pre-signed URLs.

    Uses the ``GET /api/storage/upload-url`` endpoint to obtain a pre-signed
    upload URL, then PUTs the file content directly to that URL.

    ``api_key`` is optional: when omitted the bearer resolves per request via
    the credential seam (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY``). Pass
    ``api_key`` to override, or ``token_provider`` for a custom per-call source.

    Example:
        client = StorageClient(api_key="sk_...", base_url="http://localhost:3000")
        result = client.upload_file(
            path="datasets/my-data.jsonl",
            content=b'{"q": "...", "a": "..."}',
            mime_type="application/jsonl",
        )
        print(f"Uploaded to {result['blobPath']}")
    """

    api_key: str | None = None
    base_url: str = field(default_factory=config.platform_url)
    timeout: float = 60.0
    # SAS-URL PUTs are bounded by file size, not API latency. Default to
    # 30 minutes so multi-GB datasets don't time out at the platform-API timeout.
    upload_timeout: float = 1800.0
    token_provider: TokenProvider | None = None
    _token_provider: TokenProvider = field(init=False, repr=False)
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client; auth resolves per request, never baked here."""
        self._token_provider = resolve_token_provider(self.api_key, self.token_provider)
        self._http_client = httpx.Client(
            base_url=self.base_url,
            auth=_BearerAuth(self._token_provider),
            timeout=self.timeout,
        )

    def __enter__(self) -> StorageClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Convert HTTP errors to appropriate exceptions."""
        if response.status_code in (200, 201):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code)

        raise TrainerError(message, response.status_code)

    def _get_upload_url(
        self, path: str, mime_type: str, expires_in_minutes: int | None = None
    ) -> dict:
        """Request a pre-signed upload URL from the storage API.

        Args:
            path: Storage path for the file
            mime_type: MIME type of the file
            expires_in_minutes: Optional expiration override for the signed URL

        Returns:
            Dict with ``uploadUrl``, ``expiresAt``, ``blobPath``, ``willOverwrite`` keys.
        """
        params: dict[str, Any] = {"path": path, "mimeType": mime_type}
        if expires_in_minutes is not None:
            params["expiresInMinutes"] = expires_in_minutes

        response = self._http_client.get(
            "/v1/storage/upload-url",
            params=params,
        )
        self._handle_response_errors(response)
        return response.json()

    def _put_to_signed_url(
        self,
        upload_url: str,
        content: bytes | Any,
        mime_type: str,
    ) -> None:
        """PUT file content directly to a pre-signed URL.

        ``content`` may be raw bytes or any object httpx accepts as a stream
        (file-like, generator). For large files prefer a file handle so
        memory usage stays bounded.
        """
        response = httpx.put(
            upload_url,
            content=content,
            headers={"Content-Type": mime_type, "x-ms-blob-type": "BlockBlob"},
            timeout=self.upload_timeout,
        )
        if response.status_code >= 400:
            raise TrainerError(
                f"Upload to storage failed: {response.status_code} {response.text}",
                response.status_code,
            )

    # === Core upload methods ===

    def upload_file(
        self,
        path: str,
        content: bytes,
        mime_type: str,
        *,
        expires_in_minutes: int | None = None,
    ) -> dict:
        """Upload raw bytes to storage.

        Args:
            path: Storage path for the file (e.g. "datasets/search/abc123/qa-dataset.jsonl")
            content: Raw bytes to upload
            mime_type: MIME type for the Content-Type header
            expires_in_minutes: Optional expiration override for the signed URL

        Returns:
            Dict with ``uploadUrl``, ``expiresAt``, ``blobPath``, ``willOverwrite`` keys.

        Raises:
            AuthenticationError: If API key is invalid
            TrainerError: If upload fails
        """
        url_response = self._get_upload_url(
            path,
            mime_type,
            expires_in_minutes=expires_in_minutes,
        )
        self._put_to_signed_url(url_response["uploadUrl"], content, mime_type)
        return url_response

    def upload_local_file(
        self,
        path: str,
        file_path: str | Path,
        *,
        expires_in_minutes: int | None = None,
    ) -> dict:
        """Upload a local file to storage.

        Args:
            path: Storage path for the file
            file_path: Local path to the file to upload
            expires_in_minutes: Optional expiration override for the signed URL

        Returns:
            Dict with ``uploadUrl``, ``expiresAt``, ``blobPath``, ``willOverwrite`` keys.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
            AuthenticationError: If API key is invalid
            TrainerError: If upload fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime_type = _get_mime_type(file_path)
        # Stream from disk instead of read_bytes() to keep memory bounded for
        # multi-GB datasets. httpx infers Content-Length from the file size.
        url_response = self._get_upload_url(
            path,
            mime_type,
            expires_in_minutes=expires_in_minutes,
        )
        with file_path.open("rb") as fh:
            self._put_to_signed_url(url_response["uploadUrl"], fh, mime_type)
        return url_response


@dataclass
class TrainerClient:
    """Client for launching and managing training runs.

    ``api_key`` is optional: when omitted the bearer resolves per request via
    the credential seam (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY``). Pass
    ``api_key`` to override, or ``token_provider`` for a custom per-call source.

    Example:
        client = TrainerClient(api_key="sk_...", base_url="http://localhost:3000")
        run_id = client.launch_training_run(
            training_run_type="simple",
            env_cls_path="envs/run-abc/abc123/env-cls.pkl",
            env_metadata_path="envs/run-abc/abc123/env-metadata.json",
            train_dataset_path="datasets/run-abc/def456/train.jsonl",
            eval_dataset_path="datasets/run-abc/def456/eval.jsonl",
        )
        print(f"Launched: {run_id}")
    """

    api_key: str | None = None
    base_url: str = field(default_factory=config.platform_url)
    timeout: float = 30.0
    token_provider: TokenProvider | None = None
    _token_provider: TokenProvider = field(init=False, repr=False)
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client; auth resolves per request, never baked here."""
        self._token_provider = resolve_token_provider(self.api_key, self.token_provider)
        self._http_client = httpx.Client(
            base_url=self.base_url,
            auth=_BearerAuth(self._token_provider),
            timeout=self.timeout,
        )

    def __enter__(self) -> TrainerClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Convert HTTP errors to appropriate exceptions."""
        if response.status_code in (200, 201, 202):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code)

        raise JobLaunchError(message, response.status_code)

    def launch_training_run(
        self,
        training_run_type: str,
        env_cls_path: str,
        env_metadata_path: str,
        train_dataset_path: str,
        eval_dataset_path: str,
        name: str | None = None,
        launcher_args: dict[str, Any] | None = None,
    ) -> str:
        """Launch a new training run from a job template.

        Args:
            training_run_type: Job template selector. ``"simple"`` (GPU pool —
                gpu4 for 4B, gpu8 for 35B) or ``"simple-cpu"`` (CPU-only smoke
                pool, no GPU).
            env_cls_path: Path to the environment class pickle (.pkl file)
            env_metadata_path: Path to the environment metadata JSON file
            train_dataset_path: Path to the training dataset
            eval_dataset_path: Path to the evaluation dataset
            name: Optional name for the training run
            launcher_args: Extra launcher args forwarded to the server
                (e.g. {"max_rollout_len": 4000}). The 4 required paths
                above always take precedence.

        Returns:
            The training run ID.

        Raises:
            AuthenticationError: If API key is invalid
            JobLaunchError: If training run launch fails
        """
        args: dict[str, Any] = {
            **(launcher_args or {}),
            "env_cls_path": env_cls_path,
            "env_metadata_path": env_metadata_path,
            "train_dataset_path": train_dataset_path,
            "eval_dataset_path": eval_dataset_path,
        }
        response = self._http_client.post(
            "/v1/train/runs/launch",
            json={
                "type": training_run_type,
                "name": name,
                "args": args,
            },
        )
        self._handle_response_errors(response)
        body = response.json()
        # Surface soft-cap / OOM-risk warnings via the warnings module (shown by
        # default in notebooks/REPL) — a bare logger.warning is swallowed unless
        # the caller configured logging.
        for warning in body.get("warnings", []) or []:
            warnings.warn(f"launch warning: {warning}", stacklevel=2)
        return body["runId"]

    def list_launch_args(self) -> list[LaunchArgSpec]:
        """Fetch the schema of launch args this platform accepts.

        Each entry describes a single tunable on POST /train/runs/launch:
        wire name, type, default, validation range. Use this to introspect
        what can be set in ``launcher_args`` when calling
        ``launch_training_run``.

        Returns:
            List of LaunchArgSpec, in the platform's preferred display order.

        Raises:
            AuthenticationError: If the API key is invalid.
            TrainerError: If the request fails.
        """
        response = self._http_client.get("/v1/train/launch-args")
        self._handle_response_errors(response)
        raw = response.json()["args"]
        return [
            LaunchArgSpec(
                name=spec["name"],
                label=spec["label"],
                type=spec["type"],
                required=spec["required"],
                description=spec["description"],
                default=spec.get("default"),
                min=spec.get("min"),
                max=spec.get("max"),
                warn_above=spec.get("warnAbove"),
                enum=tuple(spec["enum"]) if spec.get("enum") is not None else None,
            )
            for spec in raw
        ]

    def print_launch_args(self) -> None:
        """Pretty-print the launch-arg schema as a table.

        Convenience helper for REPL / notebook discoverability. Prints to
        stdout — does not return anything. For programmatic use, call
        :meth:`list_launch_args` instead.
        """
        specs = self.list_launch_args()
        print(_hdr("Launch args accepted by POST /train/runs/launch"))
        for spec in specs:
            req = (
                _RED + "required" + _RESET
                if spec.required
                else _CYAN + "optional" + _RESET
            )
            header = f"  {_BOLD}{spec.name}{_RESET} ({spec.type}, {req})"
            bits: list[str] = []
            if spec.default is not None:
                bits.append(f"default={spec.default!r}")
            if spec.min is not None:
                bits.append(f"min={spec.min}")
            if spec.max is not None:
                bits.append(f"max={spec.max}")
            if spec.warn_above is not None:
                bits.append(f"warn_above={spec.warn_above}")
            if spec.enum:
                bits.append(f"enum={spec.enum}")
            print(header + (f"  [{', '.join(bits)}]" if bits else ""))
            print(f"      {spec.description}")

    # --- Run-lifecycle reads (CLI: `castform runs ...`) -------------------
    # Thin GETs over platform REST. The CLI formats; these just return the
    # decoded JSON. Auth posture differs per route — see each docstring.

    def list_runs(self, *, include_config: bool = False) -> list[dict[str, Any]]:
        """GET /v1/train/runs — the caller's runs. Hard auth: 401 if logged out."""
        params = {"includeConfig": "true"} if include_config else None
        response = self._http_client.get("/v1/train/runs", params=params)
        self._handle_response_errors(response)
        return response.json()

    def get_run(self, run_id: str, *, include_config: bool = False) -> dict[str, Any]:
        """GET /v1/train/runs/{id} — optionalAuth (public runs readable logged-out)."""
        params = {"includeConfig": "true"} if include_config else None
        response = self._http_client.get(f"/v1/train/runs/{run_id}", params=params)
        self._handle_response_errors(response)
        return response.json()

    def get_run_details(self, run_id: str) -> dict[str, Any]:
        """GET /v1/train/runs/{id}/details — adds latestStep, modes[], errorCount."""
        response = self._http_client.get(f"/v1/train/runs/{run_id}/details")
        self._handle_response_errors(response)
        return response.json()

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        """GET /v1/train/runs/{id}/events — lifecycle/artifact/diagnostic, oldest first."""
        response = self._http_client.get(f"/v1/train/runs/{run_id}/events")
        self._handle_response_errors(response)
        return response.json().get("events", [])

    def get_run_scalars(
        self, run_id: str, mode: str
    ) -> dict[str, list[dict[str, Any]]]:
        """GET /v1/train/runs/{id}/scalars?mode= — {scalarName: [{step, value}]}.

        ``mode`` is required by the server (400 if omitted). Discover the valid
        set from :meth:`get_run_details` — ``modes`` is dynamic per run, not a
        fixed enum. An unknown mode returns ``{}`` (not an error).
        """
        response = self._http_client.get(
            f"/v1/train/runs/{run_id}/scalars", params={"mode": mode}
        )
        self._handle_response_errors(response)
        return response.json()

    def get_environment_logs(
        self, run_id: str, *, rollout_id: str | None = None
    ) -> list[dict[str, Any]]:
        """GET /v1/train/runs/{id}/environment-logs — run-level logs, or one
        rollout's when ``rollout_id`` is given."""
        params = {"rolloutId": rollout_id} if rollout_id else None
        response = self._http_client.get(
            f"/v1/train/runs/{run_id}/environment-logs", params=params
        )
        self._handle_response_errors(response)
        return response.json().get("logs", [])

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """POST /v1/train/runs/{id}/cancel — owner-only (403 otherwise).

        A launched run (has a launcher job) gets a ``training.cancelling`` event
        and its launcher job cancelled ("Job cancellation requested"); a run with
        no job is marked complete directly with no cancelling event
        ("Marked as complete"). Returns the ``{success, message}`` body.
        """
        response = self._http_client.post(f"/v1/train/runs/{run_id}/cancel")
        self._handle_response_errors(response)
        return response.json()


# Server URLs are resolved lazily via callables so env-var changes after
# import (e.g. test fixtures setting CASTFORM_BASE_DOMAIN) take effect.
# Resolving once at import froze RolloutClient on the first-seen URL while
# StorageClient/TrainerClient picked up env changes at construction time —
# tests for one client could pass while the other silently hit prod.
_VALIDATION_MODEL = "gpt-5.4-nano"

# ANSI colours
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> str:
    return f"{_GREEN}✔  {msg}{_RESET}"


def _err(msg: str) -> str:
    return f"{_RED}✗  {msg}{_RESET}"


def _info(msg: str) -> str:
    return f"{_CYAN}{msg}{_RESET}"


def _hdr(msg: str) -> str:
    return f"\n{_BOLD}{msg}{_RESET}"


def _iter_sse(response: httpx.Response) -> Iterator[dict]:
    """Yield parsed event dicts from a synchronous SSE response."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            try:
                yield json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                pass


def _message_text(message: dict[str, Any]) -> str:
    """Extract plain text from a message payload."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(str(block.get("text", "")))
        return "\n".join(t for t in text_blocks if t)

    return ""


_DEBUG_KEY_SUBSTRINGS = (
    "finish_reason",
    "stop_reason",
    "termination",
    "max_turn",
    "max_tool",
    "turn_count",
    "tool_call",
    "token",
    "usage",
)
_DEBUG_SKIP_KEYS = {"message", "messages", "content", "assistant_messages"}


def _debug_event_fields(
    value: Any,
    *,
    path: str = "",
    out: dict[str, Any] | None = None,
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 40,
) -> dict[str, Any]:
    """Collect rollout diagnostics from terminal events."""
    if out is None:
        out = {}
    if depth > max_depth or len(out) >= max_items:
        return out

    if isinstance(value, dict):
        for key, child in value.items():
            if key in _DEBUG_SKIP_KEYS:
                continue
            child_path = f"{path}.{key}" if path else key
            key_l = key.lower()
            path_l = child_path.lower()
            is_tracked = any(k in key_l or k in path_l for k in _DEBUG_KEY_SUBSTRINGS)

            if isinstance(child, (dict, list)):
                _debug_event_fields(
                    child,
                    path=child_path,
                    out=out,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
                continue

            if is_tracked and len(out) < max_items:
                if isinstance(child, str) and len(child) > 240:
                    out[child_path] = f"{child[:240]}…"
                else:
                    out[child_path] = child
    elif isinstance(value, list):
        for i, child in enumerate(value):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            _debug_event_fields(
                child,
                path=child_path,
                out=out,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
            if len(out) >= max_items:
                break
    return out


def _print_multiline(prefix: str, text: str) -> None:
    """Print potentially multi-line text with indentation."""
    lines = str(text).splitlines() or [""]
    print(prefix)
    for line in lines:
        print(f"      {line}")


def _print_event(
    event: dict,
    idx: int,
    *,
    full_messages: bool = False,
    include_event_meta: bool = True,
) -> None:
    """Print a single SSE event in a human-readable format."""
    etype = event.get("event", "?")
    prefix = f"  [ex {idx}]"

    if etype == "rollout_started":
        print(_info(f"{prefix} → rollout_started"))

    elif etype == "message":
        msg = event.get("message", {})
        role = msg.get("role", "?")
        content = msg.get("content", "")

        # content may be a list of blocks (tool calls) or a plain string
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        text = str(block.get("text", ""))
                        if full_messages:
                            _print_multiline(
                                f"{prefix} → message [{role}/text] (chars={len(text)}):",
                                text,
                            )
                        else:
                            preview = textwrap.shorten(text, width=120, placeholder="…")
                            print(
                                f"{prefix} → message [{role}/text] (chars={len(text)}): {preview}"
                            )
                    elif btype == "tool_use":
                        call_text = (
                            f"{block.get('name')}("
                            f"{json.dumps(block.get('input', {}), ensure_ascii=True)}"
                            ")"
                        )
                        if full_messages:
                            _print_multiline(
                                f"{prefix} → message [{role}/tool_use] (chars={len(call_text)}):",
                                call_text,
                            )
                        else:
                            print(
                                f"{prefix} → message [{role}/tool_use] "
                                f"(chars={len(call_text)}): "
                                f"{textwrap.shorten(call_text, width=120, placeholder='…')}"
                            )
                    elif btype == "tool_result":
                        tool_text = str(block.get("content", ""))
                        if full_messages:
                            _print_multiline(
                                f"{prefix} → message [{role}/tool_result] "
                                f"(chars={len(tool_text)}):",
                                tool_text,
                            )
                        else:
                            preview = textwrap.shorten(
                                tool_text, width=120, placeholder="…"
                            )
                            print(
                                f"{prefix} → message [{role}/tool_result] "
                                f"(chars={len(tool_text)}): {preview}"
                            )
                    else:
                        print(f"{prefix} → message [{role}/{btype}]")
        else:
            raw = str(content)
            if full_messages:
                _print_multiline(
                    f"{prefix} → message [{role}] (chars={len(raw)}):",
                    raw,
                )
            else:
                preview = textwrap.shorten(raw, width=120, placeholder="…")
                print(f"{prefix} → message [{role}] (chars={len(raw)}): {preview}")

    elif etype == "reward":
        print(f"{prefix} → reward: {event.get('rewards')}")

    elif etype == "rollout_completed":
        success = event.get("success")
        status = _ok("success") if success else _err("failed")
        print(
            f"{prefix} → rollout_completed  {status}  "
            f"rewards={event.get('rewards')}  error={event.get('error')}"
        )

    elif etype in ("worker_error", "error", "cancelled"):
        print(_err(f"{prefix} → {etype}: {event.get('error')}"))
        if include_event_meta:
            meta = _debug_event_fields(event)
            if meta:
                print(f"{prefix} → {etype}_meta: {json.dumps(meta, ensure_ascii=True)}")


class RolloutClient:
    """Thin synchronous client for the rollout-stream endpoint.

    Rollouts are reached through platform-service. platform-service is the API-key
    gate: it validates the ``sk_`` key and mints a short-lived act_as JWT that
    rollout-service accepts (rollout-service's own auth only takes
    auth-service-minted JWTs — never a raw platform key). The proxy is mounted at
    ``/v1/rollout/stream``.

    Supports two ways to provide the environment:

    1. **Blob paths** (default) — pass ``env_cls_path`` and ``env_metadata_path``
       pointing to already-uploaded blobs.
    2. **Raw bytes** — pass ``env_cls_bytes`` and ``env_metadata_bytes`` with the
       raw file contents; they will be base64-encoded and sent inline.

    Args:
        api_key:    Platform API key forwarded as the Bearer token
                    platform-service validates. Optional — when omitted the
                    bearer resolves per request via the credential seam
                    (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY``).
        server_url: Base URL of platform-service. Defaults to
                    ``config.platform_url()``; the ``/v1/rollout/stream`` path is
                    appended per request.
        timeout:    Per-request timeout in seconds (default 300 — rollouts can be slow).
        token_provider: Custom per-call bearer source; overrides the seam when
                    ``api_key`` is unset.
    """

    _TERMINAL = {"rollout_completed", "worker_error", "cancelled", "error"}

    def __init__(
        self,
        api_key: str | None = None,
        server_url: str | None = None,
        timeout: float = 300.0,
        *,
        token_provider: TokenProvider | None = None,
    ) -> None:
        # Bearer resolves per request (see stream_rollout): explicit api_key →
        # token_provider → platform_bearer seam. Optional so a logged-in/CI
        # caller need not pass one.
        self._token_provider = resolve_token_provider(api_key, token_provider)
        # Resolve at construction time, not import time, so env-var changes
        # take effect (mirrors StorageClient/TrainerClient default_factory pattern).
        # Target platform-service (the API-key gate), not the rollout-service
        # host directly — see the class docstring for why.
        self._server_url = (server_url or config.platform_url()).rstrip("/")
        self._timeout = timeout

    @staticmethod
    def _build_env(
        env_cls_path: str | None,
        env_metadata_path: str | None,
        env_cls_bytes: bytes | None,
        env_metadata_bytes: bytes | None,
    ) -> dict[str, str]:
        """Build the ``env`` dict for the request payload.

        Exactly one of (paths) or (bytes) must be provided.
        """
        has_paths = env_cls_path is not None and env_metadata_path is not None
        has_bytes = env_cls_bytes is not None and env_metadata_bytes is not None

        if has_paths and has_bytes:
            raise ValueError(
                "Provide either blob paths or raw bytes for the env, not both."
            )
        if not has_paths and not has_bytes:
            raise ValueError(
                "Provide either (env_cls_path, env_metadata_path) or "
                "(env_cls_bytes, env_metadata_bytes)."
            )

        if has_paths:
            return {
                "env_cls_path": env_cls_path,  # type: ignore[dict-item]
                "env_metadata_path": env_metadata_path,  # type: ignore[dict-item]
            }

        return {
            "env_cls_bytes": base64.b64encode(env_cls_bytes).decode(),  # type: ignore[arg-type]
            "env_metadata_bytes": base64.b64encode(env_metadata_bytes).decode(),  # type: ignore[arg-type]
        }

    def stream_rollout(
        self,
        raw_example: dict[str, Any],
        env_cls_path: str | None = None,
        env_metadata_path: str | None = None,
        *,
        env_cls_bytes: bytes | None = None,
        env_metadata_bytes: bytes | None = None,
        example_index: int = 0,
        llm_base_url: str | None = None,
        llm_model: str = _VALIDATION_MODEL,
        llm_api_key: str = "",
        llm_api_version: str = "",
        max_turns: int = 4,
        max_tool_calls: int = 8,
        max_completion_tokens: int = 4024,
        capture_messages: bool = False,
        full_messages: bool = False,
        include_event_meta: bool = True,
    ) -> dict[str, Any]:
        """Run one rollout against the stream endpoint, printing events live.

        The environment can be specified via **blob paths** or **raw bytes**
        (mutually exclusive — see class docstring).

        Args:
            raw_example:           Raw dataset row (passed as ``raw_example``).
            env_cls_path:          Blob path to the uploaded env .pkl file.
            env_metadata_path:     Blob path to the uploaded env-meta .json file.
            env_cls_bytes:         Raw bytes of the pickled env class (will be base64-encoded).
            env_metadata_bytes:    Raw bytes of the env metadata JSON (will be base64-encoded).
            example_index:         Display index used in printed output.
            llm_base_url:          Base URL for the LLM API.
            llm_model:             Model name to use for the rollout.
            max_turns:             Max conversation turns.
            max_tool_calls:        Max tool calls per rollout.
            max_completion_tokens: Max tokens per completion.
            capture_messages:      Whether to include full streamed messages in the return payload.
            full_messages:         Print full message text/JSON instead of truncated previews.
            include_event_meta:    Print terminal event diagnostic fields (finish/token/limit keys).

        Returns:
            The final ``rollout_completed`` event dict, or an error/cancelled event.
            When ``capture_messages=True``, includes:
              - ``messages``: all streamed message payloads
              - ``assistant_messages``: assistant-only subset
              - ``final_assistant_text``: text extracted from the last assistant message

        Raises:
            ValueError: If neither paths nor bytes are provided, or both are.
            RuntimeError: If the HTTP request itself fails or returns non-200.
        """
        env = self._build_env(
            env_cls_path,
            env_metadata_path,
            env_cls_bytes,
            env_metadata_bytes,
        )

        # Resolve the platform bearer once, per request (never frozen at
        # construction): a rotating/expiring device or act-as token is picked
        # up each call. Used for the platform-service header below AND, when the
        # LLM leg hits the platform's own endpoint, as that leg's key.
        bearer = self._token_provider()

        # Resolve LLM URL lazily. The platform key is only auto-forwarded when
        # the LLM endpoint is the platform's own LLM service — pointing at a
        # third-party host (Azure OpenAI, Anthropic) requires an explicit
        # llm_api_key so we don't silently leak the platform credential.
        platform_llm_url = config.llm_url()
        resolved_llm_url = llm_base_url or platform_llm_url
        if not llm_api_key:
            if resolved_llm_url == platform_llm_url:
                llm_api_key = bearer
            else:
                raise ValueError(
                    "llm_api_key is required when llm_base_url points outside the "
                    f"platform LLM endpoint ({platform_llm_url}). Refusing to "
                    "forward the platform API key to a third-party host."
                )

        payload = {
            "standardized_example": None,
            "raw_example": raw_example,
            "env": env,
            "llm": {
                "base_url": resolved_llm_url,
                "api_key": llm_api_key,
                "model": llm_model,
                "api-version": llm_api_version,
            },
            "options": {
                "max_turns": max_turns,
                "max_tool_calls": max_tool_calls,
                "max_completion_tokens": max_completion_tokens,
            },
        }

        # platform-service mounts the proxy at /v1/rollout/stream; it validates
        # the platform key and forwards to rollout-service with an act_as JWT.
        url = f"{self._server_url}/v1/rollout/stream"
        headers = {"Authorization": f"Bearer {bearer}"}

        with httpx.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=self._timeout,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode()
                # Typed errors so callers can distinguish retryable from
                # caller-fix from auth-fix without parsing exception messages.
                # 403 too: rollouts route through platform-service's optionalAuth
                # gate, which rejects a missing/invalid/expired key as 403
                # ("sign in to run rollouts") rather than 401 — same fix (the key).
                if response.status_code in (401, 403):
                    raise AuthenticationError(body[:300], response.status_code)
                if response.status_code == 404:
                    raise RolloutNotFound(body[:300], response.status_code)
                if 500 <= response.status_code < 600:
                    raise RolloutServerError(body[:300], response.status_code)
                raise RolloutError(body[:300], response.status_code)

            final: dict[str, Any] = {}
            messages: list[dict[str, Any]] = []
            try:
                for event in _iter_sse(response):
                    _print_event(
                        event,
                        example_index,
                        full_messages=full_messages,
                        include_event_meta=include_event_meta,
                    )
                    if capture_messages and event.get("event") == "message":
                        message = event.get("message")
                        if isinstance(message, dict):
                            messages.append(message)
                    if event.get("event") in self._TERMINAL:
                        final = event
                        break
                else:
                    # Stream closed without a terminal event — server crashed
                    # mid-rollout, network reset, or proxy timeout. Raise a
                    # specific error so callers don't see "missing 'success'
                    # key" downstream confusion.
                    raise RolloutStreamError(
                        "Rollout stream ended without a terminal event "
                        f"(expected one of {sorted(self._TERMINAL)}). "
                        "The server may have crashed or the connection dropped."
                    )
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                raise RolloutStreamError(
                    f"Rollout stream timed out or disconnected after "
                    f"{self._timeout}s of inactivity: {exc}"
                ) from exc

        if capture_messages:
            assistant_messages = [
                message for message in messages if message.get("role") == "assistant"
            ]
            final_assistant_text = (
                _message_text(assistant_messages[-1]) if assistant_messages else ""
            )
            final = {
                **final,
                "messages": messages,
                "assistant_messages": assistant_messages,
                "final_assistant_text": final_assistant_text,
            }

        return final

    def validate_examples(
        self,
        examples: list[dict[str, Any]],
        env_cls_path: str | None = None,
        env_metadata_path: str | None = None,
        n: int = 2,
        *,
        env_class: type[BaseEnv] | None = None,
        constructor_args: dict[str, Any] | None = None,
        pip_dependencies: list[str] | None = None,
        local_modules: list[ModuleType] | None = None,
        env_cls_bytes: bytes | None = None,
        env_metadata_bytes: bytes | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str = "",
        llm_model: str = _VALIDATION_MODEL,
        max_turns: int = 4,
        check_group_reward: bool = True,
        group_reward_samples: int = 2,
        verbose: bool = True,
        full_messages: bool = False,
    ) -> ValidationResult:
        """Run rollouts on the first *n* examples and report pass/fail.

        The environment can be specified three ways (mutually exclusive): an
        **env class** (bundled to bytes here, so validation needs no prior
        upload — preferred for a pre-launch smoke test), **blob paths** to an
        already-uploaded env, or **raw bytes** (see class docstring).

        Args:
            examples:           Full dataset (list of raw dicts).
            env_cls_path:       Blob path to the uploaded env .pkl file.
            env_metadata_path:  Blob path to the uploaded env-meta .json file.
            n:                  Number of examples to validate (default 2).
            env_class:          BaseEnv subclass to bundle and validate without
                                uploading. Mutually exclusive with paths/bytes.
            constructor_args:   kwargs baked into the env bundle (env_class only).
            pip_dependencies:   Pip deps recorded in the bundle (env_class only).
            local_modules:      Modules to pickle by-value (env_class only; for
                                envs that import from local .py files).
            env_cls_bytes:      Raw bytes of the pickled env class (will be base64-encoded).
            env_metadata_bytes: Raw bytes of the env metadata JSON (will be base64-encoded).
            llm_base_url:       Base URL for the rollout's LLM leg. Defaults to
                                ``config.llm_url()``. Pass explicitly when the
                                script's environment differs from the SDK default
                                (otherwise the rollout LLM silently hits the
                                default-domain host — see stream_rollout).
            llm_api_key:        API key for the LLM leg. Required when
                                ``llm_base_url`` points outside the platform LLM
                                endpoint (stream_rollout refuses to forward the
                                platform key to a third-party host).
            check_group_reward: After the rollouts, run a REAL same-example
                                group through rollout-service (one example,
                                samples_per_example=N) so the env's
                                ``compute_group_reward`` executes server-side in
                                the trainer image, over co-located siblings —
                                the trainer/external-eval path. A server-side
                                failure (raise or contract violation) comes back
                                as ``group_reward_error`` and fails validation.
                                Only fires when ``env_class`` is given and the
                                env overrides the method.
            group_reward_samples: Size of that group (the batch's
                                ``samples_per_example``). Costs this many extra
                                rollouts; ignored unless the group check runs.
            verbose:            Print colored progress to stdout (default True for
                                interactive/notebook UX). Set False for programmatic
                                callers that consume the returned ValidationResult.

        Returns:
            ValidationResult — bool-castable (``if not result``) for the common
            "did everything pass" check, with per-example detail in
            ``result.examples`` for richer reporting.
        """
        # An env class is bundled to bytes here so validation can run a smoke
        # test BEFORE uploading anything (the launch flow uploads only after
        # validation passes). Mutually exclusive with explicit paths/bytes.
        if env_class is not None:
            if any(
                (env_cls_path, env_metadata_path, env_cls_bytes, env_metadata_bytes)
            ):
                raise ValueError(
                    "Provide env_class OR explicit env paths/bytes, not both."
                )
            from benchmax.bundle import dump_bundle

            bundle = dump_bundle(
                env_class,
                constructor_args=constructor_args,
                pip_dependencies=pip_dependencies,
                local_modules=local_modules,
            )
            env_cls_bytes = bundle.pickled
            env_metadata_bytes = bundle.metadata.to_json_bytes()

        # Validate env args early so we fail before running any rollouts.
        self._build_env(
            env_cls_path, env_metadata_path, env_cls_bytes, env_metadata_bytes
        )

        # compute_group_reward runs on a whole rollout GROUP, which the
        # per-example smoke above never forms (each is a group of 1). Run a real
        # same-example group server-side (run_group, below) so the env method
        # executes on the trainer's path; the server reports any failure as
        # group_reward_error. Needs the env_class, and is pointless unless the
        # env overrides the no-op default.
        want_group = False
        if check_group_reward and env_class is not None:
            from .validation import overrides_compute_group_reward

            want_group = overrides_compute_group_reward(env_class)

        sample = examples[:n]
        if verbose:
            print(
                _hdr(
                    f"── Remote validation: {len(sample)} example(s) on {llm_model} ──"
                )
            )

        per_example: list[ExampleValidation] = []
        for i, example in enumerate(sample):
            if verbose:
                print(_info(f"\n  Example {i} — {json.dumps(example)[:120]}"))
            try:
                final = self.stream_rollout(
                    raw_example=example,
                    env_cls_path=env_cls_path,
                    env_metadata_path=env_metadata_path,
                    env_cls_bytes=env_cls_bytes,
                    env_metadata_bytes=env_metadata_bytes,
                    example_index=i,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    max_turns=max_turns,
                    full_messages=full_messages,
                )
                ok = bool(final.get("success"))
                per_example.append(
                    ExampleValidation(
                        index=i,
                        ok=ok,
                        error=None
                        if ok
                        else (final.get("error") or "rollout reported success=False"),
                        # Surface the per-rollout reward components (dropped here
                        # before — they're what `castform validate` displays).
                        rewards=final.get("rewards"),
                    )
                )
            except (RolloutError, RuntimeError) as exc:
                if verbose:
                    print(_err(f"  Example {i} failed: {exc}"))
                per_example.append(ExampleValidation(index=i, ok=False, error=str(exc)))

        group_reward: ExampleValidation | None = None
        if want_group and sample and group_reward_samples >= 1:
            # Faithful check: run a REAL same-example group through
            # rollout-service (one example, samples_per_example=N) so the env's
            # compute_group_reward runs server-side in the trainer image, over
            # co-located siblings — exactly the trainer/external-eval path. A
            # server-side failure comes back as group_reward_error per rollout.
            if verbose:
                print(
                    _info(
                        f"\n  Group reward — {group_reward_samples} server-side "
                        "sibling(s) of example 0"
                    )
                )
            try:
                events = self.run_group(
                    sample[0],
                    samples=group_reward_samples,
                    env_cls_path=env_cls_path,
                    env_metadata_path=env_metadata_path,
                    env_cls_bytes=env_cls_bytes,
                    env_metadata_bytes=env_metadata_bytes,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    max_turns=max_turns,
                    verbose=verbose,
                )
                group_reward = self._assess_group_events(
                    events, group_reward_samples, verbose
                )
            except RolloutNotFound:
                # The batch proxy (/v1/rollout/batch/stream) isn't deployed on
                # this server yet — skip rather than fail, so the SDK can land
                # ahead of platform-service. group_reward stays None; the offline
                # local check still covered shape.
                if verbose:
                    print(
                        _info(
                            "  compute_group_reward: skipped — server has no "
                            "/rollout/batch/stream yet"
                        )
                    )
            except (RolloutError, RuntimeError) as exc:
                if verbose:
                    print(_err(f"  group reward check failed: {exc}"))
                group_reward = ExampleValidation(index=-1, ok=False, error=str(exc))

        result = ValidationResult(examples=per_example, group_reward=group_reward)
        if verbose:
            print()
            if result.ok:
                print(_ok("Remote validation passed"))
            else:
                print(
                    _err(
                        "Remote validation failed — check output above before launching a full job"
                    )
                )

        return result

    def run_group(
        self,
        example: dict[str, Any],
        *,
        samples: int,
        env_cls_path: str | None = None,
        env_metadata_path: str | None = None,
        env_cls_bytes: bytes | None = None,
        env_metadata_bytes: bytes | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str = "",
        llm_model: str = _VALIDATION_MODEL,
        max_turns: int = 4,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run ONE example as a real ``samples``-member group; return its
        ``rollout_completed`` events.

        Submits a one-row batch with ``samples_per_example=samples`` to
        ``/v1/rollout/batch/stream``. rollout-service co-locates the siblings on
        one worker and runs ``env.compute_group_reward`` over them — the same
        path the trainer/external-eval use, in the trainer image. Each event
        carries ``success``, ``rewards`` and (on a server new enough to report
        it) ``group_reward_error``. Raises the same typed errors as
        ``stream_rollout`` on a non-200.
        """
        env = self._build_env(
            env_cls_path, env_metadata_path, env_cls_bytes, env_metadata_bytes
        )
        # Resolve the platform bearer once, per request (never frozen at
        # construction) — used for the request header below AND, when the LLM leg
        # hits the platform's own endpoint, as that leg's key. Mirrors stream_rollout.
        bearer = self._token_provider()

        # The platform key is only auto-forwarded to the platform's own LLM host
        # (see stream_rollout for the no-leak rationale).
        platform_llm_url = config.llm_url()
        resolved_llm_url = llm_base_url or platform_llm_url
        if not llm_api_key:
            if resolved_llm_url == platform_llm_url:
                llm_api_key = bearer
            else:
                raise ValueError(
                    "llm_api_key is required when llm_base_url points outside the "
                    f"platform LLM endpoint ({platform_llm_url}). Refusing to "
                    "forward the platform API key to a third-party host."
                )

        payload = {
            "dataset_bytes": base64.b64encode(json.dumps(example).encode()).decode(),
            "is_dataset_standardized": False,
            # One example → one group; compute_group_reward needs all siblings
            # co-located on a single worker anyway. Pin to 1 so rollout-service
            # doesn't spin up extra workers that get an empty partition and crash.
            "concurrent_workers": 1,
            "env": env,
            "llm": {
                "base_url": resolved_llm_url,
                "api_key": llm_api_key,
                "model": llm_model,
            },
            "options": {"max_turns": max_turns, "samples_per_example": samples},
        }
        # platform-service mounts the batch proxy at /v1/rollout/batch/stream; it
        # validates the platform key and forwards to rollout-service with an
        # act_as JWT, same as the single /v1/rollout/stream proxy.
        url = f"{self._server_url}/v1/rollout/batch/stream"
        headers = {"Authorization": f"Bearer {bearer}"}

        completed: list[dict[str, Any]] = []
        with httpx.stream(
            "POST", url, json=payload, headers=headers, timeout=self._timeout
        ) as response:
            if response.status_code != 200:
                body = response.read().decode()
                if response.status_code in (401, 403):
                    raise AuthenticationError(body[:300], response.status_code)
                if response.status_code == 404:
                    raise RolloutNotFound(body[:300], response.status_code)
                if 500 <= response.status_code < 600:
                    raise RolloutServerError(body[:300], response.status_code)
                raise RolloutError(body[:300], response.status_code)

            for event in _iter_sse(response):
                etype = event.get("event")
                if etype == "batch_started":
                    if verbose:
                        print(
                            _info(
                                f"  group batch started ({event.get('total')} rollouts)"
                            )
                        )
                elif etype == "rollout_completed":
                    completed.append(event)
                elif etype == "worker_error":
                    # A sandbox process crashed. Non-fatal to the group on its
                    # own — the verdict comes from the rollout_completed events
                    # (_assess_group_events fails if none succeeded). Surfaced
                    # for visibility, not raised.
                    if verbose:
                        print(_err(f"  worker_error: {str(event.get('error'))[:200]}"))
                elif etype == "error":
                    raise RolloutError(str(event.get("error"))[:300], 500)
                elif etype in ("batch_completed", "cancelled"):
                    break
        return completed

    def _assess_group_events(
        self,
        events: list[dict[str, Any]],
        samples: int,
        verbose: bool,
    ) -> ExampleValidation:
        """Turn a group's ``rollout_completed`` events into a pass/fail verdict.

        Fails when the server reported a ``group_reward_error`` for any sibling
        — compute_group_reward raised or violated its contract (see
        rollout-service's ``_compute_group_rewards_safe``). Also fails if every
        group rollout failed (nothing to assess). Index -1.

        Note: a server that predates ``group_reward_error`` can't report a
        failure, so a green verdict there means "no failure observed", not
        "verified" — the offline local check still covers shape regardless.
        """
        errors = [
            e["group_reward_error"] for e in events if e.get("group_reward_error")
        ]
        if errors:
            msg = str(errors[0])
            if verbose:
                print(_err(f"  compute_group_reward FAILED server-side: {msg}"))
            return ExampleValidation(index=-1, ok=False, error=msg)

        succeeded = [e for e in events if e.get("success")]
        if not succeeded:
            first = next(
                (e.get("error") for e in events if e.get("error")),
                "no successful group rollouts",
            )
            if verbose:
                print(
                    _err(
                        "  group reward not validated — all group rollouts "
                        f"failed: {first}"
                    )
                )
            return ExampleValidation(
                index=-1, ok=False, error=f"all group rollouts failed: {first}"
            )

        # Mean of the succeeded siblings' final rewards — the numbers the group
        # path computed and used to discard before.
        mean = _mean_rewards([e.get("rewards") for e in succeeded])
        if verbose:
            print(
                _ok(
                    "  compute_group_reward OK server-side on a group of "
                    f"{len(succeeded)}"
                )
            )
        return ExampleValidation(index=-1, ok=True, rewards=mean)
