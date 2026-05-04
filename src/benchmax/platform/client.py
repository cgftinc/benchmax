"""Trainer API clients for storage uploads and job management."""

from __future__ import annotations

import base64
import hashlib
import json
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .exceptions import AuthenticationError, JobLaunchError, TrainerError

if TYPE_CHECKING:
    pass


# MIME type mappings for common file extensions
_MIME_TYPES = {
    ".jsonl": "application/jsonl",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".pkl": "application/octet-stream",
    ".pickle": "application/octet-stream",
    ".bmxp": "application/octet-stream",
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


@dataclass
class StorageClient:
    """Client for uploading files to storage via pre-signed URLs.

    Uses the ``GET /api/storage/upload-url`` endpoint to obtain a pre-signed
    upload URL, then PUTs the file content directly to that URL.

    Example:
        client = StorageClient(api_key="sk_...", base_url="http://localhost:3000")
        result = client.upload_file(
            path="datasets/my-data.jsonl",
            content=b'{"q": "...", "a": "..."}',
            mime_type="application/jsonl",
        )
        print(f"Uploaded to {result['blobPath']}")
    """

    api_key: str
    base_url: str = "http://localhost:3000"
    timeout: float = 60.0
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client with auth headers."""
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
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
            "/api/storage/upload-url",
            params=params,
        )
        self._handle_response_errors(response)
        return response.json()

    def _put_to_signed_url(self, upload_url: str, content: bytes, mime_type: str) -> None:
        """PUT file content directly to a pre-signed URL."""
        response = httpx.put(
            upload_url,
            content=content,
            headers={"Content-Type": mime_type, "x-ms-blob-type": "BlockBlob"},
            timeout=self.timeout,
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
        content = file_path.read_bytes()
        return self.upload_file(
            path,
            content,
            mime_type,
            expires_in_minutes=expires_in_minutes,
        )


@dataclass
class TrainerClient:
    """Client for launching and managing training jobs.

    Example:
        client = TrainerClient(api_key="sk_...", base_url="http://localhost:3000")
        job = client.launch_job(
            job_type="search",
            args={"dataset": "datasets/search/abc123/qa-dataset.jsonl"},
        )
        print(f"Launched job: {job}")
    """

    api_key: str
    base_url: str = "http://localhost:3000"
    timeout: float = 30.0
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client with auth headers."""
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
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

    def launch_experiment(
        self,
        experiment_type: str,
        env_cls_path: str,
        env_metadata_path: str,
        train_dataset_path: str,
        eval_dataset_path: str,
        name: str | None = None,
        launcher_args: dict[str, Any] | None = None,
    ) -> str:
        """Launch a new experiment from a job template.
        Args:
            experiment_type: Type of experiment to launch (e.g. "search")
            env_cls_path: Path to the environment class bundle (.bmxp file)
            env_metadata_path: Path to the environment kwargs JSON file
            train_dataset_path: Path to the training dataset
            eval_dataset_path: Path to the evaluation dataset
            name: Optional name for the experiment
            launcher_args: Extra launcher args forwarded to the server
                (e.g. {"max_response_len": 4000}). The 4 required paths
                above always take precedence.
        Returns:
            The experiment ID.
        Raises:
            AuthenticationError: If API key is invalid
            JobLaunchError: If experiment launch fails
        """
        args: dict[str, Any] = {
            **(launcher_args or {}),
            "env_cls_path": env_cls_path,
            "env_metadata_path": env_metadata_path,
            "train_dataset_path": train_dataset_path,
            "eval_dataset_path": eval_dataset_path,
        }
        response = self._http_client.post(
            "/api/experiments/launch",
            json={
                "type": experiment_type,
                "name": name,
                "args": args,
            },
        )
        self._handle_response_errors(response)
        return response.json()["experimentId"]


ROLLOUT_SERVER_URL = "https://autobots.cgft.io"

_VALIDATION_MODEL = "gpt-5.4-nano"
_VALIDATION_LLM_BASE_URL = "https://llm.cgft.io/v1"

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
                            preview = textwrap.shorten(tool_text, width=120, placeholder="…")
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
    """Thin synchronous client for the /rollout/stream endpoint.

    Supports two ways to provide the environment:

    1. **Blob paths** (default) — pass ``env_cls_path`` and ``env_metadata_path``
       pointing to already-uploaded blobs.
    2. **Raw bytes** — pass ``env_cls_bytes`` and ``env_metadata_bytes`` with the
       raw file contents; they will be base64-encoded and sent inline.

    Args:
        api_key:    Bearer token for the rollout server.
        server_url: Base URL of the rollout server.
        timeout:    Per-request timeout in seconds (default 300 — rollouts can be slow).
    """

    _TERMINAL = {"rollout_completed", "worker_error", "cancelled", "error"}

    def __init__(
        self,
        api_key: str,
        server_url: str = ROLLOUT_SERVER_URL,
        timeout: float = 300.0,
    ) -> None:
        self._api_key = api_key
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout

    @staticmethod
    def _build_env(
        env_cls_path: str | None,
        env_metadata_path: str | None,
        env_cls_bytes: bytes | None,
        env_metadata_bytes: bytes | None,
        env_args_bytes: bytes | None = None,
    ) -> dict[str, str]:
        """Build the ``env`` dict for the request payload.

        Exactly one of (paths) or (bytes) must be provided.
        """
        has_paths = env_cls_path is not None and env_metadata_path is not None
        has_bytes = env_cls_bytes is not None and env_metadata_bytes is not None

        if has_paths and has_bytes:
            raise ValueError("Provide either blob paths or raw bytes for the env, not both.")
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

        result: dict[str, str] = {
            "env_cls_bytes": base64.b64encode(env_cls_bytes).decode(),  # type: ignore[arg-type]
            "env_metadata_bytes": base64.b64encode(env_metadata_bytes).decode(),  # type: ignore[arg-type]
        }
        if env_args_bytes is not None:
            result["env_args_bytes"] = base64.b64encode(env_args_bytes).decode()
        return result

    def stream_rollout(
        self,
        raw_example: dict[str, Any],
        env_cls_path: str | None = None,
        env_metadata_path: str | None = None,
        *,
        env_cls_bytes: bytes | None = None,
        env_metadata_bytes: bytes | None = None,
        env_args_bytes: bytes | None = None,
        example_index: int = 0,
        llm_base_url: str = _VALIDATION_LLM_BASE_URL,
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
            env_args_bytes=env_args_bytes,
        )

        payload = {
            "standardized_example": None,
            "raw_example": raw_example,
            "env": env,
            "llm": {
                "base_url": llm_base_url,
                "api_key": llm_api_key or self._api_key,
                "model": llm_model,
                "api-version": llm_api_version,
            },
            "options": {
                "max_turns": max_turns,
                "max_tool_calls": max_tool_calls,
                "max_completion_tokens": max_completion_tokens,
            },
        }

        url = f"{self._server_url}/rollout/stream"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=self._timeout,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode()
                raise RuntimeError(
                    f"Rollout server returned HTTP {response.status_code}: {body[:300]}"
                )

            final: dict[str, Any] = {}
            messages: list[dict[str, Any]] = []
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
        env_cls_bytes: bytes | None = None,
        env_metadata_bytes: bytes | None = None,
        llm_model: str = _VALIDATION_MODEL,
        max_turns: int = 4,
    ) -> bool:
        """Run rollouts on the first *n* examples and report pass/fail.

        The environment can be specified via **blob paths** or **raw bytes**
        (mutually exclusive — see class docstring).

        Args:
            examples:           Full dataset (list of raw dicts).
            env_cls_path:       Blob path to the uploaded env .pkl file.
            env_metadata_path:  Blob path to the uploaded env-meta .json file.
            n:                  Number of examples to validate (default 2).
            env_cls_bytes:      Raw bytes of the pickled env class (will be base64-encoded).
            env_metadata_bytes: Raw bytes of the env metadata JSON (will be base64-encoded).

        Returns:
            True if all sampled rollouts completed successfully, False otherwise.
        """
        # Validate env args early so we fail before running any rollouts.
        self._build_env(env_cls_path, env_metadata_path, env_cls_bytes, env_metadata_bytes)

        sample = examples[:n]
        print(_hdr(f"── Remote validation: {len(sample)} example(s) on {llm_model} ──"))

        all_ok = True
        for i, example in enumerate(sample):
            print(_info(f"\n  Example {i} — {json.dumps(example)[:120]}"))
            try:
                final = self.stream_rollout(
                    raw_example=example,
                    env_cls_path=env_cls_path,
                    env_metadata_path=env_metadata_path,
                    env_cls_bytes=env_cls_bytes,
                    env_metadata_bytes=env_metadata_bytes,
                    example_index=i,
                    llm_model=llm_model,
                    max_turns=max_turns,
                )
                if not final.get("success"):
                    all_ok = False
            except RuntimeError as exc:
                print(_err(f"  Example {i} failed: {exc}"))
                all_ok = False

        print()
        if all_ok:
            print(_ok("Remote validation passed"))
        else:
            print(_err("Remote validation failed — check output above before launching a full job"))

        return all_ok
