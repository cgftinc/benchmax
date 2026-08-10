"""Trainer API clients for storage uploads and job management."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    # Runtime import stays inside launch_sft_run: sft.py imports this module
    # for StorageClient, so a top-level import would be circular.
    from .sft import SftTrainingConfig, UploadedSftAssets

from castform import config

from .credentials import TokenProvider, resolve_token_provider
from .exceptions import (
    AuthenticationError,
    ClientUnsupportedError,
    JobLaunchError,
    RolloutError,
    RolloutNotFound,
    RolloutServerError,
    TrainerError,
)

logger = logging.getLogger(__name__)

# 410 Gone / 501 Not Implemented: the route was retired server-side. Checked
# before the generic 5xx branch, which would otherwise type 501 as retryable.
_RETIRED_ENDPOINT_STATUSES = (410, 501)


def _raise_if_client_unsupported(message: str, status_code: int) -> None:
    """Raise :class:`ClientUnsupportedError` when the server retired this route.

    Every status-code classifier in this module calls this first, so a retired
    endpoint reads the same regardless of which client hit it.

    Args:
        message:     Server-supplied error text, already extracted or truncated.
        status_code: HTTP status of the response.
    """
    if status_code in _RETIRED_ENDPOINT_STATUSES:
        raise ClientUnsupportedError(
            f"{message}\n"
            "This endpoint no longer exists on the server — your castform "
            "client is out of date. Upgrade with "
            "`pip install --upgrade castform`, then retry.",
            status_code,
        )


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


def _rollout_list(payload: Any) -> list[dict[str, Any]]:
    """Normalize a rollout-list response to a list. The /rollouts/{summary,heatmap}
    routes return either a bare list or an envelope under one of a few keys
    (the web UI and the view-progress recipe both hedge on this)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "rollouts", "groups"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


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
    the credential seam (``ACT_AS_TOKEN_PATH`` / ``CASTFORM_API_KEY``). Pass
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

        _raise_if_client_unsupported(message, response.status_code)
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
    the credential seam (``ACT_AS_TOKEN_PATH`` / ``CASTFORM_API_KEY``). Pass
    ``api_key`` to override, or ``token_provider`` for a custom per-call source.

    Example:
        client = TrainerClient(api_key="sk_...", base_url="http://localhost:3000")
        run_id = client.launch_training_run(
            env_cls_path="envs/run-abc/abc123/env-cls.pkl",
            env_metadata_path="envs/run-abc/abc123/env-metadata.json",
            dataset_path="datasets/run-abc/def456",
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

        _raise_if_client_unsupported(message, response.status_code)
        raise JobLaunchError(message, response.status_code)

    def launch_training_run(
        self,
        env_cls_path: str,
        env_metadata_path: str,
        dataset_path: str | None = None,
        name: str | None = None,
        launcher_args: dict[str, Any] | None = None,
        extra_secrets: Mapping[str, str] | None = None,
    ) -> str:
        """Launch a new training run.

        Args:
            env_cls_path: Path to the environment class pickle (.pkl file)
            env_metadata_path: Path to the environment metadata JSON file
            dataset_path: Optional blob prefix holding the run's dataset
                files. The trainer mirrors the whole prefix to the machine and
                exposes it to the environment as its dataset base directory.
                Omit it when the environment resolves data at runtime.
            name: Optional name for the training run
            launcher_args: Extra launcher args forwarded to the server
                (e.g. {"max_context_tokens": 4000}). Bundle and optional dataset
                path parameters always take precedence over this mapping.
            extra_secrets: Runtime-only environment secrets forwarded over the
                authenticated launch request. They are not stored in the
                environment bundle or launcher args. The server rejects reserved
                platform-owned names.
        Returns:
            The training run ID.

        Raises:
            AuthenticationError: If API key is invalid
            JobLaunchError: If training run launch fails
        """
        reserved_paths = {
            "env_cls_path",
            "env_metadata_path",
            "dataset_path",
        }
        args: dict[str, Any] = {
            key: value for key, value in (launcher_args or {}).items() if key not in reserved_paths
        }
        args.update(
            {
                "env_cls_path": env_cls_path,
                "env_metadata_path": env_metadata_path,
            }
        )
        if dataset_path is not None:
            args["dataset_path"] = dataset_path
        body: dict[str, Any] = {
            "name": name,
            "args": args,
        }
        if extra_secrets:
            body["extraSecrets"] = dict(extra_secrets)
        response = self._http_client.post(
            "/v1/train/runs/launch",
            json=body,
        )
        self._handle_response_errors(response)
        body = response.json()
        # Surface soft-cap / OOM-risk warnings via the warnings module (shown by
        # default in notebooks/REPL) — a bare logger.warning is swallowed unless
        # the caller configured logging.
        for warning in body.get("warnings", []) or []:
            warnings.warn(f"launch warning: {warning}", stacklevel=2)
        return body["runId"]

    def launch_sft_run(
        self,
        *,
        assets: UploadedSftAssets,
        name: str,
        config: SftTrainingConfig | None = None,
    ) -> str:
        """Launch an env-less SFT run through the dedicated SFT endpoint.

        Only the typed uploaded reference and the genuine v1 choices cross the
        wire; the platform owns the model, LoRA policy, topology, and
        scheduling. No environment paths, secrets, or template selectors are
        accepted or synthesized.

        Args:
            assets: Result of ``upload_sft_assets`` — the dataset prefix plus
                its literal format identifier.
            name: Run name (same safe-name limits as RL runs).
            config: The v1 training choices; defaults to
                ``SftTrainingConfig()``.

        Returns:
            The training run ID.

        Raises:
            AuthenticationError: If the bearer is invalid.
            JobLaunchError: If the platform rejects or fails the launch,
                including when SFT launch is disabled.
        """

        from .sft import SftTrainingConfig, UploadedSftAssets

        if not isinstance(assets, UploadedSftAssets):
            raise TypeError(
                f"assets must be an UploadedSftAssets from upload_sft_assets, "
                f"got {type(assets).__name__}"
            )
        resolved = config if config is not None else SftTrainingConfig()
        if not isinstance(resolved, SftTrainingConfig):
            raise TypeError(f"config must be an SftTrainingConfig, got {type(config).__name__}")
        response = self._http_client.post(
            "/v1/train/runs/sft",
            json={
                "name": name,
                "dataset": {
                    "format": assets.dataset_format,
                    "path": assets.dataset_path,
                },
                "args": resolved.as_args(),
            },
        )
        self._handle_response_errors(response)
        return response.json()["runId"]

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
            req = _RED + "required" + _RESET if spec.required else _CYAN + "optional" + _RESET
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

    def get_run_scalars(self, run_id: str, mode: str) -> dict[str, list[dict[str, Any]]]:
        """GET /v1/train/runs/{id}/scalars?mode= — {scalarName: [{step, value}]}.

        ``mode`` is required by the server (400 if omitted). Discover the valid
        set from :meth:`get_run_details` — ``modes`` is dynamic per run, not a
        fixed enum. An unknown mode returns ``{}`` (not an error).
        """
        response = self._http_client.get(f"/v1/train/runs/{run_id}/scalars", params={"mode": mode})
        self._handle_response_errors(response)
        return response.json()

    def get_environment_logs(
        self, run_id: str, *, rollout_id: str | None = None
    ) -> list[dict[str, Any]]:
        """GET /v1/train/runs/{id}/environment-logs — run-level logs, or one
        rollout's when ``rollout_id`` is given."""
        params = {"rolloutId": rollout_id} if rollout_id else None
        response = self._http_client.get(f"/v1/train/runs/{run_id}/environment-logs", params=params)
        self._handle_response_errors(response)
        return response.json().get("logs", [])

    # --- Stored-rollout reads (CLI: `castform runs rollouts/rollout ...`) ---
    # The rich per-rollout data the web UI shows — answers + per-component rewards
    # — lives behind /rollouts/*; these wrap the reads the CLI/SDK had to hand-roll
    # with raw httpx before. optionalAuth, like the other run reads. ``mode`` is
    # ``train`` | ``eval`` | ``external-eval`` (pass ``external_eval_id`` for the last).

    def get_rollout_summary(
        self,
        run_id: str,
        *,
        mode: str = "eval",
        page: int = 1,
        limit: int = 50,
        external_eval_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET .../rollouts/summary — one row per example: ``promptMessageId``,
        ``promptText`` (truncated server-side), ``rewardHistory[{step, meanReward}]``."""
        params: dict[str, Any] = {"mode": mode, "page": page, "limit": limit}
        if external_eval_id:
            params["externalEvalId"] = external_eval_id
        response = self._http_client.get(f"/v1/train/runs/{run_id}/rollouts/summary", params=params)
        self._handle_response_errors(response)
        return _rollout_list(response.json())

    def get_rollout_heatmap(
        self,
        run_id: str,
        prompt_message_id: str,
        *,
        mode: str = "eval",
        external_eval_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET .../rollouts/heatmap — every rollout for one example across steps:
        ``[{id, step, totalReward}]`` (eval re-runs the same prompt per checkpoint)."""
        params: dict[str, Any] = {"mode": mode, "promptMessageId": prompt_message_id}
        if external_eval_id:
            params["externalEvalId"] = external_eval_id
        response = self._http_client.get(f"/v1/train/runs/{run_id}/rollouts/heatmap", params=params)
        self._handle_response_errors(response)
        return _rollout_list(response.json())

    def get_rollout_details(self, run_id: str, rollout_id: str) -> dict[str, Any]:
        """GET .../rollouts/{rolloutId}/details — ``{promptMessages, messages,
        rewards[{name,value}], totalReward, step}``. Gold/ground_truth is NOT in the
        payload — join the local dataset by prompt text."""
        response = self._http_client.get(f"/v1/train/runs/{run_id}/rollouts/{rollout_id}/details")
        self._handle_response_errors(response)
        return response.json()

    def get_rollout_mode_average(
        self, run_id: str, *, mode: str = "eval", external_eval_id: str | None = None
    ) -> dict[str, Any]:
        """GET .../rollouts/mode-average — the headline ``{avg, ...}`` for a mode."""
        params: dict[str, Any] = {"mode": mode}
        if external_eval_id:
            params["externalEvalId"] = external_eval_id
        response = self._http_client.get(
            f"/v1/train/runs/{run_id}/rollouts/mode-average", params=params
        )
        self._handle_response_errors(response)
        return response.json()

    def get_rollout_component_averages(self, run_id: str, *, mode: str = "eval") -> dict[str, Any]:
        """GET .../rollouts/component-averages — latest-step per-component means
        (can be all-null; for a trajectory use ``get_run_scalars`` instead)."""
        response = self._http_client.get(
            f"/v1/train/runs/{run_id}/rollouts/component-averages",
            params={"mode": mode},
        )
        self._handle_response_errors(response)
        return response.json()

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
_DEFAULT_ROLLOUT_MODEL = "gpt-5.4-nano"

# ANSI colours
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


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


class RolloutClient:
    """Thin synchronous client for group-native rollout batches.

    Rollouts are reached through platform-service. platform-service is the API-key
    gate: it validates the ``sk_`` key and mints a short-lived act_as JWT that
    rollout-service accepts (rollout-service's own auth only takes
    auth-service-minted JWTs — never a raw platform key). The proxy is mounted at
    ``/v1/rollout/batch/stream``.

    Environment artifacts use the same already-uploaded ``env_cls_path`` and
    ``env_metadata_path`` as trainer launch.

    Args:
        api_key:    Platform API key forwarded as the Bearer token
                    platform-service validates. Optional — when omitted the
                    bearer resolves per request via the credential seam
                    (``ACT_AS_TOKEN_PATH`` / ``CASTFORM_API_KEY``).
        server_url: Base URL of platform-service. Defaults to
                    ``config.platform_url()``; the group batch path is appended
                    per request.
        timeout:    Per-request timeout in seconds (default 300 — rollouts can be slow).
        token_provider: Custom per-call bearer source; overrides the seam when
                    ``api_key`` is unset.
    """

    def __init__(
        self,
        api_key: str | None = None,
        server_url: str | None = None,
        timeout: float = 300.0,
        *,
        token_provider: TokenProvider | None = None,
    ) -> None:
        # Bearer resolves per request: explicit api_key →
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
        env_cls_path: str,
        env_metadata_path: str,
    ) -> dict[str, str]:
        return {
            "env_cls_path": env_cls_path,
            "env_metadata_path": env_metadata_path,
        }

    def run_group(
        self,
        *,
        samples: int,
        env_cls_path: str,
        env_metadata_path: str,
        dataset_path: str | None = None,
        split: str = "eval",
        model: str = _DEFAULT_ROLLOUT_MODEL,
        max_context_tokens: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the first environment-created Dataset item as one sibling group."""
        env = self._build_env(env_cls_path, env_metadata_path)
        bearer = self._token_provider()

        payload: dict[str, Any] = {
            "env": env,
            "model": {"name": model},
            "split": split,
            "max_examples": 1,
            "group_size": samples,
            "max_in_flight": samples,
        }
        if max_context_tokens is not None:
            payload["max_context_tokens"] = max_context_tokens
        if dataset_path is not None:
            payload["dataset_path"] = dataset_path

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
                _raise_if_client_unsupported(body[:300], response.status_code)
                if 500 <= response.status_code < 600:
                    raise RolloutServerError(body[:300], response.status_code)
                raise RolloutError(body[:300], response.status_code)

            for event in _iter_sse(response):
                etype = event.get("event")
                if etype == "batch_started":
                    if verbose:
                        print(_info(f"  group batch started ({event.get('total')} rollouts)"))
                elif etype == "rollout_completed":
                    completed.append(event)
                elif etype == "worker_error":
                    raise RolloutError(
                        str(event.get("error") or "hosted rollout worker failed")[:300],
                        500,
                    )
                elif etype == "error":
                    raise RolloutError(str(event.get("error"))[:300], 500)
                elif etype == "cancelled":
                    raise RolloutError(
                        str(event.get("error") or "hosted rollout was cancelled")[:300],
                        500,
                    )
                elif etype == "batch_completed":
                    break
        return completed
