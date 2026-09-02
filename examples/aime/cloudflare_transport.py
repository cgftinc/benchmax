"""HTTP client for the self-deployed Cloudflare Sandbox bridge."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

TRANSFER_ROOT = "/workspace/.harbor-transfer"
MAX_DIRECT_WRITE = 31 * 1024 * 1024
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TRANSIENT_START_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_START_RETRY_WAIT_SECONDS = 60.0
MAX_START_RETRY_DELAY_SECONDS = 10.0

logger = logging.getLogger(__name__)


class CloudflareHTTPError(RuntimeError):
    """An HTTP failure from the authenticated Sandbox bridge."""

    def __init__(self, operation: str, status_code: int, detail: str) -> None:
        super().__init__(f"Cloudflare {operation} failed (HTTP {status_code}): {detail[:500]}")
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def return_code(self) -> int:
        """Harbor spells the same field differently."""
        return self.exit_code


class CloudflareSandbox:
    """One isolated sandbox behind Cloudflare's authenticated HTTP bridge."""

    provider = "cloudflare"

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        max_stream_output_bytes: int | None = None,
    ) -> None:
        normalized_url = api_url.strip().rstrip("/")
        if not normalized_url.startswith(("https://", "http://")):
            raise ValueError("Cloudflare Sandbox api_url must be an HTTP(S) URL")
        if not api_key:
            raise ValueError("Cloudflare Sandbox api_key must be non-empty")
        if max_stream_output_bytes is not None and max_stream_output_bytes < 1:
            raise ValueError("max_stream_output_bytes must be positive")
        self.api_url = normalized_url
        self._api_key = api_key
        self.max_stream_output_bytes = max_stream_output_bytes
        self._sandbox_id: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def sandbox_id(self) -> str:
        if self._sandbox_id is None:
            raise RuntimeError("Cloudflare sandbox has not been started")
        return self._sandbox_id

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(connect=30, read=None, write=120, pool=30),
            )
        return self._client

    def _sandbox_path(self, suffix: str = "") -> str:
        return f"/v1/sandbox/{self.sandbox_id}{suffix}"

    async def start(self) -> None:
        """Create one sandbox from the bridge Worker's prepared image."""
        if self._sandbox_id is not None:
            raise RuntimeError("Cloudflare sandbox is already started")
        response = await self._http().post("/v1/sandbox")
        self._raise_for_status(response, "create sandbox")
        sandbox_id = response.json().get("id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise RuntimeError("Cloudflare bridge returned no sandbox ID")
        self._sandbox_id = sandbox_id
        try:
            setup = await self._run_initial_setup()
            if setup.exit_code != 0:
                raise RuntimeError(f"Cloudflare sandbox setup failed: {setup.stderr}")
        except BaseException:
            try:
                await self.cleanup()
            except Exception:
                # Preserve the trial's real failure. A best-effort DELETE after a
                # failed allocation must not replace it with a cleanup exception.
                logger.warning("Cloudflare sandbox cleanup failed", exc_info=True)
            raise

    async def _run_initial_setup(self) -> CommandResult:
        """Allocate the container with Modal-style bounded throttle backoff.

        The bridge allocates on the first exec rather than on POST /sandbox. This
        command is intentionally idempotent, so it is safe to replay against the
        same sandbox ID. Arbitrary agent commands are never retried.
        """
        command = (
            "mkdir -p /workspace /app /logs/agent /logs/verifier /logs/artifacts "
            "/tests /solution /harbor/skills "
            f"{shlex.quote(TRANSFER_ROOT)} && "
            "chmod 777 /workspace /app /logs /logs/agent /logs/verifier "
            "/logs/artifacts /tests /solution /harbor /harbor/skills"
        )
        delay = 1.0
        waited = 0.0
        while True:
            try:
                return await self.run_command(command, timeout=120)
            except CloudflareHTTPError as error:
                remaining = MAX_START_RETRY_WAIT_SECONDS - waited
                if error.status_code not in TRANSIENT_START_STATUS_CODES or remaining <= 0:
                    raise
                sleep_for = min(remaining, random.uniform(0.5, 1.5) * delay)
                logger.warning(
                    "Cloudflare sandbox allocation returned HTTP %s; retrying in %.1fs",
                    error.status_code,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                waited += sleep_for
                delay = min(delay * 2, MAX_START_RETRY_DELAY_SECONDS)

    async def cleanup(self) -> None:
        """Delete this sandbox and close its HTTP transport; safe to call twice."""
        await self.close(delete=True)

    async def close(self, *, delete: bool) -> None:
        """Close the client, optionally deleting the remote sandbox first."""
        try:
            if delete and self._sandbox_id is not None:
                response = await self._http().delete(self._sandbox_path())
                if response.status_code not in (204, 404):
                    self._raise_for_status(response, "destroy sandbox")
        finally:
            self._sandbox_id = None
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def run_command(
        self,
        command: str,
        *,
        workdir: str = "/workspace",
        timeout: float | None = None,
        max_output_bytes: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        """Execute a shell command and parse the bridge's SSE result stream."""
        exports: list[str] = []
        for name, value in (env or {}).items():
            if not ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            exports.append(f"export {name}={shlex.quote(str(value))}")
        script = " && ".join([*exports, f"cd {shlex.quote(workdir)}", command])
        timeout_ms = None if timeout is None else max(1, int(timeout * 1000))
        payload: dict[str, object] = {
            "argv": ["bash", "-lc", script],
            "cwd": "/workspace",
        }
        if timeout_ms is not None:
            payload["timeout_ms"] = timeout_ms

        limit = max_output_bytes if max_output_bytes is not None else self.max_stream_output_bytes
        stdout: list[str] = []
        stderr: list[str] = []
        stdout_size = 0
        stderr_size = 0
        exit_code: int | None = None
        request_timeout = None if timeout is None else timeout + 30

        def consume(event: str, data: str) -> None:
            nonlocal exit_code, stdout_size, stderr_size
            if event in {"stdout", "stderr"}:
                decoded = base64.b64decode(data).decode("utf-8", errors="replace")
                target = stdout if event == "stdout" else stderr
                size = stdout_size if event == "stdout" else stderr_size
                if limit is not None:
                    remaining = max(0, limit - size)
                    decoded = decoded[:remaining]
                target.append(decoded)
                if event == "stdout":
                    stdout_size += len(decoded)
                else:
                    stderr_size += len(decoded)
            elif event == "exit":
                exit_code = int(json.loads(data)["exit_code"])
            elif event == "error":
                try:
                    detail = json.loads(data).get("error", data)
                except json.JSONDecodeError:
                    detail = data
                raise RuntimeError(f"Cloudflare sandbox exec failed: {detail}")

        async with self._http().stream(
            "POST",
            self._sandbox_path("/exec"),
            json=payload,
            timeout=request_timeout,
        ) as response:
            if response.is_error:
                # Streaming responses are not readable through response.text until
                # explicitly consumed. Read the bridge's pool error before raising.
                await response.aread()
            self._raise_for_status(response, "execute command")
            event: str | None = None
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if event is not None:
                        consume(event, "\n".join(data_lines))
                    event = None
                    data_lines = []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if event is not None:
                consume(event, "\n".join(data_lines))

        if exit_code is None:
            raise RuntimeError("Cloudflare sandbox exec stream ended without an exit code")
        return CommandResult(stdout="".join(stdout), stderr="".join(stderr), exit_code=exit_code)

    async def write_file(
        self,
        target_path: str,
        content: bytes | str,
        *,
        executable: bool = False,
    ) -> CommandResult:
        payload = content.encode() if isinstance(content, str) else content
        mode = 0o755 if executable else 0o644
        await self._write_bytes(payload, target_path, mode=mode)
        return CommandResult(stdout="", stderr="", exit_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        await self._write_bytes(
            source.read_bytes(), target_path, mode=source.stat().st_mode & 0o777
        )

    async def _write_bytes(self, content: bytes, target_path: str, *, mode: int) -> None:
        target = str(PurePosixPath(target_path))
        transfer_id = uuid.uuid4().hex
        direct = target == "/workspace" or target.startswith("/workspace/")
        staged_target = target if direct else f"{TRANSFER_ROOT}/{transfer_id}.file"
        await self._upload_bytes(content, staged_target)
        command = f"chmod {mode:o} {shlex.quote(staged_target)}"
        if not direct:
            command += (
                f" && mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}"
                f" && mv {shlex.quote(staged_target)} {shlex.quote(target)}"
            )
        result = await self.run_command(command, timeout=120)
        if result.exit_code != 0:
            raise RuntimeError(f"failed to finalize upload to {target}: {result.stderr}")

    async def _upload_bytes(self, content: bytes, target: str) -> None:
        parent = str(PurePosixPath(target).parent)
        mkdir = await self.run_command(f"mkdir -p {shlex.quote(parent)}", timeout=120)
        if mkdir.exit_code != 0:
            raise RuntimeError(f"failed to create upload directory: {mkdir.stderr}")
        if len(content) <= MAX_DIRECT_WRITE:
            response = await self._http().put(
                self._file_path(target),
                content=content,
                headers={"Content-Type": "application/octet-stream"},
            )
            self._raise_for_status(response, f"upload {target}")
            return

        parts_dir = f"{TRANSFER_ROOT}/{uuid.uuid4().hex}.parts"
        make_parts = await self.run_command(f"mkdir -p {shlex.quote(parts_dir)}", timeout=120)
        if make_parts.exit_code != 0:
            raise RuntimeError(f"failed to create upload parts directory: {make_parts.stderr}")
        part_paths: list[str] = []
        for index, offset in enumerate(range(0, len(content), MAX_DIRECT_WRITE)):
            part = f"{parts_dir}/{index:06d}"
            response = await self._http().put(
                self._file_path(part),
                content=content[offset : offset + MAX_DIRECT_WRITE],
                headers={"Content-Type": "application/octet-stream"},
            )
            self._raise_for_status(response, f"upload part {index}")
            part_paths.append(part)
        command = (
            f"cat {' '.join(shlex.quote(path) for path in part_paths)} > "
            f"{shlex.quote(target)} && rm -rf {shlex.quote(parts_dir)}"
        )
        result = await self.run_command(command, timeout=300)
        if result.exit_code != 0:
            raise RuntimeError(f"failed to join upload parts: {result.stderr}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(source)
        remote_archive = f"{TRANSFER_ROOT}/{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory(prefix="cloudflare-upload-") as tmp:
            archive = Path(tmp) / "upload.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for child in source.iterdir():
                    tar.add(child, arcname=child.name)
            await self.upload_file(archive, remote_archive)
        result = await self.run_command(
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar xzf {shlex.quote(remote_archive)} --no-same-owner "
            f"-C {shlex.quote(target_dir)} && rm -f {shlex.quote(remote_archive)}",
            timeout=300,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"failed to unpack directory upload: {result.stderr}")

    async def read_file(self, source_path: str, *, max_bytes: int | None = None) -> CommandResult:
        try:
            content = await self._download_bytes(source_path)
        except FileNotFoundError as error:
            return CommandResult(stdout="", stderr=str(error), exit_code=1)
        if max_bytes is not None:
            content = content[:max_bytes]
        return CommandResult(
            stdout=content.decode("utf-8", errors="replace"),
            stderr="",
            exit_code=0,
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        content = await self._download_bytes(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def _download_bytes(self, source_path: str) -> bytes:
        source = str(PurePosixPath(source_path))
        # Cloudflare Sandbox 0.12.x returns HTTP 200 with an empty body when a
        # file does not exist. Check inside the container first so Harbor does
        # not record a missing required artifact as a successful zero-byte
        # download.
        exists = await self.run_command(f"test -f {shlex.quote(source)}", timeout=120)
        if exists.exit_code != 0:
            raise FileNotFoundError(source)
        staged = source
        cleanup = False
        if not (source == "/workspace" or source.startswith("/workspace/")):
            staged = f"{TRANSFER_ROOT}/{uuid.uuid4().hex}.download"
            result = await self.run_command(
                f"cp {shlex.quote(source)} {shlex.quote(staged)}", timeout=120
            )
            if result.exit_code != 0:
                raise FileNotFoundError(f"failed to stage download of {source}: {result.stderr}")
            cleanup = True
        try:
            response = await self._http().get(self._file_path(staged))
            if response.status_code == 404:
                raise FileNotFoundError(source)
            self._raise_for_status(response, f"download {source}")
            return response.content
        finally:
            if cleanup:
                await self.run_command(f"rm -f {shlex.quote(staged)}", timeout=120)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        remote_archive = f"{TRANSFER_ROOT}/{uuid.uuid4().hex}.tar.gz"
        result = await self.run_command(
            f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source_dir)} .",
            timeout=300,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"failed to pack directory download: {result.stderr}")
        with tempfile.TemporaryDirectory(prefix="cloudflare-download-") as tmp:
            archive = Path(tmp) / "download.tar.gz"
            try:
                await self.download_file(remote_archive, archive)
                target = Path(target_dir)
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(target, filter="data")
            finally:
                await self.run_command(f"rm -f {shlex.quote(remote_archive)}", timeout=120)

    def _file_path(self, path: str) -> str:
        encoded = quote(path.lstrip("/"), safe="/")
        return self._sandbox_path(f"/file/{encoded}")

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if not response.is_error:
            return
        raise CloudflareHTTPError(operation, response.status_code, response.text)
