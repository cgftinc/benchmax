"""AIME's prepared Harbor image exposed through Cloudflare Sandbox bridge."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from cloudflare_transport import CloudflareSandbox
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

AIME_IMAGE = "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
EXPECTED_DOCKERFILE = (
    f"FROM {AIME_IMAGE}",
    "WORKDIR /app",
    "ENV TEST_DIR=/tests",
)


def validate_aime_environment(environment_dir: Path | str) -> None:
    """Fail closed unless a task matches the prepared AIME image exactly."""

    root = Path(environment_dir)
    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"AIME Dockerfile not found: {dockerfile}")
    instructions = tuple(
        line.strip()
        for line in dockerfile.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if instructions != EXPECTED_DOCKERFILE:
        raise ValueError(
            "Cloudflare AIME adapter only supports the verified fixed Dockerfile; "
            f"received {instructions!r}"
        )
    unexpected = sorted(child.name for child in root.iterdir() if child.name != "Dockerfile")
    if unexpected:
        raise ValueError(
            "Cloudflare AIME adapter only supports a Dockerfile-only build context; "
            f"received {unexpected!r}"
        )


class AimeCloudflareEnvironment(BaseEnvironment):
    """Run AIME's fixed image in a prepared Cloudflare Standard-2 container."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._api_url = os.environ.get("CLOUDFLARE_SANDBOX_API_URL", "").rstrip("/")
        self._api_key = os.environ.get("CLOUDFLARE_SANDBOX_API_KEY", "")
        self._sandbox: CloudflareSandbox | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "cloudflare"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    @classmethod
    def preflight(cls) -> None:
        missing = [
            name
            for name in (
                "CLOUDFLARE_SANDBOX_API_URL",
                "CLOUDFLARE_SANDBOX_API_KEY",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise SystemExit(f"missing Cloudflare Sandbox setting(s): {', '.join(missing)}")

    def _validate_definition(self) -> None:
        if not self._api_url or not self._api_key:
            raise ValueError(
                "Cloudflare Sandbox requires CLOUDFLARE_SANDBOX_API_URL and "
                "CLOUDFLARE_SANDBOX_API_KEY"
            )
        validate_aime_environment(self.environment_dir)

    def _inner(self) -> CloudflareSandbox:
        if self._sandbox is None:
            raise RuntimeError("Cloudflare sandbox has not been started")
        return self._sandbox

    async def start(self, force_build: bool) -> None:
        if force_build:
            self.logger.info(
                "force_build ignored: the Cloudflare Worker owns the prepared AIME image"
            )
        self._sandbox = CloudflareSandbox(self._api_url, self._api_key)
        try:
            await self._sandbox.start()
        except BaseException:
            self._sandbox = None
            raise

    async def stop(self, delete: bool) -> None:
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            await sandbox.close(delete=delete)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        merged_env = self._merge_env(env) or {}
        actual_cwd = cwd or self.task_env_config.workdir or "/app"
        resolved_user = self._resolve_user(user)
        if resolved_user not in (None, "root", 0, "0"):
            exports = [
                f"export {name}={shlex.quote(str(value))}" for name, value in merged_env.items()
            ]
            script = " && ".join([*exports, f"cd {shlex.quote(actual_cwd)}", command])
            command = (
                f"exec su -s /bin/bash {shlex.quote(str(resolved_user))} -c {shlex.quote(script)}"
            )
            actual_cwd = "/app"
            merged_env = {}
        result = await self._inner().run_command(
            command,
            workdir=actual_cwd,
            env=merged_env,
            timeout=timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.exit_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._inner().upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._inner().upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._inner().download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._inner().download_dir(source_dir, target_dir)
