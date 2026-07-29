from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

DEFAULT_CONTAINER_HARVEY_ROOT = "/workspace/archive/harvey-labs"
DEFAULT_CONTAINER_RUNTIME_PATH = "/workspace/archive/harvey_runtime.py"
DEFAULT_PREBAKED_VENV_PYTHON = "/opt/harvey-labs-venv/bin/python"
DEFAULT_HARBOR_DOCUMENTS_DIR = "/workspace/documents"
DEFAULT_HARBOR_OUTPUT_DIR = "/workspace/output"
DEFAULT_MAX_TOOL_RESULT_CHARS = 12000
DEFAULT_HARVEY_REPOSITORY = "https://github.com/harveyai/harvey-labs.git"
DEFAULT_HARVEY_GIT_REF = "845a08840869b21a5c11958aae58bf5f00a7b775"
HARVEY_SPARSE_PATHS = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    "harness",
    "sandbox",
    "evaluation",
    "utils",
    "docs",
)
IGNORED_UPLOAD_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "results",
}
_source_lock = threading.Lock()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").replace(".", "-")


def bundled_harvey_root() -> Path | None:
    """The LAB source tree materialized next to this file from a bundle, if any."""

    root = Path(__file__).with_name("harvey-labs")
    return root if (root / "harness" / "run.py").is_file() else None


def fetch_harvey_source(
    *,
    repository: str | None = None,
    git_ref: str | None = None,
) -> Path:
    """Clone Harvey's sparse harness source once per host and return its root."""

    repository = repository or os.environ.get("HARBOR_HARVEY_GIT_URL") or DEFAULT_HARVEY_REPOSITORY
    git_ref = git_ref or os.environ.get("HARBOR_HARVEY_GIT_REF") or DEFAULT_HARVEY_GIT_REF
    cache = Path(tempfile.gettempdir()) / f"castform-harvey-labs-{_slug(git_ref) or 'main'}"
    with _source_lock:
        if (cache / "harness" / "run.py").is_file():
            return cache
        if cache.exists():
            shutil.rmtree(cache)
        staging = Path(tempfile.mkdtemp(prefix=f"{cache.name}.tmp-"))
        try:
            clone = subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    "--depth",
                    "1",
                    repository,
                    str(staging),
                ],
                capture_output=True,
                text=True,
            )
            if clone.returncode != 0:
                raise RuntimeError(
                    "failed to clone Harvey LAB harness "
                    f"from {repository}@{git_ref}: {clone.stderr.strip()}"
                )
            fetch = subprocess.run(
                [
                    "git",
                    "-C",
                    str(staging),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    git_ref,
                ],
                capture_output=True,
                text=True,
            )
            if fetch.returncode != 0:
                raise RuntimeError(
                    f"failed to fetch Harvey LAB ref {git_ref}: {fetch.stderr.strip()}"
                )
            sparse = subprocess.run(
                [
                    "git",
                    "-C",
                    str(staging),
                    "sparse-checkout",
                    "set",
                    "--no-cone",
                    *HARVEY_SPARSE_PATHS,
                ],
                capture_output=True,
                text=True,
            )
            if sparse.returncode != 0:
                raise RuntimeError(
                    f"failed to configure the Harvey LAB sparse checkout: {sparse.stderr.strip()}"
                )
            checkout = subprocess.run(
                [
                    "git",
                    "-C",
                    str(staging),
                    "checkout",
                    "--detach",
                    "FETCH_HEAD",
                ],
                capture_output=True,
                text=True,
            )
            if checkout.returncode != 0:
                raise RuntimeError(
                    f"failed to check out Harvey LAB ref {git_ref}: {checkout.stderr.strip()}"
                )
            staging.rename(cache)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return cache


class HarveyHarnessAgent(BaseAgent):
    """Run Harvey LAB's harness loop as a Harbor agent.

    Harbor owns the environment lifecycle. This adapter locates and uploads the
    Harvey harness code, runs it inside the environment, and copies its run
    artifacts into Harbor's mounted logs and artifacts directories.
    """

    def __init__(
        self,
        *args: Any,
        extra_env: dict[str, str] | None = None,
        harvey_root: str | None = None,
        runtime_path: str | None = None,
        container_harvey_root: str | None = None,
        container_runtime_path: str | None = None,
        max_turns: int | None = None,
        max_tool_result_chars: int | None = None,
        shell_timeout: int | None = None,
        run_timeout_sec: int | None = None,
        upload: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self.host_harvey_root = self._optional_path(harvey_root or self._env("HARBOR_HARVEY_ROOT"))
        self.host_runtime_path = self._optional_path(
            runtime_path or self._env("HARBOR_HARVEY_RUNTIME_PATH")
        )
        self.container_harvey_root = (
            container_harvey_root
            or self._env("HARBOR_HARVEY_CONTAINER_ROOT")
            or DEFAULT_CONTAINER_HARVEY_ROOT
        )
        self.container_runtime_path = (
            container_runtime_path
            or self._env("HARBOR_HARVEY_CONTAINER_RUNTIME_PATH")
            or DEFAULT_CONTAINER_RUNTIME_PATH
        )
        self.harbor_documents_dir = (
            self._env("HARBOR_HARVEY_DOCUMENTS_DIR") or DEFAULT_HARBOR_DOCUMENTS_DIR
        )
        self.harbor_output_dir = self._env("HARBOR_HARVEY_OUTPUT_DIR") or DEFAULT_HARBOR_OUTPUT_DIR
        self.max_turns = max_turns or self._env_int("HARBOR_HARVEY_MAX_TURNS", 30)
        self.max_tool_result_chars = (
            self._env_int(
                "HARBOR_HARVEY_MAX_TOOL_RESULT_CHARS",
                DEFAULT_MAX_TOOL_RESULT_CHARS,
            )
            if max_tool_result_chars is None
            else max_tool_result_chars
        )
        self.shell_timeout = shell_timeout or self._env_int("HARBOR_HARVEY_SHELL_TIMEOUT", 60)
        self.run_timeout_sec = run_timeout_sec or self._env_int(
            "HARBOR_HARVEY_RUN_TIMEOUT_SEC", 3600
        )
        self.upload = self._env_bool("HARBOR_HARVEY_UPLOAD", True) if upload is None else upload

    @staticmethod
    def name() -> str:
        return "harvey-harness"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.ensure_dirs(
            [
                str(Path(self.container_harvey_root).parent),
                str(Path(self.container_runtime_path).parent),
                EnvironmentPaths.agent_dir,
                EnvironmentPaths.artifacts_dir / "harvey-output",
            ]
        )
        if not self.upload:
            return

        await asyncio.to_thread(self._resolve_default_host_paths)
        if self.host_harvey_root and self.host_harvey_root.exists():
            with tempfile.TemporaryDirectory(prefix="harvey-labs-upload-") as tmp:
                bundle_root = Path(tmp) / "harvey-labs"
                shutil.copytree(
                    self.host_harvey_root,
                    bundle_root,
                    ignore=shutil.ignore_patterns(*IGNORED_UPLOAD_NAMES),
                )
                await environment.upload_dir(bundle_root, self.container_harvey_root)

        if self.host_runtime_path and self.host_runtime_path.exists():
            await environment.upload_file(self.host_runtime_path, self.container_runtime_path)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        run_id = self._run_id()
        command = self._run_command(run_id, instruction)
        result = await environment.exec(
            command,
            env=self._execution_env(),
            timeout_sec=self.run_timeout_sec,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Harvey harness failed with exit code "
                f"{result.return_code}\nSTDOUT:\n{result.stdout or ''}\n"
                f"STDERR:\n{result.stderr or ''}"
            )
        await self._populate_context_from_metrics(environment, context, run_id)

    def _resolve_default_host_paths(self) -> None:
        if self.host_harvey_root is None:
            # A bundled LAB tree wins: it is the exact source that was
            # validated, and trial hosts need neither git nor GitHub egress.
            self.host_harvey_root = bundled_harvey_root() or self._prepare_harvey_source()
        if not (self.host_harvey_root / "harness" / "run.py").is_file():
            raise RuntimeError(f"Harvey LAB source is incomplete at {self.host_harvey_root}")
        if self.host_runtime_path is None:
            candidate = Path(__file__).with_name("harvey_runtime.py")
            if candidate.is_file():
                self.host_runtime_path = candidate
        if self.host_runtime_path is None or not self.host_runtime_path.is_file():
            raise RuntimeError("harvey_runtime.py is missing from the bundle")

    def _prepare_harvey_source(self) -> Path:
        """Clone Harvey's sparse harness source once per trainer host."""

        return fetch_harvey_source(
            repository=self._env("HARBOR_HARVEY_GIT_URL"),
            git_ref=self._env("HARBOR_HARVEY_GIT_REF"),
        )

    def _run_command(self, run_id: str, instruction: str) -> str:
        args = [
            "--instruction-file",
            str(EnvironmentPaths.agent_dir / "harvey-instruction.md"),
            "--task-name",
            self._harbor_task_name(),
            "--documents-dir",
            self.harbor_documents_dir,
            "--output-dir",
            self.harbor_output_dir,
            "--model",
            self._model_name_for_harvey(),
            "--base-url",
            self._base_url(),
            "--run-id",
            run_id,
            "--max-turns",
            str(self.max_turns),
            "--max-tool-result-chars",
            str(self.max_tool_result_chars),
            "--shell-timeout",
            str(self.shell_timeout),
        ]
        result_dir = f"{self.container_harvey_root}/results/{run_id}"
        return "\n".join(
            [
                "set -euo pipefail",
                f"HARVEY_ROOT={shlex.quote(self.container_harvey_root)}",
                f"RUNTIME={shlex.quote(self.container_runtime_path)}",
                f"PREBAKED_PYTHON={shlex.quote(self._prebaked_python())}",
                self._write_instruction_block(instruction),
                'if [ -n "${HARBOR_HARVEY_PYTHON:-}" ]; then',
                '  RUNNER=( "${HARBOR_HARVEY_PYTHON}" "${RUNTIME}" )',
                'elif [ -x "${PREBAKED_PYTHON}" ]; then',
                '  RUNNER=( "${PREBAKED_PYTHON}" "${RUNTIME}" )',
                "elif command -v uv >/dev/null 2>&1; then",
                '  RUNNER=( uv run --project "${HARVEY_ROOT}" "${RUNTIME}" )',
                "elif command -v python3 >/dev/null 2>&1; then",
                # Current harveyai/lab images ship neither the prebaked venv
                # nor uv; bootstrap the two runtime deps once per sandbox.
                '  echo "No prebaked Harvey runtime found; bootstrapping venv" >&2',
                "  BOOTSTRAP_VENV=/tmp/harvey-bootstrap-venv",
                '  if [ ! -x "${BOOTSTRAP_VENV}/bin/python" ]; then',
                '    python3 -m venv "${BOOTSTRAP_VENV}"',
                '    "${BOOTSTRAP_VENV}/bin/pip" install --quiet "httpx>=0.27" "openai>=1.60"',
                "  fi",
                '  RUNNER=( "${BOOTSTRAP_VENV}/bin/python" "${RUNTIME}" )',
                "else",
                '  echo "Harvey runtime requires HARBOR_HARVEY_PYTHON, the prebaked venv, uv, or '
                'python3" >&2',
                "  exit 1",
                "fi",
                '"${RUNNER[@]}" ' + " ".join(shlex.quote(arg) for arg in args),
                f"RESULT_DIR={shlex.quote(result_dir)}",
                f"STAGED_RESULT={shlex.quote(str(EnvironmentPaths.agent_dir / 'harvey-results'))}",
                'rm -rf "$STAGED_RESULT"',
                'if [ -d "$RESULT_DIR" ]; then cp -a "$RESULT_DIR" "$STAGED_RESULT"; fi',
                # Harvey creates these aliases as absolute symlinks into the
                # sandbox. They are redundant with Harbor's own workspace
                # capture, and safe tar extraction rejects absolute links.
                'for path in "$STAGED_RESULT/workspace/documents" '
                '"$STAGED_RESULT/workspace/output"; do',
                '  if [ -L "$path" ]; then rm -f "$path"; fi',
                "done",
                f"mkdir -p {shlex.quote(str(EnvironmentPaths.artifacts_dir / 'harvey-output'))}",
                f"if [ -d {shlex.quote(self.harbor_output_dir)} ]; then cp -a "
                f"{shlex.quote(self.harbor_output_dir + '/.')} "
                f"{shlex.quote(str(EnvironmentPaths.artifacts_dir / 'harvey-output/'))}; fi",
            ]
        )

    def _execution_env(self) -> dict[str, str]:
        env = {
            "OPENAI_API_KEY": self._api_key(),
            "OPENAI_BASE_URL": self._base_url(),
        }
        for key, value in {**os.environ, **self.extra_env}.items():
            if key.startswith("HARBOR_HARVEY_"):
                env[key] = value
        return {key: value for key, value in env.items() if value}

    async def _populate_context_from_metrics(
        self,
        environment: BaseEnvironment,
        context: AgentContext,
        run_id: str,
    ) -> None:
        context.metadata = {"harvey_run_id": run_id}
        metrics_path = f"{self.container_harvey_root}/results/{run_id}/metrics.json"
        try:
            result = await environment.exec(
                f"cat {shlex.quote(metrics_path)}",
                timeout_sec=self.shell_timeout,
            )
        except Exception:
            return
        if result.return_code != 0:
            return
        try:
            metrics = json.loads(result.stdout or "")
        except (TypeError, ValueError):
            return
        if not isinstance(metrics, dict):
            return
        context.n_input_tokens = metrics.get("input_tokens")
        context.n_output_tokens = metrics.get("output_tokens")
        context.metadata = {
            "harvey_run_id": run_id,
            "harvey_metrics": metrics,
            # The generic harness contract HarborEnv reads; harvey_metrics
            # stays for detail.
            "termination_reason": metrics.get("termination_reason"),
        }

    def _run_id(self) -> str:
        explicit = self._env("HARBOR_HARVEY_RUN_ID")
        if explicit:
            return explicit
        trial_name = self.logs_dir.parent.name if self.logs_dir.parent.name else "trial"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_name = self._harbor_task_name()
        return (
            f"harbor-agent/{task_name}/{self._slug(self._model_name_for_harvey())}"
            f"/{trial_name}-{timestamp}"
        )

    def _harbor_task_name(self) -> str:
        task_name = self._env("HARBOR_HARVEY_TASK_NAME")
        if task_name:
            return task_name
        trial_name = self.logs_dir.parent.name if self.logs_dir.parent.name else "harbor-task"
        return self._slug(trial_name) or "harbor-task"

    @staticmethod
    def _write_instruction_block(instruction: str) -> str:
        encoded = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
        instruction_path = EnvironmentPaths.agent_dir / "harvey-instruction.md"
        return "\n".join(
            [
                f"INSTRUCTION_FILE={shlex.quote(str(instruction_path))}",
                'mkdir -p "$(dirname "$INSTRUCTION_FILE")"',
                f'printf %s {shlex.quote(encoded)} | base64 -d > "$INSTRUCTION_FILE"',
            ]
        )

    def _model_name_for_harvey(self) -> str:
        model = self._env("HARBOR_HARVEY_MODEL") or self.model_name or ""
        if not model:
            raise RuntimeError("HarveyHarnessAgent requires a model name")
        if self._env_bool("HARBOR_HARVEY_STRIP_OPENAI_PREFIX", True) and model.startswith(
            "openai/"
        ):
            return model.split("/", 1)[1]
        return model

    def _api_key(self) -> str:
        value = self.extra_env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not value:
            raise RuntimeError("HarveyHarnessAgent requires OPENAI_API_KEY")
        return value

    def _base_url(self) -> str:
        value = self.extra_env.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not value:
            raise RuntimeError("HarveyHarnessAgent requires OPENAI_BASE_URL")
        return value.rstrip("/")

    def _prebaked_python(self) -> str:
        return self._env("HARBOR_HARVEY_PREBAKED_PYTHON") or DEFAULT_PREBAKED_VENV_PYTHON

    def _env(self, name: str) -> str | None:
        return self.extra_env.get(name) or os.environ.get(name)

    def _env_int(self, name: str, default: int) -> int:
        raw = self._env(name)
        if raw is None or raw == "":
            return default
        return int(raw)

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = self._env(name)
        if raw is None or raw == "":
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _optional_path(value: str | None) -> Path | None:
        if not value:
            return None
        return Path(value).expanduser().resolve()

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").replace(".", "-")
