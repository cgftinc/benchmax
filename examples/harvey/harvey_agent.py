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
DEFAULT_CONTAINER_PROBE_PATH = "/workspace/archive/harvey_castform_probe.py"
DEFAULT_PREBAKED_VENV_PYTHON = "/opt/harvey-labs-venv/bin/python"
DEFAULT_HARBOR_DOCUMENTS_DIR = "/workspace/documents"
DEFAULT_HARBOR_OUTPUT_DIR = "/workspace/output"
DEFAULT_BOOTSTRAP_PIP_PACKAGES = "anthropic google-genai mistralai openai"
DEFAULT_HARVEY_REPOSITORY = "https://github.com/harveyai/harvey-labs.git"
DEFAULT_HARVEY_GIT_REF = "main"
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


class HarveyHarnessAgent(BaseAgent):
    """Run Harvey LAB's harness loop as a Harbor agent.

    Harbor still owns the environment lifecycle. This adapter only installs or
    locates the Harvey harness code, runs it inside the environment, and copies
    Harvey's run artifacts into Harbor's mounted logs/artifacts directories.
    """

    def __init__(
        self,
        *args: Any,
        extra_env: dict[str, str] | None = None,
        harvey_root: str | None = None,
        probe_path: str | None = None,
        container_harvey_root: str | None = None,
        container_probe_path: str | None = None,
        task: str | None = None,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        shell_timeout: int | None = None,
        run_timeout_sec: int | None = None,
        upload: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self.host_harvey_root = self._optional_path(
            harvey_root or self._env("HARBOR_HARVEY_ROOT")
        )
        self.host_probe_path = self._optional_path(
            probe_path or self._env("HARBOR_HARVEY_PROBE_PATH")
        )
        self.container_harvey_root = (
            container_harvey_root
            or self._env("HARBOR_HARVEY_CONTAINER_ROOT")
            or DEFAULT_CONTAINER_HARVEY_ROOT
        )
        self.container_probe_path = (
            container_probe_path
            or self._env("HARBOR_HARVEY_CONTAINER_PROBE_PATH")
            or DEFAULT_CONTAINER_PROBE_PATH
        )
        self.harvey_task = task or self._env("HARBOR_HARVEY_TASK")
        self.prompt_source = (
            self._env("HARBOR_HARVEY_PROMPT_SOURCE")
            or ("harvey" if self.harvey_task else "harbor")
        ).lower()
        if self.prompt_source not in {"harbor", "harvey"}:
            raise ValueError(
                f"HARBOR_HARVEY_PROMPT_SOURCE must be either 'harbor' or 'harvey', got {self.prompt_source!r}"
            )
        if self.prompt_source == "harvey" and not self.harvey_task:
            raise ValueError(
                "HARBOR_HARVEY_TASK is required when using Harvey prompt source"
            )
        self.harbor_documents_dir = (
            self._env("HARBOR_HARVEY_DOCUMENTS_DIR") or DEFAULT_HARBOR_DOCUMENTS_DIR
        )
        self.harbor_output_dir = (
            self._env("HARBOR_HARVEY_OUTPUT_DIR") or DEFAULT_HARBOR_OUTPUT_DIR
        )
        self.max_turns = max_turns or self._env_int("HARBOR_HARVEY_MAX_TURNS", 30)
        self.max_tokens = max_tokens or self._env_int("HARBOR_HARVEY_MAX_TOKENS", 16384)
        self.max_tool_result_chars = max_tool_result_chars or self._env_int(
            "HARBOR_HARVEY_MAX_TOOL_RESULT_CHARS", 12000
        )
        self.shell_timeout = shell_timeout or self._env_int(
            "HARBOR_HARVEY_SHELL_TIMEOUT", 60
        )
        self.run_timeout_sec = run_timeout_sec or self._env_int(
            "HARBOR_HARVEY_RUN_TIMEOUT_SEC", 3600
        )
        self.upload = (
            self._env_bool("HARBOR_HARVEY_UPLOAD", True) if upload is None else upload
        )

    @staticmethod
    def name() -> str:
        return "harvey-harness"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await asyncio.to_thread(self._resolve_default_host_paths)
        await environment.ensure_dirs(
            [
                str(Path(self.container_harvey_root).parent),
                str(Path(self.container_probe_path).parent),
                EnvironmentPaths.agent_dir,
                EnvironmentPaths.artifacts_dir / "harvey-output",
            ]
        )
        if self.upload and self.host_harvey_root and self.host_harvey_root.exists():
            with tempfile.TemporaryDirectory(prefix="harvey-labs-upload-") as tmp:
                bundle_root = Path(tmp) / "harvey-labs"
                shutil.copytree(
                    self.host_harvey_root,
                    bundle_root,
                    ignore=shutil.ignore_patterns(*IGNORED_UPLOAD_NAMES),
                )
                await environment.upload_dir(bundle_root, self.container_harvey_root)

        if self.upload and self.host_probe_path and self.host_probe_path.exists():
            await environment.upload_file(
                self.host_probe_path, self.container_probe_path
            )

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
                f"{result.return_code}\nSTDOUT:\n{result.stdout or ''}\nSTDERR:\n{result.stderr or ''}"
            )
        self._populate_context_from_metrics(context, run_id)

    def _resolve_default_host_paths(self) -> None:
        if self.host_harvey_root is None:
            for root in self._candidate_repo_roots():
                candidate = root / "archive" / "harvey-labs"
                if (candidate / "harness" / "run.py").is_file():
                    self.host_harvey_root = candidate
                    break
        if self.host_harvey_root is None:
            self.host_harvey_root = self._prepare_harvey_source()
        if self.host_probe_path is None:
            for candidate in (
                Path(__file__).with_name("harvey_castform_probe.py"),
                *(
                    root / "archive" / "harvey_castform_probe.py"
                    for root in self._candidate_repo_roots()
                ),
            ):
                if candidate.exists():
                    self.host_probe_path = candidate
                    break
        if self.host_probe_path is None:
            raise RuntimeError("harvey_castform_probe.py is missing from the bundle")

    def _prepare_harvey_source(self) -> Path:
        """Clone Harvey's sparse harness source once per trainer host."""

        repository = self._env("HARBOR_HARVEY_GIT_URL") or DEFAULT_HARVEY_REPOSITORY
        git_ref = self._env("HARBOR_HARVEY_GIT_REF") or DEFAULT_HARVEY_GIT_REF
        cache = (
            Path(tempfile.gettempdir())
            / f"castform-harvey-labs-{self._slug(git_ref) or 'main'}"
        )
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
                        "--depth",
                        "1",
                        "--branch",
                        git_ref,
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
                        "failed to configure the Harvey LAB sparse checkout: "
                        f"{sparse.stderr.strip()}"
                    )
                staging.rename(cache)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        return cache

    def _run_command(self, run_id: str, instruction: str) -> str:
        if self.prompt_source == "harbor":
            args = [
                "run-harbor-task",
                "--instruction-file",
                str(EnvironmentPaths.agent_dir / "harvey-instruction.md"),
                "--task-name",
                self._harbor_task_name(),
                "--documents-dir",
                self.harbor_documents_dir,
                "--output-dir",
                self.harbor_output_dir,
            ]
        else:
            args = [
                "run",
                "--task",
                self.harvey_task or "",
            ]
        args.extend(
            [
                "--model",
                self._model_name_for_harvey(),
                "--base-url",
                self._base_url(),
                "--run-id",
                run_id,
                "--max-turns",
                str(self.max_turns),
                "--max-tokens",
                str(self.max_tokens),
                "--max-tool-result-chars",
                str(self.max_tool_result_chars),
                "--shell-timeout",
                str(self.shell_timeout),
                "--sandbox-backend",
                "local",
                "--allow-local-tools",
            ]
        )
        result_dir = f"{self.container_harvey_root}/results/{run_id}"
        output_copy_dir = self._env("HARBOR_HARVEY_OUTPUT_COPY_DIR")
        if self.prompt_source == "harbor" and output_copy_dir is None:
            output_copy_dir = self.harbor_output_dir
        output_copy_block = ""
        if output_copy_dir:
            output_copy_block = (
                f"mkdir -p {shlex.quote(output_copy_dir)}; "
                f"if [ -d {shlex.quote(result_dir + '/output')} ]; then "
                f"cp -a {shlex.quote(result_dir + '/output/.')} {shlex.quote(output_copy_dir + '/')}; "
                "fi; "
            )
        return "\n".join(
            [
                "set -euo pipefail",
                f"HARVEY_ROOT={shlex.quote(self.container_harvey_root)}",
                f"PROBE={shlex.quote(self.container_probe_path)}",
                f"PREBAKED_PYTHON={shlex.quote(self._prebaked_python())}",
                f"BOOTSTRAP_PIP_PACKAGES={shlex.quote(self._bootstrap_pip_packages())}",
                self._write_instruction_block(instruction),
                'if [ -n "${BOOTSTRAP_PIP_PACKAGES}" ]; then',
                "  python -m pip install --quiet --no-cache-dir ${BOOTSTRAP_PIP_PACKAGES}",
                "fi",
                'if [ -n "${HARBOR_HARVEY_PYTHON:-}" ]; then',
                '  RUNNER=( "${HARBOR_HARVEY_PYTHON}" "${PROBE}" )',
                'elif [ -x "${PREBAKED_PYTHON}" ]; then',
                '  RUNNER=( "${PREBAKED_PYTHON}" "${PROBE}" )',
                "elif command -v uv >/dev/null 2>&1; then",
                '  RUNNER=( uv run --project "${HARVEY_ROOT}" "${PROBE}" )',
                "else",
                '  RUNNER=( python "${PROBE}" )',
                "fi",
                '"${RUNNER[@]}" ' + " ".join(shlex.quote(arg) for arg in args),
                f"RESULT_DIR={shlex.quote(result_dir)}",
                f"rm -rf {shlex.quote(str(EnvironmentPaths.agent_dir / 'harvey-results'))}",
                f'if [ -d "$RESULT_DIR" ]; then cp -a "$RESULT_DIR" {shlex.quote(str(EnvironmentPaths.agent_dir / "harvey-results"))}; fi',
                f"mkdir -p {shlex.quote(str(EnvironmentPaths.artifacts_dir / 'harvey-output'))}",
                f'if [ -d "$RESULT_DIR/output" ]; then cp -a "$RESULT_DIR/output/." {shlex.quote(str(EnvironmentPaths.artifacts_dir / "harvey-output/"))}; fi',
                output_copy_block.rstrip(),
            ]
        )

    def _execution_env(self) -> dict[str, str]:
        env = {
            "OPENAI_API_KEY": self._api_key(),
            "OPENAI_BASE_URL": self._base_url(),
            "OPENAI_API_BASE": self._base_url(),
        }
        for key, value in {**os.environ, **self.extra_env}.items():
            if key.startswith("HARBOR_HARVEY_") or key == "CASTFORM_API_KEY":
                env[key] = value
        return {key: value for key, value in env.items() if value}

    def _populate_context_from_metrics(
        self, context: AgentContext, run_id: str
    ) -> None:
        metrics_path = self.logs_dir / "harvey-results" / "metrics.json"
        if not metrics_path.exists():
            context.metadata = {"harvey_run_id": run_id}
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            context.metadata = {"harvey_run_id": run_id}
            return
        context.n_input_tokens = metrics.get("input_tokens")
        context.n_output_tokens = metrics.get("output_tokens")
        context.metadata = {
            "harvey_run_id": run_id,
            "harvey_metrics": metrics,
        }

    def _run_id(self) -> str:
        explicit = self._env("HARBOR_HARVEY_RUN_ID")
        if explicit:
            return explicit
        trial_name = self.logs_dir.parent.name if self.logs_dir.parent.name else "trial"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_name = self.harvey_task or self._harbor_task_name()
        return f"harbor-agent/{task_name}/{self._slug(self._model_name_for_harvey())}/{trial_name}-{timestamp}"

    def _harbor_task_name(self) -> str:
        task_name = self._env("HARBOR_HARVEY_TASK_NAME")
        if task_name:
            return task_name
        trial_name = (
            self.logs_dir.parent.name if self.logs_dir.parent.name else "harbor-task"
        )
        return self._slug(trial_name) or "harbor-task"

    @staticmethod
    def _write_instruction_block(instruction: str) -> str:
        encoded = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
        instruction_path = EnvironmentPaths.agent_dir / "harvey-instruction.md"
        return "\n".join(
            [
                f"INSTRUCTION_FILE={shlex.quote(str(instruction_path))}",
                'mkdir -p "$(dirname "$INSTRUCTION_FILE")"',
                "python - <<'PY'",
                "import base64",
                "from pathlib import Path",
                f"Path({str(instruction_path)!r}).write_bytes(base64.b64decode({encoded!r}))",
                "PY",
            ]
        )

    def _model_name_for_harvey(self) -> str:
        model = self._env("HARBOR_HARVEY_MODEL") or self.model_name or ""
        if self._env_bool(
            "HARBOR_HARVEY_STRIP_OPENAI_PREFIX", True
        ) and model.startswith("openai/"):
            return model.split("/", 1)[1]
        return model

    def _api_key(self) -> str:
        value = self.extra_env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not value:
            value = self.extra_env.get("CASTFORM_API_KEY") or os.environ.get(
                "CASTFORM_API_KEY"
            )
        if not value:
            raise RuntimeError(
                "HarveyHarnessAgent requires OPENAI_API_KEY or CASTFORM_API_KEY"
            )
        return value

    def _base_url(self) -> str:
        value = (
            self.extra_env.get("OPENAI_BASE_URL")
            or self.extra_env.get("OPENAI_API_BASE")
            or self.extra_env.get("NEON_AI_GATEWAY_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("NEON_AI_GATEWAY_BASE_URL")
        )
        if not value:
            raise RuntimeError(
                "HarveyHarnessAgent requires OPENAI_BASE_URL or NEON_AI_GATEWAY_BASE_URL"
            )
        return value.rstrip("/")

    def _prebaked_python(self) -> str:
        return (
            self._env("HARBOR_HARVEY_PREBAKED_PYTHON") or DEFAULT_PREBAKED_VENV_PYTHON
        )

    def _bootstrap_pip_packages(self) -> str:
        raw = self._env("HARBOR_HARVEY_BOOTSTRAP_PIP_PACKAGES")
        if raw is not None:
            return raw
        return DEFAULT_BOOTSTRAP_PIP_PACKAGES

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
    def _candidate_repo_roots() -> tuple[Path, ...]:
        """Find likely checkout roots without assuming a fixed package depth."""

        source = Path(__file__).resolve()
        candidates = [Path.cwd().resolve(), *source.parents]
        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return tuple(unique)

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").replace(".", "-")
