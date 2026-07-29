"""Mini-SWE Harbor agent installed from prefetched wheels instead of apt+PyPI.

Harbor's stock ``MiniSweAgent.install`` runs ``apt-get update`` plus two PyPI
installs inside every fresh sandbox (~40-90s per trial, with observed stalls
past the 360s setup timeout). This subclass keeps the upstream run/trajectory
logic and replaces only the install: the pinned wheel closure is downloaded
once on the host, uploaded into the sandbox, and installed offline via the
bundled pip wheel, so no external package service is on the per-trial path.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import override

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

MINI_SWE_AGENT_VERSION = "2.4.5"
LITELLM_VERSION = "1.75.5.post1"

# The AIME sandbox images are Ubuntu 24.04 (CPython 3.12, x86_64, no pip).
_WHEEL_PYTHON_VERSION = "3.12"
_WHEEL_ABIS = ("cp312", "abi3", "none")
_WHEEL_PLATFORMS = (
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_28_x86_64",
)
_CONTAINER_WHEELS_DIR = "/opt/miniswe-wheels"
_CACHE_COMPLETE_MARKER = ".complete"

_prefetch_lock = threading.Lock()


def wheel_cache_dir() -> Path:
    """Stable per-host cache location for the pinned wheel closure."""

    return (
        Path(tempfile.gettempdir())
        / f"miniswe-wheels-{MINI_SWE_AGENT_VERSION}-litellm-{LITELLM_VERSION}"
    )


def prefetch_wheels(
    packages: tuple[str, ...] | None = None,
    *,
    cache: Path | None = None,
    no_deps: bool = False,
    extra_no_deps: tuple[str, ...] = (),
) -> Path:
    """Download a pinned wheel set once per host; safe across processes.

    Downloads land in a temp directory and are atomically renamed into place,
    so concurrent runners either see the complete cache or build their own.
    """

    if packages is None:
        packages = (
            f"mini-swe-agent=={MINI_SWE_AGENT_VERSION}",
            f"litellm=={LITELLM_VERSION}",
            "pip",
        )
    if cache is None:
        cache = wheel_cache_dir()
    with _prefetch_lock:
        if (
            cache.is_dir()
            and (cache / _CACHE_COMPLETE_MARKER).is_file()
            and any(cache.glob("*.whl"))
            and any(cache.glob("pip-*.whl"))
        ):
            return cache
        # A previous interrupted download or upload can leave an empty cache
        # directory. Never treat directory existence alone as completion.
        shutil.rmtree(cache, ignore_errors=True)
        staging = Path(f"{cache}.tmp-{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        uv = shutil.which("uv")
        if uv:
            downloader = [uv, "tool", "run", "--from", "pip", "pip"]
        elif importlib.util.find_spec("pip") is not None:
            downloader = [sys.executable, "-m", "pip"]
        else:
            raise RuntimeError("wheel prefetch requires uv or an installed pip module")
        batches = [(packages, no_deps)]
        if extra_no_deps:
            batches.append((extra_no_deps, True))
        for batch, batch_no_deps in batches:
            command = [
                *downloader,
                "download",
                *batch,
                *(["--no-deps"] if batch_no_deps else []),
                "--dest",
                str(staging),
                "--only-binary",
                ":all:",
                "--python-version",
                _WHEEL_PYTHON_VERSION,
                "--implementation",
                "cp",
            ]
            for abi in _WHEEL_ABIS:
                command += ["--abi", abi]
            for platform in _WHEEL_PLATFORMS:
                command += ["--platform", platform]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                shutil.rmtree(staging, ignore_errors=True)
                raise RuntimeError(
                    f"wheel prefetch failed (exit {result.returncode}):\n{result.stderr}"
                )
        (staging / _CACHE_COMPLETE_MARKER).write_text("ok\n")
        try:
            staging.rename(cache)
        except OSError:
            # Another process won the rename race; its cache is complete.
            shutil.rmtree(staging, ignore_errors=True)
            if not cache.is_dir():
                raise
        return cache


class PrefetchedMiniSweAgent(MiniSweAgent):
    """Mini-SWE with an offline, version-pinned install from uploaded wheels."""

    @staticmethod
    @override
    def name() -> str:
        return "prefetched-mini-swe-agent"

    @override
    def get_version_command(self) -> str | None:
        # The uv-tool probe from upstream does not apply to a pip install.
        return None

    @override
    def version(self) -> str | None:
        return MINI_SWE_AGENT_VERSION

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        wheels = prefetch_wheels()
        await environment.upload_dir(wheels, _CONTAINER_WHEELS_DIR)
        # Ubuntu images ship python3 without pip (and mark it externally
        # managed), so pip runs from its own uploaded wheel.
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                f"PIP_WHEEL=$(ls {_CONTAINER_WHEELS_DIR}/pip-*.whl | head -1); "
                'python3 "$PIP_WHEEL/pip" install --quiet --no-index '
                f"--find-links {_CONTAINER_WHEELS_DIR} --break-system-packages "
                f"'mini-swe-agent=={MINI_SWE_AGENT_VERSION}' "
                f"'litellm=={LITELLM_VERSION}' "
                "&& mini-swe-agent --help >/dev/null"
            ),
        )


_CONTAINER_PROBE_PATH = "/opt/castform-miniswe/mini_swe_probe.py"

# Libraries the non-litellm code paths of upstream reach (package __init__
# pulls rich/dotenv/platformdirs; loop needs jinja2+pydantic; driver needs
# pyyaml). Resolved WITH their deps so version pairs stay consistent; only the
# mini-swe-agent wheel itself installs --no-deps so litellm never resolves.
_UPSTREAM_LIBS = (
    "jinja2",
    "pydantic",
    "pyyaml",
    "python-dotenv",
    "platformdirs",
    "rich",
    "pip",
)
_UPSTREAM_FILES = ("castform_model.py", "run_mini_castform.py")
_CONTAINER_UPSTREAM_DIR = "/opt/castform-miniswe"

# Sources captured at import time so cloudpickle by-value bundles carry them:
# on the trainer host this module has no real __file__ to read from.
_AUX_SOURCES: dict[str, str] = {
    name: Path(__file__).with_name(name).read_text()
    for name in ("mini_swe_probe.py", *_UPSTREAM_FILES)
}


class VendoredMiniSweAgent(MiniSweAgent):
    """Mini-SWE loop vendored as one stdlib-only file; zero-install setup.

    Harvey-harness pattern: agent setup uploads a single ~15KB probe script and
    runs it with the image's python3 — no apt, wheels, or PyPI in the sandbox.
    The probe mirrors mini-swe-agent 2.4.5's tool-call loop and prompts.
    """

    @staticmethod
    @override
    def name() -> str:
        return "vendored-mini-swe-agent"

    @override
    def get_version_command(self) -> str | None:
        return None

    @override
    def version(self) -> str | None:
        return f"{MINI_SWE_AGENT_VERSION}-castform-probe"

    _SCRIPT_PATH = _CONTAINER_PROBE_PATH

    async def _upload_source(
        self,
        environment: BaseEnvironment,
        name: str,
        target: str,
    ) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(_AUX_SOURCES[name])
        try:
            await environment.upload_file(handle.name, target)
        finally:
            os.unlink(handle.name)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self._upload_source(environment, "mini_swe_probe.py", _CONTAINER_PROBE_PATH)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        api_base = self._get_env("OPENAI_BASE_URL") or self._get_env("OPENAI_API_BASE")
        api_key = self._get_env("OPENAI_API_KEY")
        if not api_base or not api_key:
            raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY must be set")

        command = (
            f"python3 {self._SCRIPT_PATH} "
            f"--model {shlex.quote(self.model_name.removeprefix('openai/'))} "
            f"--base-url {shlex.quote(api_base)} "
            f"--task {shlex.quote(instruction)} "
            "--output /logs/agent/mini-swe-agent.trajectory.json "
            "> /logs/agent/mini-swe-agent.txt 2>&1 </dev/null"
        )
        result = await self.exec_as_agent(
            environment,
            command=command,
            env={"OPENAI_API_KEY": api_key},
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"vendored mini-swe probe failed with exit code {result.return_code}\n"
                f"STDOUT:\n{result.stdout or ''}\nSTDERR:\n{result.stderr or ''}"
            )


class UpstreamMiniSweAgent(VendoredMiniSweAgent):
    """Upstream minisweagent package, offline-installed, litellm swapped out.

    The real mini-swe-agent 2.4.5 code (loop, prompts, parsing, trajectory)
    runs unmodified from a --no-deps wheel install; only the model class is
    ours (stdlib HTTP to the OpenAI-compatible endpoint). ~15MB payload vs
    114MB with litellm, and no apt/PyPI in the sandbox.
    """

    _SCRIPT_PATH = f"{_CONTAINER_UPSTREAM_DIR}/run_mini_castform.py"

    @staticmethod
    @override
    def name() -> str:
        return "upstream-mini-swe-agent"

    @override
    def version(self) -> str | None:
        return f"{MINI_SWE_AGENT_VERSION}-castform-model"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        cache = Path(tempfile.gettempdir()) / f"miniswe-upstream-wheels-{MINI_SWE_AGENT_VERSION}"
        wheels = prefetch_wheels(
            _UPSTREAM_LIBS,
            cache=cache,
            extra_no_deps=(f"mini-swe-agent=={MINI_SWE_AGENT_VERSION}",),
        )
        await environment.upload_dir(wheels, _CONTAINER_WHEELS_DIR)
        for name in _UPSTREAM_FILES:
            await self._upload_source(environment, name, f"{_CONTAINER_UPSTREAM_DIR}/{name}")
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                f"PIP_WHEEL=$(ls {_CONTAINER_WHEELS_DIR}/pip-*.whl | head -1); "
                'python3 "$PIP_WHEEL/pip" install --quiet --no-index '
                f"--find-links {_CONTAINER_WHEELS_DIR} --break-system-packages "
                + " ".join(f"'{library}'" for library in _UPSTREAM_LIBS)
                + "; "
                'python3 "$PIP_WHEEL/pip" install --quiet --no-index --no-deps '
                f"--find-links {_CONTAINER_WHEELS_DIR} --break-system-packages "
                f"'mini-swe-agent=={MINI_SWE_AGENT_VERSION}'"
            ),
        )
