import logging
import subprocess
import sys
from pathlib import Path
from typing import Type, Union

import cloudpickle

from benchmax.envs.base_env import BaseEnv
from benchmax.bundle.errors import (
    DependencyError,
    IncompatiblePythonError,
    BundlingError,
)
from benchmax.bundle.payload import EnvPayload

logger = logging.getLogger(__name__)


def load_env(
    payload: EnvPayload | bytes,
    install_deps: bool = True,
    allow_python_mismatch: bool = False,
) -> Type[BaseEnv]:
    """Load a packaged environment class on the remote machine.

    Args:
        payload: An EnvPayload or raw bytes from EnvPayload.to_bytes().
        install_deps: Install pip_dependencies before unpickling.
        allow_python_mismatch: If False, raise on Python version mismatch.

    Returns:
        The unpickled BaseEnv subclass (class object, not instance).

    Raises:
        IncompatiblePythonError: Python version mismatch.
        DependencyError: pip install failed.
        BundlingError: Unpickling failed.
    """
    env_payload: EnvPayload = (
        payload if isinstance(payload, EnvPayload) else EnvPayload.from_bytes(payload)
    )

    # --- Python version check ---
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if env_payload.python_version != current_python and not allow_python_mismatch:
        raise IncompatiblePythonError(
            f"Payload was packaged with Python {env_payload.python_version} "
            f"but this machine runs Python {current_python}. "
            "Set allow_python_mismatch=True to override."
        )

    # --- Install pip dependencies ---
    if install_deps and env_payload.pip_dependencies:
        _install_dependencies(env_payload.pip_dependencies)

    # --- Unpickle the class ---
    try:
        env_class = cloudpickle.loads(env_payload.pickled_class)
    except Exception as e:
        raise BundlingError(
            f"Failed to unpickle environment class: {e}. "
            "This usually means a dependency is missing or there's a "
            "Python version mismatch."
        ) from e

    # --- Post-unpickle validation ---
    if not (isinstance(env_class, type) and issubclass(env_class, BaseEnv)):
        raise BundlingError(
            f"Unpickled object is {type(env_class)}, not a BaseEnv subclass. "
            "The payload may be corrupted."
        )

    logger.info(f"[bundling] Loaded environment class: {env_class.__name__}")
    return env_class


def _install_dependencies(deps: list[str]) -> None:
    """Install pip dependencies in the current environment."""
    logger.info(f"[bundling] Installing {len(deps)} dependencies: {deps}")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *deps]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DependencyError(
            f"pip install failed (exit code {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    logger.info("[bundling] Dependencies installed successfully.")


def load_env_from_path(
    path: Union[str, Path],
    install_deps: bool = True,
    allow_python_mismatch: bool = False,
) -> Type[BaseEnv]:
    """Load a packaged environment class from a file path.

    Args:
        path: Path to a .bmx file containing the serialized EnvPayload.
        install_deps: Install pip_dependencies before unpickling.
        allow_python_mismatch: If False, raise on Python version mismatch.

    Returns:
        The unpickled BaseEnv subclass (class object, not instance).

    Raises:
        FileNotFoundError: If the file does not exist.
        IncompatiblePythonError: Python version mismatch.
        DependencyError: pip install failed.
        BundlingError: Unpickling failed.
    """
    path = Path(path)
    with open(path, "rb") as f:
        payload_bytes = f.read()
    return load_env(payload_bytes, install_deps, allow_python_mismatch)
