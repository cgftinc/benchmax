import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import cloudpickle

from benchmax.envs.base_env import BaseEnv
from benchmax.bundle.errors import (
    DependencyError,
    IncompatiblePythonError,
    IncompatibleBenchmaxError,
    BundlingError,
)
from benchmax.bundle.payload import BundleMetadata, BundledEnv

logger = logging.getLogger(__name__)


def load_env(
    pickled_class: bytes,
    metadata: BundleMetadata | None = None,
    *,
    pip_dependencies: Optional[list[str]] = None,
    python_version: Optional[str] = None,
    benchmax_version: Optional[str] = None,
    constructor_args: Optional[Dict[str, Any]] = None,
    check_python_version: bool = True,
    check_benchmax_version: bool = False,
    install_pip_deps: bool = False,
    instantiate: Optional[bool] = None,
) -> BaseEnv | Type[BaseEnv]:
    """Load a packaged environment class (and optionally instantiate it).

    Args:
        pickled_class: Bytes of the cloudpickle-serialized class.
        metadata: Optional BundleMetadata providing defaults for version
            checks, dependencies, and constructor args.
        pip_dependencies: Optional explicit pip deps (override metadata).
        python_version: Optional explicit python version (override metadata).
        benchmax_version: Optional explicit benchmax version (override metadata).
        constructor_args: Optional constructor kwargs (override metadata).
        check_python_version: If True, enforce Python version check.
        check_benchmax_version: If True, enforce benchmax version check.
        install_pip_deps: If True, install pip dependencies before unpickling.
        instantiate: If None, auto-instantiate when constructor args exist.
            If True, always instantiate (empty kwargs if none).
            If False, always return the class.

    Returns:
        The unpickled BaseEnv subclass (class object), or an instance if
        instantiation is requested.
    """
    resolved_pip_deps = (
        pip_dependencies
        if pip_dependencies is not None
        else (metadata.pip_dependencies if metadata else [])
    )
    resolved_python_version = (
        python_version
        if python_version is not None
        else (metadata.python_version if metadata else "unknown")
    )
    resolved_benchmax_version = (
        benchmax_version
        if benchmax_version is not None
        else (metadata.benchmax_version if metadata else "unknown")
    )
    resolved_constructor_args = (
        constructor_args
        if constructor_args is not None
        else (metadata.constructor_args if metadata else None)
    )

    if check_python_version and resolved_python_version != "unknown":
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if resolved_python_version != current_python:
            raise IncompatiblePythonError(
                f"Payload was packaged with Python {resolved_python_version} "
                f"but this machine runs Python {current_python}. "
                "Set check_python_version=False to override."
            )

    if check_benchmax_version and resolved_benchmax_version != "unknown":
        try:
            from importlib.metadata import version as get_version

            current_benchmax = get_version("benchmax")
        except Exception:
            current_benchmax = "unknown"

        if current_benchmax != "unknown" and resolved_benchmax_version != current_benchmax:
            raise IncompatibleBenchmaxError(
                f"Payload was packaged with benchmax {resolved_benchmax_version} "
                f"but this machine runs benchmax {current_benchmax}. "
                "Set check_benchmax_version=False to override."
            )

    if install_pip_deps and resolved_pip_deps:
        _install_dependencies(resolved_pip_deps)

    try:
        env_class = cloudpickle.loads(pickled_class)
    except Exception as e:
        raise BundlingError(
            f"Failed to unpickle environment class: {e}. "
            "This usually means a dependency is missing or there's a "
            "Python version mismatch."
        ) from e

    if not (isinstance(env_class, type) and issubclass(env_class, BaseEnv)):
        raise BundlingError(
            f"Unpickled object is {type(env_class)}, not a BaseEnv subclass. "
            "The payload may be corrupted."
        )

    if instantiate is None:
        instantiate = resolved_constructor_args is not None

    if instantiate:
        resolved_constructor_args = resolved_constructor_args or {}
        instance = env_class(**resolved_constructor_args)
        logger.info(f"[bundling] Instantiated environment: {env_class.__name__}")
        return instance

    logger.info(f"[bundling] Loaded environment class: {env_class.__name__}")
    return env_class


def load_env_from_files(
    pickle_path: Union[str, Path],
    metadata_path: Union[str, Path],
    *,
    pip_dependencies: Optional[list[str]] = None,
    python_version: Optional[str] = None,
    benchmax_version: Optional[str] = None,
    constructor_args: Optional[Dict[str, Any]] = None,
    check_python_version: bool = True,
    check_benchmax_version: bool = False,
    install_pip_deps: bool = False,
    instantiate: Optional[bool] = None,
) -> BaseEnv | Type[BaseEnv]:
    pickle_path = Path(pickle_path)
    metadata_path = Path(metadata_path)
    pickled_class = pickle_path.read_bytes()
    metadata = BundleMetadata.from_json_bytes(metadata_path.read_bytes())
    return load_env(
        pickled_class,
        metadata,
        pip_dependencies=pip_dependencies,
        python_version=python_version,
        benchmax_version=benchmax_version,
        constructor_args=constructor_args,
        check_python_version=check_python_version,
        check_benchmax_version=check_benchmax_version,
        install_pip_deps=install_pip_deps,
        instantiate=instantiate,
    )


def load_env_from_bundle(
    bundle: BundledEnv,
    *,
    pip_dependencies: Optional[list[str]] = None,
    python_version: Optional[str] = None,
    benchmax_version: Optional[str] = None,
    constructor_args: Optional[Dict[str, Any]] = None,
    check_python_version: bool = True,
    check_benchmax_version: bool = False,
    install_pip_deps: bool = False,
    instantiate: Optional[bool] = None,
) -> BaseEnv | Type[BaseEnv]:
    return load_env(
        bundle.pickled_class,
        bundle.metadata,
        pip_dependencies=pip_dependencies,
        python_version=python_version,
        benchmax_version=benchmax_version,
        constructor_args=constructor_args,
        check_python_version=check_python_version,
        check_benchmax_version=check_benchmax_version,
        install_pip_deps=install_pip_deps,
        instantiate=instantiate,
    )


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
