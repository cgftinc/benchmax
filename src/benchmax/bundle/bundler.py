import logging
import sys
import types
from typing import Any, Dict, List, Optional, Type

from pathlib import Path

import cloudpickle

from benchmax.envs.base_env import BaseEnv
from benchmax.bundle.errors import BundlingError
from benchmax.bundle.payload import BundleMetadata, BundledEnv
from benchmax.bundle.validator import validate_structure

logger = logging.getLogger(__name__)


def bundle_env(
    env_class: Type[BaseEnv],
    pip_dependencies: Optional[List[str]] = None,
    local_modules: Optional[List[types.ModuleType]] = None,
    validate: bool = True,
    constructor_args: Optional[Dict[str, Any]] = None,
) -> BundledEnv:
    """Bundle a BaseEnv subclass for remote execution.

    Serializes the class (not an instance) using cloudpickle and bundles
    metadata needed to reconstruct it on a remote machine. Optional constructor
    kwargs can be included in metadata for auto-instantiation.

    Args:
        env_class: The class to bundle. Must be a BaseEnv subclass.
        pip_dependencies: Pip dependency strings (e.g. ["aiohttp>=3.9", "numpy"]).
            Installed on the remote machine before unpickling.
        local_modules: Module objects to register with cloudpickle for
            pickle-by-value. Required when the class imports from local .py
            files that are not installed packages. NOT needed for code
            defined in notebook cells.
        validate: Run structural validation before bundling.
        constructor_args: Optional constructor kwargs to store in metadata.

    Returns:
        BundledEnv containing the serialized class and metadata.

    Raises:
        BundlingError: If serialization fails.
        ValidationError: If structural validation fails.
    """
    pip_dependencies = pip_dependencies or []
    # --- Structural validation ---
    if validate:
        warnings = validate_structure(env_class, pip_dependencies)
        for w in warnings:
            logger.warning(f"[bundling] {w}")

    # --- Register local modules for pickle-by-value ---
    registered_modules: List[types.ModuleType] = []
    if local_modules:
        for mod in local_modules:
            if not isinstance(mod, types.ModuleType):
                raise BundlingError(
                    f"local_modules must contain module objects, got "
                    f"{type(mod)}: {mod}"
                )
            cloudpickle.register_pickle_by_value(mod)
            registered_modules.append(mod)
            logger.info(
                f"[bundling] Registered module for pickle-by-value: "
                f"{mod.__name__}"
            )

    # --- Serialize the class ---
    try:
        pickled_class = cloudpickle.dumps(env_class)
    except Exception as e:
        raise BundlingError(
            f"Failed to serialize {env_class.__name__} with cloudpickle: {e}"
        ) from e
    finally:
        for mod in registered_modules:
            try:
                cloudpickle.unregister_pickle_by_value(mod)
            except Exception:
                pass

    # --- Version info ---
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        from importlib.metadata import version as get_version

        benchmax_version = get_version("benchmax")
    except Exception:
        benchmax_version = "unknown"

    # --- Build payload ---
    metadata = BundleMetadata(
        pip_dependencies=pip_dependencies,
        python_version=python_version,
        benchmax_version=benchmax_version,
        constructor_args=constructor_args,
    )
    payload = BundledEnv(pickled_class=pickled_class, metadata=metadata)

    size_kb = len(pickled_class) / 1024
    logger.info(
        f"[bundling] Bundled {env_class.__name__}: "
        f"{size_kb:.1f} KB pickled, {len(pip_dependencies)} pip deps"
    )

    return payload


def write_bundle_files(
    bundle: BundledEnv, pickle_path: Path, metadata_path: Path
) -> None:
    pickle_path = Path(pickle_path)
    metadata_path = Path(metadata_path)
    pickle_path.write_bytes(bundle.pickled_class)
    metadata_path.write_bytes(bundle.metadata.to_json_bytes())


def read_bundle_files(pickle_path: Path, metadata_path: Path) -> BundledEnv:
    pickle_path = Path(pickle_path)
    metadata_path = Path(metadata_path)
    pickled_class = pickle_path.read_bytes()
    metadata = BundleMetadata.from_json_bytes(metadata_path.read_bytes())
    return BundledEnv(pickled_class=pickled_class, metadata=metadata)
