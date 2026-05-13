import dataclasses
import inspect
import json
import logging
import sys
import threading
import types
from typing import Any, Dict, List, Optional, Type

from pathlib import Path

import cloudpickle

from benchmax.envs.base_env import BaseEnv
from benchmax.bundle.errors import BundlingError
from benchmax.bundle.payload import BundleMetadata, BundledEnv
from benchmax.bundle.validator import validate_structure

logger = logging.getLogger(__name__)

# cloudpickle.register_pickle_by_value mutates process-global state. Two
# concurrent bundle_env() calls registering the same module would race —
# T1's unregister would silently un-register T2's still-needed registration,
# and T2's pickled output would fall back to by-reference (wrong behavior).
# This lock serializes the register/dump/unregister window only — the
# underlying cloudpickle.dumps releases the GIL so it's still parallel-friendly
# for the bytes-shuffling phase.
_BUNDLE_LOCK = threading.Lock()


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
    ensure_safe_python_version()
    pip_dependencies = pip_dependencies or []
    # --- Structural validation ---
    if validate:
        warnings = validate_structure(env_class, pip_dependencies)
        for w in warnings:
            logger.warning(f"[bundling] {w}")

    # --- Register local modules + serialize, all under a single lock ---
    # See module-level _BUNDLE_LOCK comment for rationale.
    pickled_constructor_args: bytes | None = None
    with _BUNDLE_LOCK:
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

        try:
            pickled_class = cloudpickle.dumps(env_class)
            # Pickle constructor_args now (while local_modules are registered)
            # so non-JSON-serializable objects get pickled by value.
            if constructor_args is not None:
                try:
                    json.dumps(constructor_args)
                except TypeError:
                    pickled_constructor_args = cloudpickle.dumps(constructor_args)
            sources = _collect_sources(env_class)
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
        constructor_args_pickled=pickled_constructor_args is not None,
        sources=sources or None,
    )
    payload = BundledEnv(
        pickled_class=pickled_class,
        metadata=metadata,
        pickled_constructor_args=pickled_constructor_args,
    )

    size_kb = len(pickled_class) / 1024
    logger.info(
        f"[bundling] Bundled {env_class.__name__}: "
        f"{size_kb:.1f} KB pickled, {len(pip_dependencies)} pip deps"
    )

    return payload


def _collect_sources(env_class: Type[BaseEnv]) -> Dict[str, str]:
    """Capture just the env class definition as Python source.

    Returns ``{class_name: source_text}``. We deliberately do NOT capture the
    enclosing module, ``local_modules``, or transitive imports — for the "show
    the user the env code" use case the class block is what matters; helpers
    and dataset-generation code in the same file are noise.

    A consequence: module-level helpers the class calls (``_extract_answer_block``,
    etc.) won't appear here. Readers see *what* the class does, not *how* its
    private helpers work.
    """
    out: Dict[str, str] = {}
    try:
        out[env_class.__name__] = inspect.getsource(env_class)
    except (OSError, TypeError) as e:
        logger.debug("[bundling] no source for class %s: %s", env_class.__name__, e)
    return out


def ensure_safe_python_version() -> None:
    v = sys.version_info

    if (v.major, v.minor) == (3, 13):
        full_version = f"{v.major}.{v.minor}.{v.micro}"
        raise RuntimeError(
            "\n"
            "❌ Unsupported Python version detected.\n\n"
            f"Current interpreter: Python {full_version}\n\n"
            "Python 3.13.x has a pathlib.Path pickle incompatibility "
            "that breaks cross-version unpickling.\n\n"
            "Please use:\n"
            "  • Python <= 3.12\n"
            "  • OR Python >= 3.14\n"
        )


def write_bundle_files(
    bundle: BundledEnv, pickle_path: Path, metadata_path: Path
) -> None:
    pickle_path = Path(pickle_path)
    metadata_path = Path(metadata_path)
    pickle_path.write_bytes(bundle.pickled_class)
    metadata_path.write_bytes(bundle.metadata.to_json_bytes())

    # Write pre-pickled constructor_args (pickled during bundling while
    # local_modules were registered, so module code is inline).
    if bundle.pickled_constructor_args is not None:
        args_path = pickle_path.with_suffix(".args.pkl")
        args_path.write_bytes(bundle.pickled_constructor_args)
        logger.info("[bundling] Wrote pickled constructor_args to %s", args_path)


def read_bundle_files(pickle_path: Path, metadata_path: Path) -> BundledEnv:
    pickle_path = Path(pickle_path)
    metadata_path = Path(metadata_path)
    pickled_class = pickle_path.read_bytes()
    metadata = BundleMetadata.from_json_bytes(metadata_path.read_bytes())

    # Load pickled constructor_args if they were serialized separately.
    if metadata.constructor_args_pickled:
        args_pickle_path = pickle_path.with_suffix(".args.pkl")
        if args_pickle_path.exists():
            constructor_args = cloudpickle.loads(args_pickle_path.read_bytes())
            metadata = dataclasses.replace(metadata, constructor_args=constructor_args)
            logger.info("[bundling] Loaded pickled constructor_args from %s", args_pickle_path)
        else:
            logger.warning(
                "[bundling] Metadata indicates pickled constructor_args but %s not found",
                args_pickle_path,
            )

    return BundledEnv(pickled_class=pickled_class, metadata=metadata)
