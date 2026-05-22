"""Bundling a BaseEnv subclass for remote execution.

Two functions, two dataclasses. The bundle module is pure serialization —
disk, blob storage, and transport are the caller's concern.

For validation (does this env actually work end-to-end?), use
``benchmax.platform.validation.validate_env`` *before* bundling. If that
passes, ``dump_bundle`` will produce a loadable bundle.
"""

from __future__ import annotations

import inspect
import io
import json
import logging
import pickle
import sys
import threading
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from types import ModuleType
from typing import Any

import cloudpickle

from benchmax.envs.base_env import BaseEnv

logger = logging.getLogger(__name__)


class BundlingError(Exception):
    """Raised when cloudpickle serialization or class-shape checks fail."""


class IncompatiblePythonError(Exception):
    """Raised when the loading interpreter doesn't match the bundle's python_version."""


# cloudpickle.register_pickle_by_value mutates process-global state. Two
# concurrent dump_bundle() calls registering the same module would race —
# T1's unregister would silently un-register T2's still-needed registration,
# and T2's pickled output would fall back to by-reference (wrong behavior).
_BUNDLE_LOCK = threading.Lock()


@dataclass(frozen=True)
class BundleMetadata:
    """Pre-unpickle info — readable without touching the pickle.

    The trainer / rollout-service / frontend read this to decide what to
    install, which interpreter to launch, or to render the env source in
    a UI tab, before (or without) unpickling the class.
    """

    pip_dependencies: list[str]
    python_version: str
    benchmax_version: str
    env_class_source: str | None

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            {
                "pip_dependencies": self.pip_dependencies,
                "python_version": self.python_version,
                "benchmax_version": self.benchmax_version,
                "env_class_source": self.env_class_source,
            }
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "BundleMetadata":
        d = json.loads(data.decode("utf-8"))
        return cls(
            pip_dependencies=list(d["pip_dependencies"]),
            python_version=d["python_version"],
            benchmax_version=d["benchmax_version"],
            env_class_source=d.get("env_class_source"),
        )


@dataclass(frozen=True)
class Bundle:
    """A serialized env.

    ``pickled`` is ``cloudpickle.dumps((env_class, constructor_args))``.
    """

    pickled: bytes
    metadata: BundleMetadata


def dump_bundle(
    env_class: type[BaseEnv],
    *,
    constructor_args: dict[str, Any] | None = None,
    pip_dependencies: list[str] | None = None,
    local_modules: list[ModuleType] | None = None,
) -> Bundle:
    """Pickle ``(env_class, constructor_args)`` and stamp metadata.

    Args:
        env_class: A concrete BaseEnv subclass.
        constructor_args: kwargs passed to ``env_class(**...)`` on load.
            Defaults to ``{}``.
        pip_dependencies: Packages the trainer/worker must have installed
            before loading. Recorded in metadata; this function does NOT
            install anything.
        local_modules: Module objects to register with cloudpickle for
            pickle-by-value. Required when the env class — or anything it
            transitively references — lives in a local ``.py`` file that
            isn't installed as a distribution.

    Raises:
        BundlingError: ``env_class`` is not a concrete BaseEnv subclass,
            cloudpickle fails, or the pickle references non-installed modules
            that weren't passed in ``local_modules`` (a bundle that would
            ``ModuleNotFoundError`` on a fresh worker process).
    """
    _ensure_safe_python_version()
    _check_env_class(env_class)

    constructor_args = constructor_args or {}
    pip_dependencies = pip_dependencies or []
    local_modules = local_modules or []

    with _BUNDLE_LOCK:
        for mod in local_modules:
            if not isinstance(mod, ModuleType):
                raise BundlingError(
                    f"local_modules must contain module objects, got "
                    f"{type(mod).__name__}: {mod!r}"
                )
            cloudpickle.register_pickle_by_value(mod)
        try:
            try:
                pickled = cloudpickle.dumps((env_class, constructor_args))
            except Exception as e:
                raise BundlingError(
                    f"Failed to serialize {env_class.__name__} with cloudpickle: {e}"
                ) from e
        finally:
            for mod in local_modules:
                try:
                    cloudpickle.unregister_pickle_by_value(mod)
                except Exception:
                    pass

    risky = _unregistered_local_refs(pickled)
    if risky:
        msg = (
            f"{env_class.__name__}: missing "
            f"local_modules=[{', '.join(risky)}]"
        )
        if local_modules:
            already = ", ".join(sorted(m.__name__ for m in local_modules))
            msg += f" (already registered: [{already}])"
        raise BundlingError(msg)

    metadata = BundleMetadata(
        pip_dependencies=pip_dependencies,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        benchmax_version=_benchmax_version(),
        env_class_source=_get_source(env_class),
    )

    logger.info(
        "[bundle] dumped %s: %.1f KB pickled, %d pip deps",
        env_class.__name__,
        len(pickled) / 1024,
        len(pip_dependencies),
    )
    return Bundle(pickled=pickled, metadata=metadata)


def load_bundle(
    bundle: Bundle,
    *,
    instantiate: bool = True,
) -> BaseEnv | tuple[type[BaseEnv], dict[str, Any]]:
    """Unpickle and (optionally) instantiate.

    Always verifies the bundle's ``python_version`` matches the current
    interpreter. Never installs pip dependencies — the caller's image must
    already have them.

    Args:
        bundle: The Bundle to load.
        instantiate: If True (default), returns ``env_class(**constructor_args)``.
            If False, returns ``(env_class, constructor_args)``.

    Raises:
        IncompatiblePythonError: Bundle's python_version != current interpreter.
        BundlingError: Pickle bytes are corrupt, or the unpickled object isn't
            a (env_class, constructor_args) tuple with a BaseEnv subclass.
    """
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if bundle.metadata.python_version != current:
        raise IncompatiblePythonError(
            f"Bundle was packaged with Python {bundle.metadata.python_version} "
            f"but this machine runs Python {current}."
        )

    try:
        payload = cloudpickle.loads(bundle.pickled)
    except Exception as e:
        raise BundlingError(
            f"Failed to unpickle bundle: {e}. "
            "This usually means a dependency is missing or there's a "
            "Python version mismatch."
        ) from e

    if not (isinstance(payload, tuple) and len(payload) == 2):
        raise BundlingError(
            f"Unpickled payload is {type(payload).__name__}, expected "
            "(env_class, constructor_args) tuple. The bundle may be corrupt."
        )
    env_class, constructor_args = payload
    if not (isinstance(env_class, type) and issubclass(env_class, BaseEnv)):
        raise BundlingError(
            f"Unpickled class is {type(env_class).__name__}, not a BaseEnv subclass."
        )
    if not isinstance(constructor_args, dict):
        raise BundlingError(
            f"Unpickled constructor_args is {type(constructor_args).__name__}, expected dict."
        )

    if instantiate:
        logger.info("[bundle] instantiating %s", env_class.__name__)
        return env_class(**constructor_args)
    return env_class, constructor_args


def _check_env_class(env_class: type[BaseEnv]) -> None:
    if not (isinstance(env_class, type) and issubclass(env_class, BaseEnv)):
        raise BundlingError(
            f"{env_class!r} is not a BaseEnv subclass. "
            "Bundled classes must inherit from benchmax.envs.base_env.BaseEnv."
        )
    if env_class is BaseEnv:
        raise BundlingError("Cannot bundle BaseEnv directly. Provide a concrete subclass.")
    abstract = getattr(env_class, "__abstractmethods__", frozenset())
    if abstract:
        raise BundlingError(
            f"{env_class.__name__} has unimplemented abstract methods: "
            f"{', '.join(sorted(abstract))}"
        )


def _get_source(env_class: type[BaseEnv]) -> str | None:
    try:
        return inspect.getsource(env_class)
    except (OSError, TypeError) as e:
        logger.debug("[bundle] no source for %s: %s", env_class.__name__, e)
        return None


def _benchmax_version() -> str:
    try:
        from importlib.metadata import version

        return version("benchmax")
    except Exception:
        return "unknown"


def _ensure_safe_python_version() -> None:
    v = sys.version_info
    if (v.major, v.minor) == (3, 13):
        raise BundlingError(
            f"Python {v.major}.{v.minor}.{v.micro} is unsupported. "
            "Python 3.13.x has a pathlib.Path pickle incompatibility that "
            "breaks cross-version unpickling. Use Python 3.12 or >= 3.14."
        )


def unregistered_local_refs(pickled: bytes) -> list[str]:
    """Return modules this pickle would try to import that aren't installed.

    These are the modules the unpickler would attempt via ``find_class``;
    if none of them is a registered distribution or stdlib module, a fresh
    worker process will hit ``ModuleNotFoundError``. Exposed for callers
    (e.g. ``validate_env``) that want to surface the same check at a
    different layer.
    """
    return _unregistered_local_refs(pickled)


def _unregistered_local_refs(pickled: bytes) -> list[str]:
    return sorted(m for m in _referenced_modules(pickled) if not _looks_like_installed(m))


def _referenced_modules(pickled: bytes) -> set[str]:
    """Modules the unpickler would resolve via import on load.

    Hooks ``pickle.Unpickler.find_class`` to record every ``(module, name)``
    lookup the unpickler attempts — i.e. exactly the imports that'd fail
    with ``ModuleNotFoundError`` on a fresh interpreter. Returns a stub so
    unpickling proceeds past missing classes and we collect every ref.
    """
    refs: set[str] = set()

    class _Stub:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __call__(self, *a: Any, **kw: Any) -> "_Stub":
            return self

        def __reduce__(self) -> tuple:
            return (type(self), ())

    class _Recorder(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            refs.add(module)
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    try:
        _Recorder(io.BytesIO(pickled)).load()
    except Exception:
        # We only care which modules got looked up; later REDUCE failures
        # against stubs are expected and ignored.
        pass
    return refs


def _looks_like_installed(mod_name: str) -> bool:
    """Heuristic: True if a fresh interpreter could import this top-level module."""
    top = mod_name.split(".")[0]
    if top in sys.stdlib_module_names:
        return True
    try:
        distribution(top)
        return True
    except PackageNotFoundError:
        pass
    try:
        if packages_distributions().get(top):
            return True
    except Exception:
        pass
    return False
