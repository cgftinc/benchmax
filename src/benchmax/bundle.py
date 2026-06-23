from __future__ import annotations

import importlib
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
    """Cloudpickle serialization or class-shape failure."""


class IncompatiblePythonError(Exception):
    """Loader's interpreter doesn't match the bundle's python_version."""


# register_pickle_by_value mutates process-global state; serialize against races.
_BUNDLE_LOCK = threading.Lock()


@dataclass(frozen=True)
class BundleMetadata:
    """Readable without unpickling: install deps, version checks, UI source."""

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
    """Serialized env. ``pickled`` is ``cloudpickle.dumps((env_class, constructor_args))``."""

    pickled: bytes
    metadata: BundleMetadata


def dump_bundle(
    env_class: type[BaseEnv],
    *,
    constructor_args: dict[str, Any] | None = None,
    pip_dependencies: list[str] | None = None,
    local_modules: list[ModuleType] | None = None,
    env_class_source: str | None = None,
    auto_local_modules: bool = True,
) -> Bundle:
    """Pickle ``(env_class, constructor_args)`` and stamp metadata.

    Args:
        env_class: A concrete BaseEnv subclass.
        constructor_args: kwargs for ``env_class(**...)`` on load.
        pip_dependencies: Recorded in metadata. NOT installed by this call.
        local_modules: Modules to pickle by-value. Required when the env class
            (or anything it references) lives in a local ``.py``.
        env_class_source: Override for the recorded source. Pass this when the
            caller already holds the source and ``inspect.getsource`` can't
            recover it — e.g. a class produced by ``exec()`` into an in-memory
            namespace, which has no source file on disk. When ``None``
            (default), source is introspected from ``env_class``.
        auto_local_modules: When True (default), any local module the pickle
            references but that wasn't passed in ``local_modules`` is imported
            and pickled by value automatically (a warning names them). When
            False, such a reference raises ``BundlingError`` instead.

    Raises:
        BundlingError: bad env_class, cloudpickle failure, or pickle references
            non-installed modules absent from ``local_modules``.
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
                    f"Failed to serialize {env_class.__name__}: {e}"
                ) from e
        finally:
            for mod in local_modules:
                try:
                    cloudpickle.unregister_pickle_by_value(mod)
                except Exception:
                    pass

    if auto_local_modules and _unregistered_local_refs(pickled):
        # Import each referenced local module and re-dump with it pickled by
        # value. Loop because a by-value module can surface further local refs;
        # registrations accumulate (and are torn down once at the end) so an
        # earlier module stays by-value while we resolve the ones it pulled in.
        seen: set[str] = {m.__name__ for m in local_modules}
        registered: list[ModuleType] = []
        with _BUNDLE_LOCK:
            try:
                for _ in range(10):
                    pending = [
                        m for m in _unregistered_local_refs(pickled) if m not in seen
                    ]
                    if not pending:
                        break
                    new_mods: list[ModuleType] = []
                    for name in pending:
                        seen.add(name)  # unimportable names fall through to the guard
                        try:
                            new_mods.append(importlib.import_module(name))
                        except Exception:
                            pass
                    if not new_mods:
                        break
                    logger.warning(
                        "[bundle] %s: auto-bundling local module(s): %s ",
                        env_class.__name__,
                        ", ".join(sorted(m.__name__ for m in new_mods)),
                    )
                    for mod in new_mods:
                        cloudpickle.register_pickle_by_value(mod)
                        registered.append(mod)
                    pickled = cloudpickle.dumps((env_class, constructor_args))
            finally:
                for mod in registered:
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
        env_class_source=(
            env_class_source
            if env_class_source is not None
            else _get_source(env_class)
        ),
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

    Verifies ``python_version`` matches. Never installs pip deps — image must.

    Args:
        bundle: The Bundle to load.
        instantiate: If True (default), return ``env_class(**constructor_args)``.
            If False, return ``(env_class, constructor_args)``.

    Raises:
        IncompatiblePythonError: bundle's python_version != current.
        BundlingError: corrupt bytes or non-BaseEnv payload.
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
        raise BundlingError(f"Failed to unpickle bundle: {e}") from e

    if not (isinstance(payload, tuple) and len(payload) == 2):
        raise BundlingError(
            f"Unpickled payload is {type(payload).__name__}, expected "
            "(env_class, constructor_args) tuple."
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
        raise BundlingError(f"{env_class!r} is not a BaseEnv subclass.")
    if env_class is BaseEnv:
        raise BundlingError("Cannot bundle BaseEnv directly; provide a concrete subclass.")
    abstract = getattr(env_class, "__abstractmethods__", frozenset())
    if abstract:
        raise BundlingError(
            f"{env_class.__name__} has unimplemented abstract methods: "
            f"{', '.join(sorted(abstract))}"
        )
    if env_class.run_rollouts is BaseEnv.run_rollouts:
        raise BundlingError(
            f"{env_class.__name__} must implement run_rollouts(). "
            "Stepped tool/reward envs should inherit ToolEnv."
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
    # 3.13 has a pathlib.Path pickle incompatibility that breaks cross-version unpickling.
    v = sys.version_info
    if (v.major, v.minor) == (3, 13):
        raise BundlingError(
            f"Python {v.major}.{v.minor}.{v.micro} is unsupported. "
            "Use Python 3.12 or >= 3.14."
        )


def unregistered_local_refs(pickled: bytes) -> list[str]:
    """Modules this pickle would import that aren't installed locally."""
    return _unregistered_local_refs(pickled)


def _unregistered_local_refs(pickled: bytes) -> list[str]:
    return sorted(m for m in _referenced_modules(pickled) if not _looks_like_installed(m))


def _referenced_modules(pickled: bytes) -> set[str]:
    # Hooks find_class so we see every (module, name) the unpickler would import —
    # i.e. exactly what'd raise ModuleNotFoundError on a fresh interpreter. The stub
    # lets unpickling proceed past missing classes so we collect every ref.
    #
    # find_class alone has a blind spot: a bare ``import foo`` that leaves a
    # module *object* in the env's globals is pickled as
    # ``cloudpickle.subimport("foo")`` — the module name is a REDUCE argument,
    # not a find_class path, so we'd only see ``cloudpickle.cloudpickle`` (which
    # looks installed) and miss ``foo``. We shim subimport to record its arg and
    # return a stub instead of importing, so a missing module is captured rather
    # than aborting the whole load early. (``dynamic_subimport`` is by-value /
    # self-contained — leave it to the real find_class so we don't flag it.)
    refs: set[str] = set()

    class _Stub:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __call__(self, *a: Any, **kw: Any) -> "_Stub":
            return self

        def __reduce__(self) -> tuple:
            return (type(self), ())

    def _recording_subimport(name: str, *a: Any, **kw: Any) -> ModuleType:
        refs.add(name)
        return ModuleType(str(name))

    def _noop_setstate(obj: Any, *a: Any, **kw: Any) -> Any:
        # cloudpickle's _make_skeleton_class resolves the class_tracker_id back
        # to the *live* class (it was tracked when env_class was dumped), so the
        # real ``_class_setstate``/``_function_setstate`` would setattr the
        # reconstructed (stub-globals) members onto the live class/function —
        # mutating the caller's class mid-bundle and poisoning any later dump.
        # We only need the refs from ``state``, which are already recorded while
        # it's unpickled; the setter itself is a no-op here.
        return obj

    class _Recorder(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            refs.add(module)
            if module.startswith("cloudpickle"):
                if name == "subimport":
                    return _recording_subimport
                if name in ("_class_setstate", "_function_setstate"):
                    return _noop_setstate
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    try:
        _Recorder(io.BytesIO(pickled)).load()
    except Exception:
        pass  # later REDUCE failures against stubs are expected
    return refs


def _looks_like_installed(mod_name: str) -> bool:
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
