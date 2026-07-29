from __future__ import annotations

import dis
import hashlib
import importlib
import inspect
import io
import json
import logging
import pickle
import site
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

import cloudpickle
from benchmax.envs.environment import Environment
from benchmax.envs.shared_types import RolloutAttempt
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

logger = logging.getLogger(__name__)


class BundlingError(Exception):
    """Cloudpickle serialization or class-shape failure."""


class IncompatibleRuntimeError(Exception):
    """Bundle metadata is incompatible with the current benchmax runtime."""


class IncompatiblePythonError(IncompatibleRuntimeError):
    """Loader's interpreter doesn't match the bundle's python_version."""


class IncompatibleBenchmaxError(IncompatibleRuntimeError):
    """Loader's benchmax major.minor doesn't match the bundle's benchmax_version."""


# register_pickle_by_value mutates process-global state; serialize against races.
_BUNDLE_LOCK = threading.Lock()

_METADATA_KEYS = frozenset(
    {"pip_dependencies", "python_version", "benchmax_version", "env_class_source"}
)


@dataclass(frozen=True)
class BundleMetadata:
    """Readable without unpickling: install deps, version checks, UI source."""

    pip_dependencies: tuple[str, ...]
    python_version: str
    benchmax_version: str
    env_class_source: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pip_dependencies",
            _normalize_pip_dependencies(self.pip_dependencies),
        )
        for name in ("python_version", "benchmax_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.env_class_source is not None and not isinstance(self.env_class_source, str):
            raise TypeError("env_class_source must be a string or None")

    def to_json_bytes(self) -> bytes:
        """Return canonical metadata bytes used by the artifact digest."""

        return json.dumps(
            {
                "pip_dependencies": self.pip_dependencies,
                "python_version": self.python_version,
                "benchmax_version": self.benchmax_version,
                "env_class_source": self.env_class_source,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> BundleMetadata:
        try:
            d = json.loads(data.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("bundle metadata must be a UTF-8 JSON object") from exc
        if not isinstance(d, dict):
            raise ValueError("bundle metadata must be a JSON object")
        unknown = sorted(set(d) - _METADATA_KEYS)
        if unknown:
            raise ValueError(
                f"bundle metadata has unsupported keys {unknown}; "
                "re-bundle with a current benchmax release"
            )
        try:
            pip_dependencies = d["pip_dependencies"]
            python_version = d["python_version"]
            benchmax_version = d["benchmax_version"]
        except KeyError as exc:
            raise ValueError(f"bundle metadata is missing {exc.args[0]!r}") from exc
        return cls(
            pip_dependencies=pip_dependencies,
            python_version=python_version,
            benchmax_version=benchmax_version,
            env_class_source=d.get("env_class_source"),
        )


@dataclass(frozen=True)
class Bundle:
    """Serialized env. ``pickled`` is ``cloudpickle.dumps((env_class, constructor_args))``."""

    pickled: bytes
    metadata: BundleMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.pickled, bytes):
            raise TypeError("bundle pickled payload must be bytes")
        if not isinstance(self.metadata, BundleMetadata):
            raise TypeError("bundle metadata must be BundleMetadata")


_BUNDLE_DIGEST_DOMAIN = b"benchmax.bundle.v1\0"


def bundle_digest(bundle: Bundle) -> str:
    """Return the SHA-256 identity of the complete deployable artifact.

    The identity covers the exact pickle and the canonical metadata bytes.
    Length prefixes keep the components unambiguous, and the domain marker
    leaves room for future artifact formats without silently reusing hashes.
    """

    if not isinstance(bundle, Bundle):
        raise TypeError("bundle_digest requires a Bundle")
    digest = hashlib.sha256(_BUNDLE_DIGEST_DOMAIN)
    for component in (bundle.pickled, bundle.metadata.to_json_bytes()):
        digest.update(len(component).to_bytes(8, byteorder="big"))
        digest.update(component)
    return digest.hexdigest()


def validate_bundle_compatibility(metadata: BundleMetadata) -> None:
    """Reject metadata built for a different Python or benchmax runtime.

    This check never reads the pickle or installs environment dependencies, so
    execution runtimes can call it before performing either higher-risk step.
    """

    if not isinstance(metadata, BundleMetadata):
        raise TypeError("bundle compatibility requires BundleMetadata")

    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if metadata.python_version != current_python:
        raise IncompatiblePythonError(
            f"Bundle was packaged with Python {metadata.python_version} "
            f"but this machine runs Python {current_python}."
        )

    current_benchmax = _benchmax_version()
    if metadata.benchmax_version == "unknown" or current_benchmax == "unknown":
        raise IncompatibleBenchmaxError(
            "Cannot verify benchmax compatibility because the bundle or "
            "runtime version is unknown. Install benchmax as a versioned package."
        )
    bundle_series = _version_major_minor(metadata.benchmax_version)
    current_series = _version_major_minor(current_benchmax)
    if bundle_series is None or current_series is None:
        raise IncompatibleBenchmaxError(
            f"Cannot parse benchmax versions (bundle {metadata.benchmax_version}, "
            f"runtime {current_benchmax}); expected major.minor[.patch]."
        )
    if bundle_series != current_series:
        raise IncompatibleBenchmaxError(
            f"Bundle was packaged with benchmax {metadata.benchmax_version} "
            f"but this runtime uses benchmax {current_benchmax}; "
            "major.minor versions must match."
        )


def dump_bundle(
    env_class: type[Environment[Any, RolloutAttempt]],
    *,
    constructor_args: dict[str, Any] | None = None,
    pip_dependencies: Sequence[str] | None = None,
    local_modules: list[ModuleType] | None = None,
    env_class_source: str | None = None,
    auto_local_modules: bool = True,
) -> Bundle:
    """Pickle ``(env_class, constructor_args)`` and stamp metadata.

    Args:
        env_class: A concrete Environment implementation.
        constructor_args: kwargs for ``env_class(**...)`` on load.
        pip_dependencies: Recorded in metadata. NOT installed by this call.
        local_modules: Additional modules to pickle by value. Use this for
            local source outside the environment's own Python project. Such
            source otherwise fails loudly unless its installed distribution is
            named in ``pip_dependencies``.
        env_class_source: Override for the recorded source. Pass this when the
            caller already holds the source and ``inspect.getsource`` can't
            recover it — e.g. a class produced by ``exec()`` into an in-memory
            namespace, which has no source file on disk. When ``None``
            (default), source is introspected from ``env_class``.
        auto_local_modules: When True (default), referenced modules whose
            source belongs to the environment's nearest ``pyproject.toml`` are
            imported and pickled by value automatically (a warning names
            them). When False, such a reference raises ``BundlingError``.

    Raises:
        BundlingError: bad env_class, cloudpickle failure, or uncaptured
            local source references.
    """
    _ensure_safe_python_version()
    _check_env_class(env_class)

    constructor_args = constructor_args or {}
    try:
        normalized_pip_dependencies = _normalize_pip_dependencies(
            () if pip_dependencies is None else pip_dependencies
        )
    except (TypeError, ValueError) as exc:
        raise BundlingError(f"Invalid pip_dependencies: {exc}") from exc
    benchmax_version = _benchmax_version()
    if benchmax_version == "unknown":
        raise BundlingError(
            "Cannot determine the benchmax package version; install benchmax as "
            "a versioned package before creating a bundle"
        )
    local_modules = local_modules or []
    captured_modules = list(local_modules)
    project_roots = _bundle_project_roots(env_class, local_modules)

    with _BUNDLE_LOCK:
        for mod in local_modules:
            if not isinstance(mod, ModuleType):
                raise BundlingError(
                    f"local_modules must contain module objects, got {type(mod).__name__}: {mod!r}"
                )
            cloudpickle.register_pickle_by_value(mod)
        try:
            try:
                pickled = cloudpickle.dumps((env_class, constructor_args))
            except Exception as e:
                raise BundlingError(f"Failed to serialize {env_class.__name__}: {e}") from e
        finally:
            for mod in local_modules:
                try:
                    cloudpickle.unregister_pickle_by_value(mod)
                except Exception:
                    pass

    if auto_local_modules and _unregistered_local_refs(pickled, project_roots):
        # Import each referenced local module and re-dump with it pickled by
        # value. Loop because a by-value module can surface further local refs;
        # registrations accumulate (and are torn down once at the end) so an
        # earlier module stays by-value while we resolve the ones it pulled in.
        seen: set[str] = {m.__name__ for m in local_modules}
        registered: list[ModuleType] = []
        with _BUNDLE_LOCK:
            try:
                # Explicit modules must remain by-value across every recursive
                # re-dump, not only the initial pickle above.
                for mod in local_modules:
                    cloudpickle.register_pickle_by_value(mod)
                    registered.append(mod)
                for _ in range(10):
                    pending = [
                        m for m in _unregistered_local_refs(pickled, project_roots) if m not in seen
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
                        captured_modules.append(mod)
                    pickled = cloudpickle.dumps((env_class, constructor_args))
            finally:
                for mod in reversed(registered):
                    try:
                        cloudpickle.unregister_pickle_by_value(mod)
                    except Exception:
                        pass

    risky = _unregistered_local_refs(pickled, project_roots)
    if risky:
        msg = f"{env_class.__name__}: missing local_modules=[{', '.join(risky)}]"
        if local_modules:
            already = ", ".join(sorted(m.__name__ for m in local_modules))
            msg += f" (already registered: [{already}])"
        raise BundlingError(msg)

    undeclared = _undeclared_external_source_refs(
        pickled,
        project_roots,
        normalized_pip_dependencies,
    )
    if undeclared:
        raise BundlingError(
            f"{env_class.__name__}: referenced source outside the environment "
            f"project: {', '.join(undeclared)}. Pass the module object in "
            "local_modules to capture it, or declare its distribution in "
            "pip_dependencies to keep it as a remote dependency."
        )

    delayed_source_imports = _unsafe_delayed_source_imports(
        env_class,
        captured_modules,
        project_roots,
        normalized_pip_dependencies,
    )
    if delayed_source_imports:
        raise BundlingError(
            f"{env_class.__name__}: delayed import of local source cannot be "
            f"captured safely: {', '.join(delayed_source_imports)}. Move the "
            "import to module scope so cloudpickle can capture the reference, "
            "or install the module remotely and declare its distribution in "
            "pip_dependencies. Passing it in local_modules does not satisfy a "
            "later import statement."
        )

    metadata = BundleMetadata(
        pip_dependencies=normalized_pip_dependencies,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        benchmax_version=benchmax_version,
        env_class_source=(
            env_class_source if env_class_source is not None else _get_source(env_class)
        ),
    )

    logger.info(
        "[bundle] dumped %s: %.1f KB pickled, %d pip deps",
        env_class.__name__,
        len(pickled) / 1024,
        len(normalized_pip_dependencies),
    )
    return Bundle(pickled=pickled, metadata=metadata)


def load_bundle(
    bundle: Bundle,
    *,
    instantiate: bool = True,
) -> (
    Environment[Any, RolloutAttempt] | tuple[type[Environment[Any, RolloutAttempt]], dict[str, Any]]
):
    """Unpickle and (optionally) instantiate.

    Verifies ``python_version`` matches exactly and ``benchmax_version`` shares
    the runtime's major.minor. Never installs pip deps — image must.

    Args:
        bundle: The Bundle to load.
        instantiate: If True (default), return ``env_class(**constructor_args)``.
            If False, return ``(env_class, constructor_args)``.

    Raises:
        IncompatibleRuntimeError: bundle's Python or benchmax version differs.
        BundlingError: corrupt bytes or a class that does not implement Environment.
    """
    validate_bundle_compatibility(bundle.metadata)

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
    if not (isinstance(env_class, type) and issubclass(env_class, Environment)):
        raise BundlingError(
            f"Unpickled class is {type(env_class).__name__}, not an Environment implementation."
        )
    if not isinstance(constructor_args, dict):
        raise BundlingError(
            f"Unpickled constructor_args is {type(constructor_args).__name__}, expected dict."
        )

    if instantiate:
        logger.info("[bundle] instantiating %s", env_class.__name__)
        return env_class(**constructor_args)
    return env_class, constructor_args


def _check_env_class(
    env_class: type[Environment[Any, RolloutAttempt]],
) -> None:
    if not (isinstance(env_class, type) and issubclass(env_class, Environment)):
        raise BundlingError(f"{env_class!r} does not implement Environment.")
    if env_class is Environment:
        raise BundlingError(
            "Cannot bundle the Environment protocol; provide a concrete implementation."
        )
    abstract = getattr(env_class, "__abstractmethods__", frozenset())
    if abstract:
        raise BundlingError(
            f"{env_class.__name__} has unimplemented abstract methods: "
            f"{', '.join(sorted(abstract))}"
        )


def _get_source(
    env_class: type[Environment[Any, RolloutAttempt]],
) -> str | None:
    try:
        return inspect.getsource(env_class)
    except (OSError, TypeError) as e:
        logger.debug("[bundle] no source for %s: %s", env_class.__name__, e)
        return None


def _benchmax_version() -> str:
    from importlib.metadata import version

    try:
        return version("benchmax")
    except Exception:
        return "unknown"


def _version_major_minor(value: str) -> tuple[int, int] | None:
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def _ensure_safe_python_version() -> None:
    # 3.13 has a pathlib.Path pickle incompatibility that breaks cross-version unpickling.
    v = sys.version_info
    if (v.major, v.minor) == (3, 13):
        raise BundlingError(
            f"Python {v.major}.{v.minor}.{v.micro} is unsupported. Use Python 3.12 or >= 3.14."
        )


def unregistered_local_refs(pickled: bytes) -> list[str]:
    """Project-local modules referenced by ``pickled``.

    This inspection-only helper infers candidate project roots from the
    referenced modules themselves. ``dump_bundle`` has the stronger signal of
    the environment class's project root and uses that directly.
    """

    refs = _referenced_modules(pickled)
    project_roots = tuple(
        root for name in refs if (root := _project_root_for_module_name(name)) is not None
    )
    return _unregistered_local_refs(pickled, project_roots)


def _unregistered_local_refs(
    pickled: bytes,
    project_roots: tuple[Path, ...],
) -> list[str]:
    return sorted(
        module_name
        for module_name in _referenced_modules(pickled)
        if _module_is_project_local(module_name, project_roots)
    )


def _undeclared_external_source_refs(
    pickled: bytes,
    project_roots: tuple[Path, ...],
    pip_dependencies: Sequence[str],
) -> list[str]:
    """Local source outside the env project must have an explicit owner.

    A referenced module beneath an environment root is handled by automatic
    source capture. A module installed in site-packages is already an ordinary
    remote dependency. Source from any other project is ambiguous: it is safe
    by reference only when its distribution appears in ``pip_dependencies``;
    otherwise the caller must capture it explicitly with ``local_modules``.
    """

    declared_distributions = _declared_distribution_names(pip_dependencies)
    return sorted(
        module_name
        for module_name in _referenced_modules(pickled)
        if _module_is_external_source(module_name, project_roots)
        and not _module_has_declared_distribution(
            module_name,
            declared_distributions,
        )
    )


def _unsafe_delayed_source_imports(
    env_class: type[Environment[Any, RolloutAttempt]],
    captured_modules: Sequence[ModuleType],
    project_roots: tuple[Path, ...],
    pip_dependencies: Sequence[str],
) -> list[str]:
    """Find ordinary import statements that source capture cannot satisfy.

    Cloudpickle can serialize an eagerly referenced module by value, but it
    does not register that reconstructed module in ``sys.modules``. An import
    executed later inside an environment method therefore still asks the
    remote interpreter for an installed module. Reject local source here
    instead of producing an artifact that fails only when that method runs.

    Literal calls to ``importlib.import_module`` and ``__import__`` are checked
    too. Dynamic names assembled at runtime are outside static inspection; they
    are runtime dependencies and must be declared in ``pip_dependencies``.
    """

    declared_distributions = _declared_distribution_names(pip_dependencies)
    imported_names = _delayed_import_names(env_class, captured_modules)
    unsafe: list[str] = []
    for module_name in imported_names:
        is_local_source = _module_is_project_local(module_name, project_roots)
        is_external_source = _module_is_external_source(module_name, project_roots)
        if not (is_local_source or is_external_source):
            continue
        if _module_has_declared_distribution(module_name, declared_distributions):
            continue
        unsafe.append(module_name)
    return sorted(set(unsafe))


def _delayed_import_names(
    env_class: type[Environment[Any, RolloutAttempt]],
    captured_modules: Sequence[ModuleType],
) -> set[str]:
    names: set[str] = set()
    seen_code: set[int] = set()
    seen_values: set[int] = set()
    captured_names = {module.__name__ for module in captured_modules}
    source_modules = {env_class.__module__, *captured_names}

    def inspect_value(value: object) -> None:
        if isinstance(value, (staticmethod, classmethod)):
            inspect_value(value.__func__)
            return
        if isinstance(value, property):
            for accessor in (value.fget, value.fset, value.fdel):
                if accessor is not None:
                    inspect_value(accessor)
            return
        value_id = id(value)
        if value_id in seen_values:
            return
        seen_values.add(value_id)
        if isinstance(value, ModuleType):
            if value.__name__ not in captured_names:
                return
            for member in vars(value).values():
                if getattr(member, "__module__", None) == value.__name__:
                    inspect_value(member)
            return
        if isinstance(value, type):
            if value.__module__ not in source_modules:
                return
            for member in vars(value).values():
                inspect_value(member)
            return
        code = getattr(value, "__code__", None)
        owner_module = getattr(value, "__module__", None)
        if not isinstance(code, CodeType) or owner_module not in source_modules:
            return

        globals_by_name = getattr(value, "__globals__", {})
        names.update(
            _imports_from_code(
                code,
                owner_module,
                seen_code,
            )
        )
        for global_name in _code_names(code):
            if global_name in globals_by_name:
                inspect_value(globals_by_name[global_name])
        for default in getattr(value, "__defaults__", ()) or ():
            inspect_value(default)
        for default in (getattr(value, "__kwdefaults__", None) or {}).values():
            inspect_value(default)
        for cell in getattr(value, "__closure__", ()) or ():
            try:
                inspect_value(cell.cell_contents)
            except ValueError:
                pass

    inspect_value(env_class)
    return names


def _imports_from_code(
    code: CodeType,
    owner_module: str,
    seen_code: set[int],
) -> set[str]:
    if id(code) in seen_code:
        return set()
    seen_code.add(id(code))

    names: set[str] = set()
    instructions = tuple(dis.get_instructions(code))
    names.update(_literal_dynamic_imports(instructions))
    for index, instruction in enumerate(instructions):
        if instruction.opname != "IMPORT_NAME" or not isinstance(instruction.argval, str):
            continue
        level = 0
        fromlist: object = None
        if index >= 2:
            level_instruction = instructions[index - 2]
            fromlist_instruction = instructions[index - 1]
            if level_instruction.opname == "LOAD_CONST" and isinstance(
                level_instruction.argval, int
            ):
                level = level_instruction.argval
            if fromlist_instruction.opname == "LOAD_CONST":
                fromlist = fromlist_instruction.argval
        resolved = _resolve_import_name(
            instruction.argval,
            owner_module=owner_module,
            level=level,
        )
        if resolved:
            names.add(resolved)
            if not instruction.argval and isinstance(fromlist, tuple):
                names.update(
                    f"{resolved}.{item}"
                    for item in fromlist
                    if isinstance(item, str) and item != "*"
                )

    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_imports_from_code(constant, owner_module, seen_code))
    return names


def _code_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_code_names(constant))
    return names


def _literal_dynamic_imports(
    instructions: Sequence[dis.Instruction],
) -> set[str]:
    """Recognize literal dynamic imports without pretending to solve inference."""

    imports: set[str] = set()
    boundary_opnames = {"CALL", "RETURN_VALUE", "YIELD_VALUE"}
    for call_index, instruction in enumerate(instructions):
        if instruction.opname != "CALL":
            continue
        start = call_index - 1
        while start >= 0 and instructions[start].opname not in boundary_opnames:
            start -= 1
        call_setup = instructions[start + 1 : call_index]
        uses_import_callable = any(
            (item.opname == "LOAD_GLOBAL" and item.argval in {"__import__", "import_module"})
            or (item.opname in {"LOAD_ATTR", "LOAD_METHOD"} and item.argval == "import_module")
            for item in call_setup
        )
        if not uses_import_callable:
            continue
        module_name = next(
            (
                item.argval
                for item in call_setup
                if item.opname == "LOAD_CONST" and isinstance(item.argval, str) and item.argval
            ),
            None,
        )
        if module_name is not None:
            imports.add(module_name)
    return imports


def _resolve_import_name(
    name: str,
    *,
    owner_module: str,
    level: int,
) -> str | None:
    if level == 0:
        return name or None
    owner = sys.modules.get(owner_module)
    package = getattr(owner, "__package__", None) if owner is not None else None
    if not package:
        package = owner_module.rpartition(".")[0]
    if not package:
        return None
    try:
        return importlib.util.resolve_name(f"{'.' * level}{name}", package)
    except (ImportError, ValueError):
        return None


def _declared_distribution_names(pip_dependencies: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for dependency in pip_dependencies:
        try:
            names.add(canonicalize_name(Requirement(dependency).name))
        except InvalidRequirement:
            # Dependency installation remains the runtime's responsibility.
            # An invalid declaration simply cannot authorize an ambiguous
            # outside-project source reference here.
            continue
    return names


def _normalize_pip_dependencies(dependencies: Sequence[str]) -> tuple[str, ...]:
    """Validate and canonicalize an order-independent PEP 508 collection."""

    if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
        raise TypeError("pip dependencies must be a sequence of PEP 508 strings")

    normalized: list[str] = []
    declared_targets: dict[str, int] = {}
    for index, value in enumerate(dependencies):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"dependency at index {index} must be a non-empty string")
        try:
            requirement = Requirement(value.strip())
        except InvalidRequirement as exc:
            raise ValueError(
                f"dependency at index {index} is not valid PEP 508: {value!r}"
            ) from exc
        target = canonicalize_name(requirement.name)
        previous_index = declared_targets.get(target)
        if previous_index is not None:
            raise ValueError(
                f"dependency target {target!r} is declared more than once "
                f"(indexes {previous_index} and {index}); combine constraints and "
                "extras into one declaration"
            )
        declared_targets[target] = index
        normalized.append(_canonical_requirement(requirement))
    return tuple(sorted(normalized))


def _canonical_requirement(requirement: Requirement) -> str:
    """Render a parsed requirement with normalized names and stable ordering."""

    value = canonicalize_name(requirement.name)
    if requirement.extras:
        extras = sorted(canonicalize_name(extra) for extra in requirement.extras)
        value += f"[{','.join(extras)}]"
    if requirement.url:
        value += f" @ {requirement.url}"
    else:
        value += str(requirement.specifier)
    if requirement.marker is not None:
        value += f"; {requirement.marker}"
    return value


def _module_has_declared_distribution(
    module_name: str,
    declared_distributions: set[str],
) -> bool:
    top_level = module_name.partition(".")[0]
    distributions = importlib_metadata.packages_distributions().get(top_level, ())
    return any(
        canonicalize_name(distribution) in declared_distributions for distribution in distributions
    )


def _module_is_external_source(
    module_name: str,
    project_roots: tuple[Path, ...],
) -> bool:
    top_level = module_name.partition(".")[0]
    if top_level in sys.stdlib_module_names or top_level == "benchmax":
        return False

    module = sys.modules.get(module_name)
    if module is None:
        try:
            spec = importlib.util.find_spec(module_name)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            return False
        if spec is None:
            return False
        module = _module_from_spec(module_name, spec)

    paths = _module_source_paths(module)
    return bool(paths) and any(
        not _is_site_package_path(path)
        and not any(path.is_relative_to(root) for root in project_roots)
        for path in paths
    )


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

        def __call__(self, *a: Any, **kw: Any) -> _Stub:
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


def _bundle_project_roots(
    env_class: type[Environment[Any, RolloutAttempt]],
    local_modules: list[ModuleType],
) -> tuple[Path, ...]:
    """Return source-project roots that belong to this environment bundle."""

    modules = [sys.modules.get(env_class.__module__), *local_modules]
    roots = {
        root
        for module in modules
        if module is not None
        if (root := _project_root_for_module(module)) is not None
    }
    return tuple(sorted(roots))


def _project_root_for_module(module: ModuleType) -> Path | None:
    for path in _module_source_paths(module):
        if _is_site_package_path(path):
            continue
        if root := _nearest_project_root(path):
            return root
    return None


def _project_root_for_module_name(module_name: str) -> Path | None:
    module = sys.modules.get(module_name)
    if module is not None:
        return _project_root_for_module(module)

    try:
        spec = importlib.util.find_spec(module_name)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    return _project_root_for_module(_module_from_spec(module_name, spec))


def _module_is_project_local(
    module_name: str,
    project_roots: tuple[Path, ...],
) -> bool:
    """Whether a referenced module is source owned by the env's project."""

    top_level = module_name.partition(".")[0]
    if top_level in sys.stdlib_module_names or top_level == "benchmax":
        return False
    if not project_roots:
        return False

    module = sys.modules.get(module_name)
    if module is None:
        try:
            spec = importlib.util.find_spec(module_name)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            return False
        if spec is None:
            return False
        module = _module_from_spec(module_name, spec)

    return any(
        not _is_site_package_path(path) and any(path.is_relative_to(root) for root in project_roots)
        for path in _module_source_paths(module)
    )


def _module_from_spec(module_name: str, spec: Any) -> ModuleType:
    """Build a path-only module view without executing its loader."""

    module = ModuleType(module_name)
    module.__spec__ = spec
    module.__file__ = spec.origin
    module.__path__ = spec.submodule_search_locations
    return module


def _module_source_paths(module: ModuleType) -> tuple[Path, ...]:
    candidates: list[str] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        candidates.append(module_file)

    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        candidates.append(origin)

    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is not None:
        candidates.extend(str(location) for location in search_locations)

    paths: list[Path] = []
    for candidate in candidates:
        try:
            path = Path(candidate).resolve()
        except (OSError, RuntimeError):
            continue
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _nearest_project_root(path: Path) -> Path | None:
    directory = path if path.is_dir() else path.parent
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _is_site_package_path(path: Path) -> bool:
    for candidate in (*site.getsitepackages(), site.getusersitepackages()):
        try:
            if path.is_relative_to(Path(candidate).resolve()):
                return True
        except (OSError, RuntimeError):
            continue
    return False
