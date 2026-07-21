from __future__ import annotations

import atexit
import hashlib
import importlib
import shutil
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harbor.models.trial.config import AgentConfig

__all__ = ["BundledAgentSource", "BundledHarborAgent"]

_materialization_lock = threading.RLock()
_materialization_root: Path | None = None
_materialized_sources: dict[str, Path] = {}


@dataclass(frozen=True, slots=True, repr=False)
class BundledAgentSource:
    """Immutable source and resource files for one custom Harbor agent.

    Paths are canonical POSIX paths relative to the agent source root. File
    contents are read eagerly so a serialized environment never depends on the
    authoring checkout at runtime.
    """

    files: tuple[tuple[str, bytes], ...]
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        normalized = _normalize_files(self.files)
        if not normalized:
            raise ValueError("bundled agent source must contain at least one file")
        object.__setattr__(self, "files", normalized)
        object.__setattr__(self, "content_id", _content_id(normalized))

    @classmethod
    def from_directory(
        cls,
        root: Path,
        *,
        files: Sequence[str | Path],
    ) -> BundledAgentSource:
        """Capture an explicit set of files beneath ``root`` immediately."""

        root = Path(root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"bundled agent source root is not a directory: {root}")

        captured: list[tuple[str, bytes]] = []
        for requested in files:
            relative = _safe_relative_path(requested)
            source = root
            for part in PurePosixPath(relative).parts:
                source /= part
                if source.is_symlink():
                    raise ValueError(
                        "bundled agent source paths cannot traverse symlinks: "
                        f"{relative!r}"
                    )
            if not source.is_file():
                raise ValueError(f"bundled agent source file is missing: {relative!r}")
            captured.append((relative, source.read_bytes()))
        return cls(tuple(captured))

    @classmethod
    def from_files(cls, files: Mapping[str, bytes]) -> BundledAgentSource:
        """Build a source tree from already-captured bytes."""

        return cls(tuple(files.items()))

    def __repr__(self) -> str:
        return (
            "BundledAgentSource("
            f"content_id={self.content_id!r}, files={len(self.files)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class BundledHarborAgent:
    """A normal Harbor custom-agent config with self-contained source files."""

    config: AgentConfig
    source: BundledAgentSource

    def __post_init__(self) -> None:
        from harbor.models.trial.config import AgentConfig

        if not isinstance(self.config, AgentConfig):
            raise TypeError(
                "bundled Harbor agent config must be Harbor AgentConfig, got "
                f"{type(self.config).__name__}"
            )
        if self.config.name is not None or self.config.import_path is None:
            raise ValueError(
                "bundled Harbor agents require an explicit import_path and no name"
            )
        _split_import_path(self.config.import_path)
        _require_entry_module(self.config.import_path, self.source)
        object.__setattr__(self, "config", self.config.model_copy(deep=True))

    def _harbor_config(self) -> AgentConfig:
        """Prepare the source and return Harbor's ordinary import-path config."""

        original = self.config
        resolved_import_path = _load_agent_import_path(
            self.source,
            original.import_path,
        )
        return original.model_copy(
            deep=True,
            update={
                "name": None,
                "import_path": resolved_import_path,
            },
        )

    def __repr__(self) -> str:
        return (
            "BundledHarborAgent("
            f"import_path={self.config.import_path!r}, "
            f"source={self.source.content_id!r})"
        )


def _normalize_files(
    files: Sequence[tuple[str, bytes]],
) -> tuple[tuple[str, bytes], ...]:
    normalized: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for path, content in files:
        safe_path = _safe_relative_path(path)
        if safe_path in seen:
            raise ValueError(f"duplicate bundled agent source path: {safe_path!r}")
        if not isinstance(content, bytes):
            raise TypeError(
                f"bundled agent source content for {safe_path!r} must be bytes"
            )
        seen.add(safe_path)
        normalized.append((safe_path, content))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _safe_relative_path(path: str | Path) -> str:
    raw = str(path)
    if not raw or "\\" in raw or "\0" in raw:
        raise ValueError(f"invalid bundled agent source path: {raw!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"unsafe bundled agent source path: {raw!r}")
    canonical = candidate.as_posix()
    if canonical != raw:
        raise ValueError(
            f"bundled agent source paths must be canonical POSIX paths: {raw!r}"
        )
    return canonical


def _content_id(files: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256(b"benchmax-bundled-harbor-agent-v1\0")
    for path, content in files:
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _split_import_path(import_path: str) -> tuple[str, str]:
    if import_path.count(":") != 1:
        raise ValueError("agent import_path must have the form 'module.path:ClassName'")
    module_name, class_name = import_path.split(":", 1)
    if not module_name or any(
        not part.isidentifier() for part in module_name.split(".")
    ):
        raise ValueError("agent import_path contains an invalid module name")
    if not class_name.isidentifier():
        raise ValueError("agent import_path contains an invalid class name")
    return module_name, class_name


def _require_entry_module(
    import_path: str,
    source: BundledAgentSource,
) -> None:
    module_name, _ = _split_import_path(import_path)
    module_path = module_name.replace(".", "/")
    possible_paths = {f"{module_path}.py", f"{module_path}/__init__.py"}
    available = {path for path, _ in source.files}
    if possible_paths.isdisjoint(available):
        raise ValueError(
            f"bundled agent entry module {module_name!r} is absent from its source"
        )


def _load_agent_import_path(source: BundledAgentSource, import_path: str) -> str:
    from harbor.agents.base import BaseAgent

    module_name, class_name = _split_import_path(import_path)
    root = _materialize_source(source)
    namespace = f"_benchmax_harbor_agent_{source.content_id.removeprefix('sha256:')}"
    qualified_module = f"{namespace}.{module_name}"

    with _materialization_lock:
        if namespace not in sys.modules:
            package = ModuleType(namespace)
            package.__package__ = namespace
            package.__path__ = [str(root)]  # type: ignore[attr-defined]
            package.__spec__ = ModuleSpec(namespace, loader=None, is_package=True)
            sys.modules[namespace] = package
        module = importlib.import_module(qualified_module)

    try:
        agent_class = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(
            f"bundled agent module {module_name!r} has no class {class_name!r}"
        ) from error
    if not isinstance(agent_class, type) or not issubclass(agent_class, BaseAgent):
        raise TypeError(
            f"bundled agent {import_path!r} must resolve to a Harbor BaseAgent subclass"
        )
    return f"{qualified_module}:{class_name}"


def _materialize_source(source: BundledAgentSource) -> Path:
    global _materialization_root

    with _materialization_lock:
        existing = _materialized_sources.get(source.content_id)
        if existing is not None:
            return existing

        if _materialization_root is None:
            _materialization_root = Path(
                tempfile.mkdtemp(prefix="benchmax-harbor-agents-")
            )
            atexit.register(_remove_materialization_root, _materialization_root)

        directory_name = source.content_id.removeprefix("sha256:")
        staging = _materialization_root / f".{directory_name}.staging"
        target = _materialization_root / directory_name
        staging.mkdir()
        try:
            for relative, content in source.files:
                destination = staging.joinpath(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        _make_read_only(target)
        _materialized_sources[source.content_id] = target
        return target


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _remove_materialization_root(root: Path) -> None:
    if not root.exists():
        return
    for path in (root, *root.rglob("*")):
        if path.is_dir():
            try:
                path.chmod(0o755)
            except OSError:
                pass
    shutil.rmtree(root, ignore_errors=True)
