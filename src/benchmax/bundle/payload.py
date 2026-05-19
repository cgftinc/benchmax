import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

FORMAT_VERSION = 1


@dataclass(frozen=True)
class BundleMetadata:
    """JSON-serializable metadata for a bundled environment class."""

    pip_dependencies: List[str]
    python_version: str
    benchmax_version: str
    constructor_args: Optional[Dict[str, Any]] = None
    constructor_args_pickled: bool = False
    format_version: int = FORMAT_VERSION
    # Plaintext source for the env class's module + every register_pickle_by_value
    # module. Captured at bundle time via inspect.getsource so consumers (UIs,
    # JS frontends, reviewers) can display the env's code without unpickling —
    # which would otherwise need a Python runtime + matching deps + arbitrary
    # code execution.
    sources: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pip_dependencies": self.pip_dependencies,
            "python_version": self.python_version,
            "benchmax_version": self.benchmax_version,
            "constructor_args": self.constructor_args,
            "constructor_args_pickled": self.constructor_args_pickled,
            "format_version": self.format_version,
            "sources": self.sources,
        }

    def to_json_bytes(self) -> bytes:
        data = self.to_dict()
        # constructor_args may contain non-JSON-serializable objects
        # (e.g. SearchClient instances). Null them out and set the flag
        # so the loader knows to look for the separate pickled args file.
        if self.has_pickled_constructor_args():
            data["constructor_args"] = None
            data["constructor_args_pickled"] = True
        return json.dumps(data).encode("utf-8")

    def has_pickled_constructor_args(self) -> bool:
        """True if constructor_args need separate pickle serialization."""
        if self.constructor_args is None:
            return False
        try:
            json.dumps(self.constructor_args)
            return False
        except TypeError:
            return True

    def pickled_constructor_args_bytes(self) -> bytes | None:
        """Pickle constructor_args if they aren't JSON-serializable."""
        if not self.has_pickled_constructor_args():
            return None
        import cloudpickle
        return cloudpickle.dumps(self.constructor_args)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleMetadata":
        return cls(
            pip_dependencies=list(data.get("pip_dependencies", [])),
            python_version=data.get("python_version", "unknown"),
            benchmax_version=data.get("benchmax_version", "unknown"),
            constructor_args=data.get("constructor_args"),
            constructor_args_pickled=bool(data.get("constructor_args_pickled", False)),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
            sources=data.get("sources"),
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "BundleMetadata":
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass(frozen=True)
class BundledEnv:
    """In-memory bundle containing pickled class bytes and metadata."""

    pickled_class: bytes
    metadata: BundleMetadata
    pickled_constructor_args: Optional[bytes] = None
