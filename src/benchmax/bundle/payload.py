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
    format_version: int = FORMAT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pip_dependencies": self.pip_dependencies,
            "python_version": self.python_version,
            "benchmax_version": self.benchmax_version,
            "constructor_args": self.constructor_args,
            "format_version": self.format_version,
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleMetadata":
        return cls(
            pip_dependencies=list(data.get("pip_dependencies", [])),
            python_version=data.get("python_version", "unknown"),
            benchmax_version=data.get("benchmax_version", "unknown"),
            constructor_args=data.get("constructor_args"),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "BundleMetadata":
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass(frozen=True)
class BundledEnv:
    """In-memory bundle containing pickled class bytes and metadata."""

    pickled_class: bytes
    metadata: BundleMetadata
