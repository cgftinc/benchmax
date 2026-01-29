import json
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

MAGIC = b"BMXP"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class EnvPayload:
    """Self-contained serialized environment class with metadata.

    The payload bundles a cloudpickle-serialized BaseEnv subclass with
    metadata needed to reconstruct it on a remote machine (pip dependencies,
    python version, etc.). Constructor args are NOT included — they are
    provided separately at instantiation time so the same payload can be
    reused with different configurations.
    """

    pickled_class: bytes
    pip_dependencies: List[str]
    python_version: str
    benchmax_version: str
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize to a portable binary format.

        Layout: MAGIC (4B) | version (2B) | metadata_len (4B) | metadata_json | pickled_class
        """
        metadata = {
            "pip_dependencies": self.pip_dependencies,
            "python_version": self.python_version,
            "benchmax_version": self.benchmax_version,
            "extra_metadata": self.extra_metadata,
        }
        meta_bytes = json.dumps(metadata).encode("utf-8")
        header = (
            MAGIC
            + struct.pack("!H", FORMAT_VERSION)
            + struct.pack("!I", len(meta_bytes))
        )
        return header + meta_bytes + self.pickled_class

    @classmethod
    def from_bytes(cls, data: bytes) -> "EnvPayload":
        """Deserialize from binary format. Does NOT unpickle the class."""
        if len(data) < 10:
            raise ValueError("Invalid payload: too short")
        if data[:4] != MAGIC:
            raise ValueError("Invalid payload: bad magic bytes")
        version = struct.unpack("!H", data[4:6])[0]
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported payload format version: {version}")
        meta_len = struct.unpack("!I", data[6:10])[0]
        if len(data) < 10 + meta_len:
            raise ValueError("Invalid payload: truncated metadata")
        meta_bytes = data[10 : 10 + meta_len]
        pickled_class = data[10 + meta_len :]
        metadata = json.loads(meta_bytes.decode("utf-8"))
        return cls(
            pickled_class=pickled_class,
            pip_dependencies=metadata["pip_dependencies"],
            python_version=metadata["python_version"],
            benchmax_version=metadata["benchmax_version"],
            extra_metadata=metadata.get("extra_metadata", {}),
        )
