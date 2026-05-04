"""Shared checkpoint I/O for long-running pipelines.

Provides atomic manifest writes, crash-safe JSONL append, and
truncation recovery.  Subclassed by ``qa_generation.CheckpointManager``
and ``traces.PivotCheckpointManager``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointBase:
    """Base class for checkpoint I/O with crash-safety guarantees.

    Guarantees:
    1. Atomic manifest: ``tmp`` + ``json.dump`` + ``os.replace``
    2. Crash-safe JSONL: ``flush()`` + ``os.fsync()`` after append
    3. Truncation recovery: corrupted trailing lines are skipped on load
    4. ``mkdir(parents=True, exist_ok=True)`` before any write
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir

    @property
    def manifest_path(self) -> Path:
        return self.checkpoint_dir / "manifest.json"

    def _ensure_dir(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        """Load ``manifest.json``.  Returns ``None`` if missing."""
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def save_manifest(self, data: dict[str, Any]) -> None:
        """Write ``manifest.json`` atomically."""
        self._ensure_dir()
        tmp = self.manifest_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, self.manifest_path)

    def append_jsonl(self, filename: str, items: list[dict[str, Any]]) -> None:
        """Append dicts as JSON lines.  ``flush`` + ``fsync`` after write."""
        self._ensure_dir()
        path = self.checkpoint_dir / filename
        with path.open("a", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        """Load JSON lines from a file.  Skips corrupted trailing lines."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupted checkpoint line in %s", filename)
                    break  # truncated last line from crash
        return items

    def cleanup(self) -> None:
        """Remove checkpoint directory."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            logger.info("Cleaned up checkpoints at %s", self.checkpoint_dir)
