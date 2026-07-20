"""The SFT dataset canonicalization boundary.

:func:`load_sft_dataset` is the only JSONL reader for SFT data — it reads
raw lines itself (not ``cli/_project.py``'s ``_load_jsonl``, which drops
blank lines, reindexes, and raises before any report could exist), retains
per-row ``(source_path, physical_line)`` provenance, and normalizes every
row on load so callers never see a legacy shape.
:func:`canonical_jsonl` is the only serializer for the upload path — no path
exists where un-normalized rows reach storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmax.sft.normalize import normalize_row

Severity = Literal["error", "notice"]


@dataclass(frozen=True)
class SftIssue:
    """A single dataset-loading or validation issue.

    ``physical_line`` is 1-indexed and counts blank lines, matching the
    on-disk file exactly. ``physical_line == 0`` marks a dataset-level
    issue with no single source line (e.g. an empty dataset).
    """

    source_path: str
    physical_line: int
    severity: Severity
    message: str


@dataclass(frozen=True)
class SftRow:
    """One canonicalized dataset row plus its on-disk provenance."""

    source_path: str
    physical_line: int
    data: dict[str, Any]


@dataclass(frozen=True)
class SftDataset:
    """A loaded, canonicalized SFT dataset: rows plus per-line load issues."""

    path: str
    rows: list[SftRow]
    load_issues: list[SftIssue]


def load_sft_dataset(path: str | Path) -> SftDataset:
    """Read raw JSONL at ``path``, normalize every row, and retain provenance.

    Blank lines are skipped but still counted toward ``physical_line``. A
    line that isn't valid JSON, or whose top-level value isn't a JSON
    object, becomes an error :class:`SftIssue` instead of raising —
    :func:`benchmax.sft.validate.validate_sft_dataset` is where a malformed
    dataset surfaces to the caller.
    """
    file_path = Path(path)
    source_path = str(file_path)
    rows: list[SftRow] = []
    load_issues: list[SftIssue] = []

    text = file_path.read_text(encoding="utf-8")
    for physical_line, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            load_issues.append(SftIssue(source_path, physical_line, "error", f"invalid JSON: {exc}"))
            continue
        if not isinstance(parsed, dict):
            load_issues.append(
                SftIssue(
                    source_path,
                    physical_line,
                    "error",
                    f"row must be a JSON object, got {type(parsed).__name__}",
                )
            )
            continue
        rows.append(SftRow(source_path, physical_line, normalize_row(parsed)))

    return SftDataset(path=source_path, rows=rows, load_issues=load_issues)


def canonical_jsonl(dataset: SftDataset) -> bytes:
    """Render ``dataset``'s rows as canonical JSONL bytes — the only shape the upload path accepts."""
    return "".join(
        json.dumps(row.data, ensure_ascii=False) + "\n" for row in dataset.rows
    ).encode("utf-8")
