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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmax.sft.normalize import normalize_row

Severity = Literal["error", "notice"]

# str.splitlines() also breaks on U+2028/U+2029 (and other unicode line
# separators), which are legal unescaped inside a JSON string value and can
# round-trip straight through canonical_jsonl's ensure_ascii=False. Split
# strictly on the actual on-disk line terminators instead.
_LINE_BOUNDARY = re.compile(r"\r\n|\n")


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


def _reject_non_finite(constant: str) -> Any:
    # json.loads accepts the non-standard NaN/Infinity/-Infinity tokens by
    # default (not valid JSON per RFC 8259); treat them as a decode failure
    # instead of silently admitting a value canonical_jsonl can't round-trip.
    raise ValueError(f"non-finite JSON constant {constant!r} is not allowed")


def load_sft_dataset(path: str | Path) -> SftDataset:
    """Read raw JSONL at ``path``, normalize every row, and retain provenance.

    Blank lines are skipped but still counted toward ``physical_line``. A
    line that isn't valid JSON, whose top-level value isn't a JSON object,
    or that contains a ``NaN``/``Infinity``/``-Infinity`` constant becomes
    an error :class:`SftIssue` instead of raising —
    :func:`benchmax.sft.validate.validate_sft_dataset` is where a malformed
    dataset surfaces to the caller.
    """
    file_path = Path(path)
    source_path = str(file_path)
    rows: list[SftRow] = []
    load_issues: list[SftIssue] = []

    text = file_path.read_text(encoding="utf-8")
    for physical_line, raw_line in enumerate(_LINE_BOUNDARY.split(text), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line, parse_constant=_reject_non_finite)
        except (ValueError, RecursionError) as exc:
            # deeply nested JSON overflows the parser stack (RecursionError, not a
            # ValueError) — surface it on the per-line issue path, never as a crash
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


def canonical_row_bytes(row_data: dict[str, Any]) -> bytes:
    """The exact bytes ``canonical_jsonl`` produces for one row.

    Shared with :func:`benchmax.sft.schema.validate_row` and
    :func:`benchmax.sft.validate.validate_sft_dataset` so "is this row
    serializable" is answered by the same code path that actually
    canonicalizes it — ``ensure_ascii=True`` (json.dumps' default) never
    attempts the UTF-8 encode step, so a schema check that only called
    ``json.dumps`` could pass a row containing a lone surrogate that then
    raises ``UnicodeEncodeError`` here. ``allow_nan=False`` rejects
    ``NaN``/``Infinity`` tokens that aren't valid JSON.
    """
    return json.dumps(row_data, ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_jsonl(dataset: SftDataset) -> bytes:
    """Render ``dataset``'s rows as canonical JSONL bytes — the only shape the upload path accepts."""
    return b"".join(canonical_row_bytes(row.data) + b"\n" for row in dataset.rows)
