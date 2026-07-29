"""The SFT dataset canonicalization boundary.

:func:`load_sft_dataset` is the only JSONL reader for SFT data — it reads
raw lines itself, retains per-row ``(source_path, physical_line)``
provenance, and normalizes every row on load so callers never see a legacy
shape. :func:`canonical_jsonl` is the only serializer for the upload path,
and it enforces the boundary rather than documenting it: a dataset carrying
any load issue or schema-invalid rows is refused, so no path exists where
un-normalized or partially-loaded rows reach storage — including for a
caller that builds an :class:`SftDataset` by hand and skips
:func:`benchmax.sft.validate.validate_sft_dataset` entirely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmax.sft.normalize import normalize_row
from benchmax.sft.schema import validate_row
from benchmax.sft.serialization import canonical_row_bytes

Severity = Literal["error", "notice"]

# str.splitlines() also breaks on U+2028/U+2029 (and other unicode line
# separators), which are legal unescaped inside a JSON string value and can
# round-trip straight through canonical_jsonl's ensure_ascii=False. Split
# strictly on the actual on-disk line terminators instead.
_LINE_BOUNDARY = re.compile(r"\r\n|\n")

# How many blocking issues SftSerializationError names before summarizing the
# rest — a refusal message stays readable on one screen.
_MAX_REPORTED_ISSUES = 5


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


class SftSerializationError(RuntimeError):
    """:func:`canonical_jsonl` was asked to serialize a dataset that has not
    cleared the load and schema gates.

    ``issues`` carries every blocking :class:`SftIssue`, each with its source
    path and physical line, so a caller can report exactly what to fix rather
    than re-deriving it. ``path`` is the dataset's own path. A ``RuntimeError``
    subclass so a CLI's top-level error handling prints it as one clean stderr
    line instead of a traceback.
    """

    def __init__(self, path: str, issues: list[SftIssue]) -> None:
        self.path = path
        self.issues = issues
        super().__init__(_refusal_message(path, issues))


def _refusal_message(path: str, issues: list[SftIssue]) -> str:
    shown = "; ".join(
        f"line {issue.physical_line}: {issue.message}" for issue in issues[:_MAX_REPORTED_ISSUES]
    )
    remaining = len(issues) - _MAX_REPORTED_ISSUES
    if remaining > 0:
        shown = f"{shown}; (+{remaining} more)"
    return (
        f"refusing to serialize {path!r}: {len(issues)} blocking issue(s) — "
        f"the dataset must load and validate cleanly first. {shown}"
    )


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
            load_issues.append(
                SftIssue(source_path, physical_line, "error", f"invalid JSON: {exc}")
            )
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


def _blocking_issues(dataset: SftDataset) -> list[SftIssue]:
    """Every issue that makes ``dataset`` unfit to serialize, in dataset order.

    Every entry of ``load_issues`` comes first, whatever its severity: a
    dataset that did not load cleanly must never upload as if it were whole,
    and severity says how loudly to report a load problem, not whether the
    result is trustworthy. Each row's structural schema errors follow. The
    notice-severity advisories :func:`benchmax.sft.validate.validate_sft_dataset`
    raises — size, token length, masking, empty eval — live on the validation
    report rather than on ``load_issues``, so they remain non-blocking.
    """
    blocking = list(dataset.load_issues)
    for row in dataset.rows:
        blocking.extend(
            SftIssue(row.source_path, row.physical_line, "error", message)
            for message in validate_row(row.data)
        )
    return blocking


def canonical_jsonl(dataset: SftDataset) -> bytes:
    """Render ``dataset``'s rows as canonical JSONL bytes — the only shape the upload path accepts.

    Raises :class:`SftSerializationError` when ``dataset`` carries any load
    issue at all, whatever its severity, or when any row fails
    :func:`benchmax.sft.schema.validate_row`, so partially-loaded or
    schema-invalid data cannot reach storage even when a caller skips
    :func:`benchmax.sft.validate.validate_sft_dataset`. This makes the
    canonicalize -> validate -> upload boundary enforced rather than
    documentary.

    An empty dataset renders as ``b""``: whether there is *enough* data to
    train on is a dataset-level policy owned by validation and by the upload
    path, not by canonicalization.
    """
    blocking = _blocking_issues(dataset)
    if blocking:
        raise SftSerializationError(dataset.path, blocking)
    return b"".join(canonical_row_bytes(row.data) + b"\n" for row in dataset.rows)
