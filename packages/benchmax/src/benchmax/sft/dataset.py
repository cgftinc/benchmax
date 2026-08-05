"""The validated `benchmax-sft-v1` dataset artifact.

:class:`SftDataset` is the single trust boundary for SFT training data: it is
either fully valid or it does not exist. Both constructors run the complete
strict contract — closed row schema, strict JSON rules, size limits — and raise
:class:`~benchmax.sft.issues.SftDatasetError` with every ordered issue on any
violation, including empty input.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from benchmax.sft.issues import SftDatasetError, SftIssue
from benchmax.sft.rows import MAX_ROW_BYTES, validate_row
from benchmax.sft.strict_json import (
    StrictJsonError,
    canonical_json_bytes,
    check_json_tree,
    parse_strict_json,
)

__all__ = ["SFT_DATASET_FORMAT", "SftDataset"]

SFT_DATASET_FORMAT = "benchmax-sft-v1"
"""Format identifier used at the platform boundary for this row contract."""

_INTERNAL = object()


class SftDataset:
    """Immutable, fully validated SFT training dataset.

    Construct only through :meth:`from_jsonl` or :meth:`from_rows`; direct
    instantiation is rejected so a partially valid dataset can never exist.
    Row order is preserved from the input. :meth:`to_jsonl_bytes` returns
    canonical bytes — UTF-8 without a BOM, ``ensure_ascii=False``, sorted keys,
    compact separators, one trailing newline per row — deterministic for
    equivalent input objects.
    """

    __slots__ = ("_row_bytes", "_rows")

    def __init__(
        self,
        rows: tuple[dict[str, object], ...],
        row_bytes: tuple[bytes, ...],
        token: object = None,
    ) -> None:
        if token is not _INTERNAL:
            raise TypeError("use SftDataset.from_jsonl(...) or SftDataset.from_rows(...)")
        self._rows = rows
        self._row_bytes = row_bytes

    @classmethod
    def from_jsonl(cls, path: str | os.PathLike[str]) -> SftDataset:
        """Load and validate a JSONL file.

        Whitespace-only physical lines are ignored while physical line numbers
        are preserved in diagnostics. Raises
        :class:`~benchmax.sft.issues.SftDatasetError` on any violation.
        """

        data = Path(path).read_bytes()
        collector = _IssueCollector()
        rows: list[dict[str, object]] = []
        row_bytes: list[bytes] = []
        row_index = 0
        for line_number, raw_line in enumerate(data.split(b"\n"), start=1):
            if raw_line.strip(b" \t\r\f\v") == b"":
                continue
            row = row_index
            row_index += 1
            if line_number == 1 and raw_line.startswith(b"\xef\xbb\xbf"):
                collector.add(line_number, row, "$", "a UTF-8 BOM is not allowed")
                continue
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                collector.add(line_number, row, "$", "line is not valid UTF-8")
                continue
            try:
                parsed = parse_strict_json(text)
            except StrictJsonError as error:
                collector.add(line_number, row, "$", str(error))
                continue
            encoded = _process_row(parsed, collector.for_row(line_number, row))
            if encoded is not None:
                rows.append(json.loads(encoded.decode("utf-8")))
                row_bytes.append(encoded)
        if row_index == 0:
            collector.add(None, None, "$", "dataset contains no rows")
        collector.raise_if_any()
        return cls(tuple(rows), tuple(row_bytes), _INTERNAL)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, object]]) -> SftDataset:
        """Validate in-memory row objects.

        Rows must already use plain JSON-compatible Python types; anything else
        is reported as an issue rather than coerced. Raises
        :class:`~benchmax.sft.issues.SftDatasetError` on any violation.
        """

        collector = _IssueCollector()
        stored: list[dict[str, object]] = []
        row_bytes: list[bytes] = []
        row_index = -1
        for row_index, row in enumerate(iter(rows)):
            encoded = _process_row(row, collector.for_row(None, row_index))
            if encoded is not None:
                # Round-trip through canonical bytes to decouple stored rows
                # from caller-owned objects.
                stored.append(json.loads(encoded.decode("utf-8")))
                row_bytes.append(encoded)
        if row_index < 0:
            collector.add(None, None, "$", "dataset contains no rows")
        collector.raise_if_any()
        return cls(tuple(stored), tuple(row_bytes), _INTERNAL)

    def to_jsonl_bytes(self) -> bytes:
        """Serialize every row as canonical JSONL bytes in original order."""

        return b"".join(encoded + b"\n" for encoded in self._row_bytes)

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        """The validated rows in original order. Treat as read-only."""

        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self._rows)


class _IssueCollector:
    """Ordered issue sink shared by both constructors."""

    def __init__(self) -> None:
        self._issues: list[SftIssue] = []

    def add(self, line: int | None, row: int | None, location: str, message: str) -> None:
        self._issues.append(SftIssue(message=message, location=location, line=line, row=row))

    def for_row(self, line: int | None, row: int | None) -> _RowSink:
        return _RowSink(self, line, row)

    def raise_if_any(self) -> None:
        if self._issues:
            raise SftDatasetError(self._issues)


class _RowSink:
    """Emit adapter binding one row's line/row coordinates."""

    def __init__(self, collector: _IssueCollector, line: int | None, row: int | None) -> None:
        self._collector = collector
        self.line = line
        self.row = row
        self.count = 0

    def __call__(self, location: str, message: str) -> None:
        self.count += 1
        self._collector.add(self.line, self.row, location, message)


def _process_row(row: object, sink: _RowSink) -> bytes | None:
    """Fully validate one row; return its canonical bytes only when clean."""

    check_json_tree(row, "$", sink)
    if type(row) is not dict:
        sink("$", "row must be a JSON object")
        return None
    validate_row(row, sink)
    if sink.count:
        return None
    encoded = canonical_json_bytes(row)
    if len(encoded) > MAX_ROW_BYTES:
        sink("$", f"row canonical size is {len(encoded)} bytes; the maximum is {MAX_ROW_BYTES}")
        return None
    return encoded
