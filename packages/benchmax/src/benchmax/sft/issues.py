"""Diagnostics for `benchmax-sft-v1` dataset validation.

Validation is all-or-nothing: constructing an :class:`~benchmax.sft.SftDataset`
either succeeds completely or raises :class:`SftDatasetError` carrying every
:class:`SftIssue` found, in deterministic document order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["SftDatasetError", "SftIssue"]

_PREVIEW_ISSUES = 20


@dataclass(frozen=True, slots=True)
class SftIssue:
    """One validation problem at one location in the input.

    ``location`` addresses the offending value inside the row using ``$`` for
    the row object itself, e.g. ``$.messages[2].weight``. ``line`` is the
    1-based physical JSONL line for :meth:`SftDataset.from_jsonl` input and
    ``None`` for :meth:`SftDataset.from_rows` input. ``row`` is the 0-based
    dataset row index, or ``None`` for dataset-level problems such as an empty
    input.
    """

    message: str
    location: str
    line: int | None = None
    row: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON-friendly shape used by golden fixtures."""

        return {
            "line": self.line,
            "location": self.location,
            "message": self.message,
            "row": self.row,
        }

    def describe(self) -> str:
        """Render a one-line human-readable description."""

        origin: list[str] = []
        if self.line is not None:
            origin.append(f"line {self.line}")
        if self.row is not None:
            origin.append(f"row {self.row}")
        prefix = f"{' '.join(origin)} " if origin else ""
        return f"{prefix}{self.location}: {self.message}"


class SftDatasetError(ValueError):
    """Raised when input cannot form a valid `benchmax-sft-v1` dataset.

    ``issues`` holds every problem found, ordered by input position and then by
    a fixed within-row validation order, so equivalent inputs always produce an
    identical issue sequence.
    """

    def __init__(self, issues: Sequence[SftIssue]) -> None:
        self.issues: tuple[SftIssue, ...] = tuple(issues)
        if not self.issues:
            raise ValueError("SftDatasetError requires at least one issue")
        lines = [f"invalid benchmax-sft-v1 dataset: {len(self.issues)} issue(s)"]
        lines += [f"  {issue.describe()}" for issue in self.issues[:_PREVIEW_ISSUES]]
        if len(self.issues) > _PREVIEW_ISSUES:
            lines.append(f"  ... and {len(self.issues) - _PREVIEW_ISSUES} more")
        super().__init__("\n".join(lines))
