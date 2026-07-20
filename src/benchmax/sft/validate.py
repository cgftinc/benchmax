"""Dataset-level SFT validation: :func:`validate_sft_dataset` gates a
train/eval pair before upload, mirroring ``ValidationReport``'s "nothing
validated is not a pass" rule for the SFT pathway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmax.envs.base.content import message_text
from benchmax.sft.dataset import SftDataset, SftIssue, canonical_row_bytes
from benchmax.sft.schema import validate_row

DEFAULT_MAX_SEQ_LEN = 8192
DEFAULT_MAX_ROW_BYTES = 1024 * 1024  # 1 MiB

# The VALIDATE_CONFIG / --config keys validate_sft_dataset actually accepts as
# kwargs — shared so a project's config dict is filtered identically everywhere
# it's read (the scaffold's own validate()/launch(), and `castform validate`).
VALIDATE_CONFIG_KEYS = ("max_seq_len", "max_row_bytes")


def sft_validate_kwargs(config: dict) -> dict:
    """``config`` (e.g. a project's ``VALIDATE_CONFIG``) filtered down to the
    kwargs :func:`validate_sft_dataset` accepts."""
    return {k: v for k, v in config.items() if k in VALIDATE_CONFIG_KEYS}


@dataclass(frozen=True)
class TokenLengthStats:
    """Char/4 heuristic token-length stats across every row in the pair."""

    min_tokens: int
    max_tokens: int
    mean_tokens: float
    rows_over_max_seq_len: int


@dataclass(frozen=True)
class MaskingSummary:
    """Usage of per-assistant-message ``weight`` across the pair."""

    rows_with_weight: int
    trained_assistant_messages: int
    masked_assistant_messages: int


@dataclass(frozen=True)
class SftValidationReport:
    """Combined outcome of :func:`validate_sft_dataset`.

    Bool-castable so gate code reads identically to ``ValidationReport``::

        if not validate_sft_dataset(train, eval_dataset):
            raise SystemExit("Fix the dataset before launching.")
    """

    issues: list[SftIssue] = field(default_factory=list)
    train_row_count: int = 0
    eval_row_count: int = 0
    token_length_stats: TokenLengthStats = field(
        default_factory=lambda: TokenLengthStats(0, 0, 0.0, 0)
    )
    masking_summary: MaskingSummary = field(
        default_factory=lambda: MaskingSummary(0, 0, 0)
    )

    @property
    def ok(self) -> bool:
        # An empty/absent train set is failure, matching ValidationReport's
        # "nothing validated is not a pass" — even with zero error issues.
        has_errors = any(issue.severity == "error" for issue in self.issues)
        return not has_errors and self.train_row_count >= 1

    def __bool__(self) -> bool:
        return self.ok


def validate_sft_dataset(
    train: SftDataset,
    eval: SftDataset | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES,
) -> SftValidationReport:
    """Validate a train (+ optional eval) :class:`SftDataset` pair.

    Every row is checked against :func:`benchmax.sft.schema.validate_row`
    (error-severity issues); rows serializing at/above ``max_row_bytes`` or
    estimated (char/4) over ``max_seq_len`` tokens, and rows carrying a
    per-message ``weight``, get informational notice-severity issues. An
    empty train dataset is an error; an empty eval dataset is a notice.
    """
    issues: list[SftIssue] = list(train.load_issues)
    if eval is not None:
        issues.extend(eval.load_issues)

    token_counts: list[int] = []
    rows_over_max_seq_len = 0
    rows_with_weight = 0
    trained_assistant_messages = 0
    masked_assistant_messages = 0

    for dataset in (train, eval) if eval is not None else (train,):
        for row in dataset.rows:
            for message in validate_row(row.data):
                issues.append(
                    SftIssue(row.source_path, row.physical_line, "error", message)
                )

            serialized_len = _safe_serialized_len(row.data)
            if serialized_len is not None and serialized_len >= max_row_bytes:
                issues.append(
                    SftIssue(
                        row.source_path,
                        row.physical_line,
                        "notice",
                        f"row is {serialized_len} bytes, at/above max_row_bytes ({max_row_bytes})",
                    )
                )

            token_count = _estimate_tokens(row.data)
            token_counts.append(token_count)
            if token_count > max_seq_len:
                rows_over_max_seq_len += 1
                issues.append(
                    SftIssue(
                        row.source_path,
                        row.physical_line,
                        "notice",
                        f"estimated token length ~{token_count} exceeds max_seq_len "
                        f"({max_seq_len}, char/4 heuristic)",
                    )
                )

            row_has_weight, trained, masked = _weight_counts(row.data)
            trained_assistant_messages += trained
            masked_assistant_messages += masked
            if row_has_weight:
                rows_with_weight += 1
                issues.append(
                    SftIssue(
                        row.source_path,
                        row.physical_line,
                        "notice",
                        "row has per-message 'weight' set (experimental — masking support "
                        "unconfirmed)",
                    )
                )

    if len(train.rows) == 0:
        issues.append(SftIssue(train.path, 0, "error", "train dataset is empty"))
    if eval is None:
        issues.append(SftIssue("", 0, "notice", "no eval dataset provided"))
    elif len(eval.rows) == 0:
        issues.append(SftIssue(eval.path, 0, "notice", "eval dataset is empty"))

    return SftValidationReport(
        issues=issues,
        train_row_count=len(train.rows),
        eval_row_count=len(eval.rows) if eval is not None else 0,
        token_length_stats=_token_stats(token_counts, rows_over_max_seq_len),
        masking_summary=MaskingSummary(
            rows_with_weight=rows_with_weight,
            trained_assistant_messages=trained_assistant_messages,
            masked_assistant_messages=masked_assistant_messages,
        ),
    )


def _safe_serialized_len(row_data: dict) -> int | None:
    """Byte length of ``row_data`` via the same encoding ``canonical_jsonl`` uses,
    or ``None`` if it can't serialize.

    A non-serializable row already gets a "not JSON-serializable" error
    from :func:`benchmax.sft.schema.validate_row` above — this guard just
    keeps a bad row from crashing the rest of the report.
    """
    try:
        return len(canonical_row_bytes(row_data))
    except (TypeError, ValueError):
        return None


def _estimate_tokens(row_data: dict) -> int:
    messages = row_data.get("messages")
    if not isinstance(messages, list):
        return 0
    total_chars = sum(len(message_text(m)) for m in messages if isinstance(m, dict))
    return total_chars // 4


def _weight_counts(row_data: dict) -> tuple[bool, int, int]:
    messages = row_data.get("messages")
    if not isinstance(messages, list):
        return False, 0, 0

    row_has_weight = False
    trained = 0
    masked = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if "weight" in message:
            row_has_weight = True
            if message.get("weight") == 0:
                masked += 1
            else:
                trained += 1
        else:
            trained += 1
    return row_has_weight, trained, masked


def _token_stats(
    token_counts: list[int], rows_over_max_seq_len: int
) -> TokenLengthStats:
    if not token_counts:
        return TokenLengthStats(
            min_tokens=0, max_tokens=0, mean_tokens=0.0, rows_over_max_seq_len=0
        )
    return TokenLengthStats(
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        mean_tokens=sum(token_counts) / len(token_counts),
        rows_over_max_seq_len=rows_over_max_seq_len,
    )
