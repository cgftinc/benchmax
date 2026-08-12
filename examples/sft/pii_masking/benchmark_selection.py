"""Deterministic sample selection, frozen before any model output exists.

Every identity this module produces is a function of the source bytes and a
versioned hash procedure — never of a score, a length, an entity count, or
anything else observed after the fact. That is the property that lets the final
report claim the samples were not chosen to flatter the result.

Order matters and is not an implementation detail. The evaluation universe and
its nested pilot/smoke/audit subsets are frozen *first*; train and development
are then selected from the disjoint remainder, excluding evaluation lineage and
exact text. Doing it the other way round would let training data leak into the
measurement and there would be no artifact showing it had.

Determinism has one more requirement that is easy to miss: the result must not
depend on the order the source happens to stream in. Duplicate resolution
therefore picks a *global* canonical representative per exact text before any
quota is applied, so a forward and a reversed stream produce byte-identical
outputs. The tests assert exactly that.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark_protocol import (
    ProtocolError,
    canonical_bytes,
)

# Versioned selection procedure. Any change to how identities are derived is a
# new value here AND a new benchmark_revision.
SELECTION_VERSION = "castform-pii-benchmark-v1"

ENGLISH = "en"

# Frozen order. Quota remainders are handed out in exactly this sequence, so the
# order is part of the contract, not a presentation choice.
NON_ENGLISH_LANGUAGES: tuple[str, ...] = (
    "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hr", "hu",
    "it", "lt", "lv", "nl", "pl", "pt", "ro", "sk", "sl", "sr", "sv",
)  # fmt: skip

OPENPII_DATASET = "ai4privacy/pii-masking-openpii-1m"
OPENPII_REVISION = "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0"

PIIMB_DATASET = "piimb/pii-masking-benchmark"
PIIMB_REVISION = "7d797b9fc8dc1942cef60fbe532e7d1a0e31b655"
PIIMB_CONFIG = "sentences"

TASK_OPENPII_EN = "castform-openpii-en-v1"
TASK_OPENPII_NONEN = "castform-openpii-nonen-v1"
TASK_PIIMB_EN = "ai4privacy-en"
TASK_PIIMB_MULTI = "ai4privacy-multi"

OPENPII_TASKS = (TASK_OPENPII_EN, TASK_OPENPII_NONEN)
PIIMB_TASKS = (TASK_PIIMB_EN, TASK_PIIMB_MULTI)

# Frozen counts.
TRAIN_ROWS = 4096
DEVELOPMENT_ROWS = 256
OPENPII_TASK_ROWS = 5000
PILOT_ROWS_PER_TASK = 1000
SMOKE_ROWS_PER_TASK = 10
AUDIT_ROWS_PER_TASK = 25

# The placeholder grammar the model is trained to emit. Source text containing
# it is unusable for evaluation: a literal copied from the input would be
# indistinguishable from a placeholder the model produced.
PLACEHOLDER_PATTERN = re.compile(r"\[(?:[A-Z][A-Z0-9]*)(?:_[0-9]+)?\]")


class SelectionError(ProtocolError):
    """Selection cannot produce a frozen, reproducible sample."""


# ── hashes ────────────────────────────────────────────────────────────────────
def selection_hash(domain: str, uid: str) -> str:
    """Hash that orders candidates within one domain.

    Domain-scoped so the same source row ranks differently for evaluation, for
    train, and for development — otherwise the three would prefer the same rows
    and the disjointness rules would fight the ordering.
    """
    return hashlib.sha256(canonical_bytes([SELECTION_VERSION, domain, uid])).hexdigest()


def text_hash(text: str) -> str:
    """SHA-256 of the exact source text. No normalization of any kind."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical bytes of a complete source row."""
    return hashlib.sha256(canonical_bytes(dict(row))).hexdigest()


def lineage_key(upstream_uid: object) -> tuple[str, str]:
    """Cross-source lineage identity.

    Both sources are views of the same OpenPII family, so lineage lives in one
    namespace: a row reused across dataset or split boundaries collides here
    rather than hiding behind a different dataset name.
    """
    return (OPENPII_DATASET, str(upstream_uid))


def language_quotas(total: int, languages: Sequence[str] = NON_ENGLISH_LANGUAGES) -> dict[str, int]:
    """Split ``total`` across ``languages`` as evenly as the count allows.

    Every language gets ``total // len(languages)``; the first ``total %
    len(languages)`` codes in frozen order get one more. Deterministic, and the
    remainder never lands wherever a dict happened to iterate.
    """
    if total < 0:
        raise SelectionError("quota total must not be negative")
    if not languages:
        raise SelectionError("quota needs at least one language")
    base, remainder = divmod(total, len(languages))
    return {code: base + (1 if index < remainder else 0) for index, code in enumerate(languages)}


# ── rows ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SourceRow:
    """One normalized source record plus the identities derived from it."""

    uid: str
    language: str
    source_text: str
    masked_text: str
    upstream_uid: str
    payload: Mapping[str, Any]

    @property
    def text_hash(self) -> str:
        return text_hash(self.source_text)

    @property
    def row_digest(self) -> str:
        return row_digest(self.payload)

    def selection_hash(self, domain: str) -> str:
        return selection_hash(domain, self.uid)

    def order_key(self, domain: str) -> tuple[str, str, str, str]:
        """The frozen total order: selection hash, language, uid, row digest."""
        return (self.selection_hash(domain), self.language, self.uid, self.row_digest)


@dataclass(frozen=True)
class SelectedRow:
    """One frozen selection: the row plus the domain it was selected for."""

    row: SourceRow
    domain: str

    @property
    def selection_hash(self) -> str:
        return self.row.selection_hash(self.domain)

    def manifest_entry(self) -> dict[str, Any]:
        """Hashes, ids, and language only — never source text."""
        return {
            "uid": self.row.uid,
            "language": self.row.language,
            "selection_hash": self.selection_hash,
            "text_hash": self.row.text_hash,
            "row_digest": self.row.row_digest,
        }


def normalize_openpii_row(payload: Mapping[str, Any]) -> SourceRow:
    """Map one OpenPII record into a ``SourceRow``."""
    for field in ("uid", "language", "source_text", "masked_text"):
        if field not in payload:
            raise SelectionError(f"OpenPII row is missing required field {field!r}")
    return SourceRow(
        uid=str(payload["uid"]),
        language=str(payload["language"]),
        source_text=str(payload["source_text"]),
        masked_text=str(payload["masked_text"]),
        upstream_uid=str(payload["uid"]),
        payload=dict(payload),
    )


def normalize_piimb_row(payload: Mapping[str, Any], task_name: str) -> SourceRow:
    """Map one PIIMB record into a ``SourceRow`` with a task-scoped uid.

    PIIMB reuses uids across its tasks, so identity is scoped by task while
    lineage stays in the shared OpenPII namespace via ``source_uid``.
    """
    upstream = payload.get("source_uid")
    if upstream is None or str(upstream).strip() == "":
        raise SelectionError(
            f"PIIMB row in {task_name!r} has missing or empty source_uid; "
            "cross-source lineage cannot be established"
        )
    for field in ("language", "source_text", "masked_text"):
        if field not in payload:
            raise SelectionError(f"PIIMB row is missing required field {field!r}")
    return SourceRow(
        uid=f"{task_name}:{payload.get('uid', upstream)}",
        language=str(payload["language"]),
        source_text=str(payload["source_text"]),
        masked_text=str(payload["masked_text"]),
        upstream_uid=str(upstream),
        payload=dict(payload),
    )


# ── duplicate ledger ──────────────────────────────────────────────────────────
class DuplicateLedger:
    """Disk-backed ledger of uid and exact-text identity.

    SQLite rather than a dict because the pinned OpenPII train split is ~1M rows
    and preparation must not depend on holding all of them in memory.

    Two rules it enforces, both fatal:

    * one uid must map to exactly one row digest — the same identifier
      describing two different records means the source is not what we think;
    * for each exact text, one *global* canonical representative is chosen by
      the frozen total order, independent of arrival order.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._connection = sqlite3.connect(str(path) if path else ":memory:")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rows (
                uid TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                row_digest TEXT NOT NULL,
                upstream_uid TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS rows_text ON rows(text_hash)")

    def add(self, row: SourceRow, domain: str) -> None:
        """Record one row, rejecting a uid that already means something else.

        ``domain`` is accepted for symmetry with the rest of the API but is not
        stored: the selection hash is a pure function of (domain, uid), and one
        ledger serves several domains, so persisting a single domain's hash
        would be actively misleading.
        """
        digest = row.row_digest
        existing = self._connection.execute(
            "SELECT row_digest FROM rows WHERE uid = ?", (row.uid,)
        ).fetchone()
        if existing is not None:
            if existing[0] != digest:
                raise SelectionError(
                    f"uid {row.uid!r} maps to two different rows ({existing[0]} and {digest}); "
                    "the source is not stable enough to select from"
                )
            return

        self._connection.execute(
            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.uid,
                row.language,
                row.text_hash,
                digest,
                row.upstream_uid,
                canonical_bytes(dict(row.payload)).decode("utf-8"),
            ),
        )

    def extend(self, rows: Iterable[SourceRow], domain: str) -> None:
        for row in rows:
            self.add(row, domain)
        self._connection.commit()

    def canonical_uids(self, domain: str) -> list[str]:
        """Return one uid per exact text, chosen by the frozen total order.

        Streams the identity columns from disk rather than materializing rows:
        only one small ordering key per *distinct text* is ever held in memory,
        never the row payloads.
        """
        by_text: dict[str, tuple[tuple[str, str, str, str], str]] = {}
        cursor = self._connection.execute("SELECT uid, language, text_hash, row_digest FROM rows")
        for uid, language, text_hash, row_digest in cursor:
            key = (selection_hash(domain, uid), language, uid, row_digest)
            current = by_text.get(text_hash)
            if current is None or key < current[0]:
                by_text[text_hash] = (key, uid)
        return [uid for _, uid in sorted(by_text.values())]

    def row(self, uid: str) -> SourceRow:
        """Rehydrate one row from disk."""
        found = self._connection.execute(
            "SELECT language, upstream_uid, payload FROM rows WHERE uid = ?", (uid,)
        ).fetchone()
        if found is None:
            raise SelectionError(f"uid {uid!r} is not in the ledger")
        language, upstream_uid, payload = found
        decoded = json.loads(payload)
        return SourceRow(
            uid=uid,
            language=language,
            source_text=str(decoded["source_text"]),
            masked_text=str(decoded["masked_text"]),
            upstream_uid=upstream_uid,
            payload=decoded,
        )

    def __len__(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])

    def close(self) -> None:
        self._connection.close()


# ── quota allocation ──────────────────────────────────────────────────────────
def take_english(rows: Sequence[SourceRow], count: int, domain: str) -> list[SourceRow]:
    """Take the lowest-ordered ``count`` English rows."""
    candidates = sorted(
        (row for row in rows if row.language == ENGLISH), key=lambda row: row.order_key(domain)
    )
    if len(candidates) < count:
        raise SelectionError(
            f"need {count} {ENGLISH!r} rows for domain {domain!r} "
            f"but only {len(candidates)} are available"
        )
    return candidates[:count]


def take_language_balanced(
    rows: Sequence[SourceRow],
    count: int,
    domain: str,
    languages: Sequence[str] = NON_ENGLISH_LANGUAGES,
) -> list[SourceRow]:
    """Take ``count`` rows spread across ``languages`` by the frozen quota.

    An insufficient quota for any single language is fatal rather than
    reallocated: silently borrowing from another language would change the
    documented mix without changing the protocol.
    """
    quotas = language_quotas(count, languages)
    by_language: dict[str, list[SourceRow]] = {code: [] for code in languages}
    for row in rows:
        if row.language in by_language:
            by_language[row.language].append(row)

    selected: list[SourceRow] = []
    for code in languages:
        wanted = quotas[code]
        available = sorted(by_language[code], key=lambda row: row.order_key(domain))
        if len(available) < wanted:
            raise SelectionError(
                f"language {code!r} needs {wanted} rows for domain {domain!r} "
                f"but only {len(available)} are available"
            )
        selected.extend(available[:wanted])
    return selected


def sort_selection(rows: Iterable[SourceRow], domain: str) -> list[SourceRow]:
    """Return rows in the frozen artifact order."""
    return sorted(rows, key=lambda row: row.order_key(domain))


def nested_subset(rows: Sequence[SourceRow], count: int, domain: str) -> list[SourceRow]:
    """Take the lowest-ordered ``count`` rows, preserving nesting.

    Because the order is total and deterministic, the smoke set is a prefix of
    the pilot and the pilot a prefix of the task — which is what lets a later
    phase reuse an earlier phase's responses instead of paying twice.
    """
    if len(rows) < count:
        raise SelectionError(f"cannot take {count} nested rows from {len(rows)}")
    return sort_selection(rows, domain)[:count]


# ── validation ────────────────────────────────────────────────────────────────
def reject_placeholder_collisions(rows: Iterable[SourceRow], task_name: str) -> None:
    """Fail if evaluation source text already contains the placeholder grammar.

    Such a row cannot be scored: a placeholder copied verbatim from the input is
    indistinguishable from one the model emitted, so alignment would credit or
    penalize the model for the source's own formatting.
    """
    offenders = [row.uid for row in rows if PLACEHOLDER_PATTERN.search(row.source_text)]
    if offenders:
        shown = ", ".join(offenders[:5])
        raise SelectionError(
            f"{len(offenders)} row(s) in task {task_name!r} contain the placeholder grammar "
            f"in their source text (e.g. {shown}); they cannot be scored unambiguously"
        )


def assert_disjoint(
    evaluation: Iterable[SourceRow],
    other: Iterable[SourceRow],
    *,
    label: str,
) -> None:
    """Fail on any lineage or exact-text overlap with the evaluation set."""
    eval_rows = list(evaluation)
    eval_lineage = {lineage_key(row.upstream_uid) for row in eval_rows}
    eval_texts = {row.text_hash for row in eval_rows}

    for row in other:
        if lineage_key(row.upstream_uid) in eval_lineage:
            raise SelectionError(
                f"{label} row {row.uid!r} shares source lineage with the evaluation set"
            )
        if row.text_hash in eval_texts:
            raise SelectionError(f"{label} row {row.uid!r} has text identical to an evaluation row")


def observed_labels(rows: Iterable[SourceRow]) -> set[str]:
    """Every placeholder label appearing in the masked text of ``rows``."""
    labels: set[str] = set()
    for row in rows:
        for match in PLACEHOLDER_PATTERN.finditer(row.masked_text):
            labels.add(match.group(0).strip("[]").rsplit("_", 1)[0])
    return labels


def assert_label_coverage(train: Iterable[SourceRow], population: Iterable[SourceRow]) -> None:
    """Require the train split to exhibit every label the source stream shows."""
    missing = observed_labels(population) - observed_labels(train)
    if missing:
        raise SelectionError(
            f"train split omits {len(missing)} label(s) present in the source stream: "
            f"{', '.join(sorted(missing))}"
        )


# ── source adapters ───────────────────────────────────────────────────────────
def stream_openpii(split: str) -> Iterator[Mapping[str, Any]]:
    """Stream the pinned OpenPII split. Imported lazily; never at module import."""
    from datasets import load_dataset

    return iter(
        load_dataset(OPENPII_DATASET, revision=OPENPII_REVISION, split=split, streaming=True)
    )


def stream_piimb(task_name: str) -> Iterator[Mapping[str, Any]]:
    """Stream one pinned PIIMB task. Imported lazily; never at module import."""
    from datasets import load_dataset

    dataset = load_dataset(
        PIIMB_DATASET,
        PIIMB_CONFIG,
        revision=PIIMB_REVISION,
        split="test",
        streaming=True,
    )
    return (row for row in dataset if str(row.get("task")) == task_name)


# ── frozen universes ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvaluationUniverse:
    """The frozen evaluation set and its nested subsets, per task."""

    tasks: Mapping[str, list[SourceRow]]
    pilot: Mapping[str, list[SourceRow]]
    smoke: Mapping[str, list[SourceRow]]
    audit: Mapping[str, list[SourceRow]]

    def all_rows(self) -> list[SourceRow]:
        return [row for task in self.tasks for row in self.tasks[task]]

    def manifest(self) -> dict[str, Any]:
        """Hashes, ids, and counts only."""
        return {
            "tasks": {
                task: {
                    "count": len(rows),
                    "rows": [SelectedRow(row, task).manifest_entry() for row in rows],
                }
                for task, rows in self.tasks.items()
            },
            "pilot_counts": {task: len(rows) for task, rows in self.pilot.items()},
            "smoke_counts": {task: len(rows) for task, rows in self.smoke.items()},
            "audit_counts": {task: len(rows) for task, rows in self.audit.items()},
        }


def _nested_pilot(task: str, rows: Sequence[SourceRow], count: int) -> list[SourceRow]:
    """Pilot rows for one task, preserving that task's language mix."""
    if task in (TASK_OPENPII_NONEN, TASK_PIIMB_MULTI):
        return sort_selection(take_language_balanced(rows, count, task), task)
    return nested_subset(rows, count, task)


def build_evaluation_universe(
    task_rows: Mapping[str, Sequence[SourceRow]],
    *,
    pilot_rows: int = PILOT_ROWS_PER_TASK,
    smoke_rows: int = SMOKE_ROWS_PER_TASK,
    audit_rows: int = AUDIT_ROWS_PER_TASK,
) -> EvaluationUniverse:
    """Freeze the evaluation tasks and their nested pilot/smoke/audit subsets."""
    tasks: dict[str, list[SourceRow]] = {}
    pilot: dict[str, list[SourceRow]] = {}
    smoke: dict[str, list[SourceRow]] = {}
    audit: dict[str, list[SourceRow]] = {}

    for task, rows in task_rows.items():
        reject_placeholder_collisions(rows, task)
        ordered = sort_selection(rows, task)
        tasks[task] = ordered
        pilot[task] = _nested_pilot(task, ordered, pilot_rows)
        smoke[task] = nested_subset(pilot[task], smoke_rows, task)
        audit[task] = nested_subset(pilot[task], audit_rows, task)

    return EvaluationUniverse(tasks=tasks, pilot=pilot, smoke=smoke, audit=audit)


def select_openpii_evaluation(
    rows: Iterable[SourceRow],
    *,
    ledger_path: Path | None = None,
    task_rows: int = OPENPII_TASK_ROWS,
    pilot_rows: int = PILOT_ROWS_PER_TASK,
    smoke_rows: int = SMOKE_ROWS_PER_TASK,
    audit_rows: int = AUDIT_ROWS_PER_TASK,
) -> EvaluationUniverse:
    """Freeze the two OpenPII validation tasks from a validation stream."""
    ledger = DuplicateLedger(ledger_path)
    try:
        ledger.extend(rows, TASK_OPENPII_EN)
        english_pool = [ledger.row(uid) for uid in ledger.canonical_uids(TASK_OPENPII_EN)]
        multi_pool = [ledger.row(uid) for uid in ledger.canonical_uids(TASK_OPENPII_NONEN)]

        selected = {
            TASK_OPENPII_EN: take_english(english_pool, task_rows, TASK_OPENPII_EN),
            TASK_OPENPII_NONEN: take_language_balanced(multi_pool, task_rows, TASK_OPENPII_NONEN),
        }
    finally:
        ledger.close()

    return build_evaluation_universe(
        selected, pilot_rows=pilot_rows, smoke_rows=smoke_rows, audit_rows=audit_rows
    )


def select_piimb_evaluation(
    task_rows: Mapping[str, Iterable[SourceRow]],
    *,
    pilot_rows: int = PILOT_ROWS_PER_TASK,
    smoke_rows: int = SMOKE_ROWS_PER_TASK,
    audit_rows: int = AUDIT_ROWS_PER_TASK,
) -> EvaluationUniverse:
    """Freeze the two pinned PIIMB tasks, retaining every pinned row.

    Deliberately NOT deduplicated. The pinned tasks contain known cross-task
    duplicates; removing them would change what the published task membership
    means, so they stay and are reported as diagnostics instead.
    """
    return build_evaluation_universe(
        {task: list(rows) for task, rows in task_rows.items()},
        pilot_rows=pilot_rows,
        smoke_rows=smoke_rows,
        audit_rows=audit_rows,
    )


def duplicate_diagnostics(universe: EvaluationUniverse) -> dict[str, Any]:
    """Report within- and across-task duplicates without removing any."""
    seen_lineage: dict[tuple[str, str], list[str]] = {}
    seen_text: dict[str, list[str]] = {}
    for task, rows in universe.tasks.items():
        for row in rows:
            seen_lineage.setdefault(lineage_key(row.upstream_uid), []).append(task)
            seen_text.setdefault(row.text_hash, []).append(task)

    return {
        "cross_task_lineage_overlaps": sum(1 for tasks in seen_lineage.values() if len(tasks) > 1),
        "cross_task_text_overlaps": sum(1 for tasks in seen_text.values() if len(tasks) > 1),
    }


@dataclass(frozen=True)
class TrainingSplits:
    """The frozen train and development splits."""

    train: list[SourceRow]
    development: list[SourceRow]

    def manifest(self) -> dict[str, Any]:
        return {
            "train": {
                "count": len(self.train),
                "languages": _language_counts(self.train),
                "rows": [SelectedRow(row, "train").manifest_entry() for row in self.train],
            },
            "development": {
                "count": len(self.development),
                "languages": _language_counts(self.development),
                "rows": [
                    SelectedRow(row, "development").manifest_entry() for row in self.development
                ],
            },
        }


def _language_counts(rows: Iterable[SourceRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.language] = counts.get(row.language, 0) + 1
    return dict(sorted(counts.items()))


def select_train_and_development(
    rows: Iterable[SourceRow],
    evaluation: EvaluationUniverse,
    *,
    ledger_path: Path | None = None,
    train_rows: int = TRAIN_ROWS,
    development_rows: int = DEVELOPMENT_ROWS,
) -> TrainingSplits:
    """Select train then development from the evaluation-disjoint remainder.

    Train is allocated first and development from what remains, so the two are
    disjoint by construction rather than by a later check.
    """
    evaluation_rows = evaluation.all_rows()
    excluded_lineage = {lineage_key(row.upstream_uid) for row in evaluation_rows}
    excluded_texts = {row.text_hash for row in evaluation_rows}

    ledger = DuplicateLedger(ledger_path)
    try:
        ledger.extend(rows, "train")
        population = [ledger.row(uid) for uid in ledger.canonical_uids("train")]
    finally:
        ledger.close()

    candidates = [
        row
        for row in population
        if lineage_key(row.upstream_uid) not in excluded_lineage
        and row.text_hash not in excluded_texts
    ]

    half_train, half_dev = train_rows // 2, development_rows // 2
    train = sort_selection(
        take_english(candidates, half_train, "train")
        + take_language_balanced(candidates, half_train, "train"),
        "train",
    )

    taken = {row.uid for row in train}
    remaining = [row for row in candidates if row.uid not in taken]
    development = sort_selection(
        take_english(remaining, half_dev, "development")
        + take_language_balanced(remaining, half_dev, "development"),
        "development",
    )

    assert_label_coverage(train, population)
    assert_disjoint(evaluation_rows, train, label="train")
    assert_disjoint(evaluation_rows, development, label="development")
    return TrainingSplits(train=train, development=development)


def to_sft_bytes(rows: Sequence[SourceRow]) -> bytes:
    """Render selected rows as canonical ``benchmax-sft-v1`` bytes.

    Uses the tutorial's own mapper, so the benchmark trains on exactly the
    prompt the released example documents rather than a private variant.
    """
    from benchmax.sft import SftDataset

    from .main import map_source_row

    return SftDataset.from_rows([map_source_row(row.payload) for row in rows]).to_jsonl_bytes()


# ── preparation ───────────────────────────────────────────────────────────────
def _load_evaluation(benchmark_source: str) -> EvaluationUniverse:
    if benchmark_source == "openpii-validation":
        return select_openpii_evaluation(
            normalize_openpii_row(payload) for payload in stream_openpii("validation")
        )
    return select_piimb_evaluation(
        {
            task: [normalize_piimb_row(payload, task) for payload in stream_piimb(task)]
            for task in PIIMB_TASKS
        }
    )


def prepare(*, benchmark_source: str, output_root: Path) -> Path:
    """Read the pinned sources, freeze every sample, and write the protocol.

    Networked but free, and it produces no model output: every identity here is
    fixed before the first request is ever issued.
    """
    from .benchmark_protocol import Protocol, build_protocol_payload
    from .main import SOURCE_DATASET, SOURCE_REVISION, SYSTEM_PROMPT

    evaluation = _load_evaluation(benchmark_source)
    splits = select_train_and_development(
        (normalize_openpii_row(payload) for payload in stream_openpii("train")),
        evaluation,
    )

    payload = build_protocol_payload(
        benchmark_source=benchmark_source,
        execution_environment="production",
        base_model={"model_id": "Qwen/Qwen3.5-4B"},
        training={
            "train_rows": len(splits.train),
            "development_rows": len(splits.development),
            "lora_rank": 64,
        },
        generation={"temperature": 0, "top_p": 1, "n": 1, "max_tokens": 2048},
        prompt={"system": SYSTEM_PROMPT},
        selection={
            "selection_version": SELECTION_VERSION,
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "non_english_languages": list(NON_ENGLISH_LANGUAGES),
            "evaluation": evaluation.manifest(),
            "splits": splits.manifest(),
            "duplicate_diagnostics": duplicate_diagnostics(evaluation),
        },
        scoring={"placeholder_grammar": PLACEHOLDER_PATTERN.pattern},
        expected_counts={
            "tasks": {task: len(rows) for task, rows in evaluation.tasks.items()},
            "pilot": {task: len(rows) for task, rows in evaluation.pilot.items()},
            "smoke": {task: len(rows) for task, rows in evaluation.smoke.items()},
        },
    )

    protocol = Protocol.from_payload(payload)
    directory = protocol.directory(output_root)
    directory.mkdir(parents=True, exist_ok=True)
    protocol.write(directory)
    (directory / "train.jsonl").write_bytes(to_sft_bytes(splits.train))
    (directory / "eval.jsonl").write_bytes(to_sft_bytes(splits.development))
    return directory
