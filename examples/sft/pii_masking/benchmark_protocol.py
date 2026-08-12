"""Immutable protocol identity and the append-only execution record.

The benchmark separates two kinds of fact, and this module owns the boundary:

*Protocol* facts are decisions made before any model output exists — source
revisions, selected sample identities, prompts, request shape, training config,
alignment and scoring versions. They live in ``protocol.json``, they are
canonicalized to exact bytes, and ``protocol_id`` is the SHA-256 of those bytes.
Nothing rewrites them. A changed decision is a *new* protocol, not an edit,
which is what makes "we did not tune this after seeing results" checkable.

*Execution* facts are everything learned by running — endpoints, code SHAs,
attestations, run ids, checkpoint refs, cost estimates. They append to a
hash-chained ``execution-events.jsonl``. ``execution.json`` is a materialized
view recomputed from that chain and is never the authority.

``protocol.json`` deliberately does not contain ``protocol_id``: a document
cannot contain its own hash. The id lives in the directory name and in the
execution envelope.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Semantic contract version. Bump for ANY change to selection, prompts,
# alignment, scoring, request construction, or journal semantics. Code SHAs are
# execution evidence and are not a substitute for this.
BENCHMARK_REVISION = "pii-masking-benchmark-v1"

OUTPUT_ROOT = Path("outputs") / "pii_masking_benchmark"

PROTOCOL_FILENAME = "protocol.json"
EVENTS_FILENAME = "execution-events.jsonl"
EXECUTION_FILENAME = "execution.json"

EXECUTION_EVENT_DOMAIN = b"castform-execution-event-v1\0"
PREDICTION_EVENT_DOMAIN = b"castform-prediction-event-v1\0"

# First link in a chain has no predecessor. 64 zeroes, not an empty string, so
# every record has the same shape and a truncated file cannot look like a start.
GENESIS_DIGEST = "0" * 64

BENCHMARK_SOURCES = ("piimb-ai4privacy", "openpii-validation")


class ProtocolError(RuntimeError):
    """A protocol document or execution chain violates its contract."""


# ── canonical bytes ───────────────────────────────────────────────────────────
def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ProtocolError(f"duplicate object key in JSON input: {key!r}")
        seen[key] = value
    return seen


def loads(raw: str | bytes) -> Any:
    """Parse JSON, rejecting duplicate object keys.

    ``json.loads`` keeps the last duplicate silently, which would let two
    different documents canonicalize to the same bytes.
    """
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)


def _check_jsonable(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("non-finite floats are not representable in canonical JSON")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"object keys must be strings, got {type(key).__name__}")
            _check_jsonable(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_jsonable(item)
        return
    raise ProtocolError(f"value of type {type(value).__name__} is not JSON")


def canonical_bytes(value: Any) -> bytes:
    """Return the one byte encoding this value is allowed to have.

    Recursively sorted keys, no ASCII escaping, no NaN/Infinity, no incidental
    whitespace, exactly one trailing newline. Two structurally equal documents
    always produce identical bytes, which is the whole basis for hashing them.
    """
    _check_jsonable(value)
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return encoded.encode("utf-8") + b"\n"


def canonical_digest(value: Any) -> str:
    """SHA-256 over this value's canonical bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes so a crash cannot leave a partially written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# ── protocol identity ─────────────────────────────────────────────────────────
def protocol_id_for(payload: Mapping[str, Any]) -> str:
    """Return the protocol id for a payload, refusing a self-referential one."""
    if "protocol_id" in payload:
        raise ProtocolError(
            "protocol.json must not contain protocol_id; a document cannot carry its own hash"
        )
    return canonical_digest(payload)


@dataclass(frozen=True)
class Protocol:
    """One frozen protocol document plus the id its bytes produce."""

    payload: Mapping[str, Any]
    protocol_id: str

    @property
    def benchmark_source(self) -> str:
        return str(self.payload["benchmark_source"])

    @property
    def benchmark_revision(self) -> str:
        return str(self.payload["benchmark_revision"])

    def directory(self, root: Path = OUTPUT_ROOT) -> Path:
        """The output directory this protocol owns."""
        return Path(root) / self.protocol_id

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Protocol:
        """Freeze a payload, computing its id from the exact persisted bytes."""
        return cls(payload=payload, protocol_id=protocol_id_for(payload))

    @classmethod
    def load(cls, directory: Path) -> Protocol:
        """Read a protocol and verify its bytes still hash to its directory name.

        The directory name is the claim; the bytes are the evidence. Checking
        them on every load is what makes later "the protocol was not edited
        mid-run" statements true rather than assumed.
        """
        path = Path(directory) / PROTOCOL_FILENAME
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(f"no protocol at {path}") from exc

        payload = loads(raw)
        if canonical_bytes(payload) != raw:
            raise ProtocolError(f"{path} is not in canonical form; it has been rewritten")

        protocol_id = protocol_id_for(payload)
        expected = Path(directory).name
        if protocol_id != expected:
            raise ProtocolError(
                f"{path} hashes to {protocol_id} but lives in a directory named {expected}"
            )
        return cls(payload=payload, protocol_id=protocol_id)

    def write(self, directory: Path) -> Path:
        """Persist the protocol; refuses to overwrite an existing one."""
        path = Path(directory) / PROTOCOL_FILENAME
        if path.exists():
            raise ProtocolError(f"{path} already exists; a protocol is never rewritten")
        atomic_write(path, canonical_bytes(self.payload))
        return path


def build_protocol_payload(
    *,
    benchmark_source: str,
    execution_environment: str,
    base_model: Mapping[str, Any],
    training: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt: Mapping[str, Any],
    selection: Mapping[str, Any],
    scoring: Mapping[str, Any],
    expected_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the immutable protocol document.

    Every argument is a decision that must not change once model outputs exist.
    Anything learned by running belongs in the execution chain instead.
    """
    if benchmark_source not in BENCHMARK_SOURCES:
        raise ProtocolError(f"unknown benchmark_source {benchmark_source!r}")
    return {
        "benchmark_revision": BENCHMARK_REVISION,
        "benchmark_source": benchmark_source,
        "execution_environment": execution_environment,
        "base_model": dict(base_model),
        "training": dict(training),
        "generation": dict(generation),
        "prompt": dict(prompt),
        "selection": dict(selection),
        "scoring": dict(scoring),
        "expected_counts": dict(expected_counts),
    }


# ── request identity ──────────────────────────────────────────────────────────
def request_payload_hash(request: Mapping[str, Any]) -> str:
    """Hash an OpenAI request with ``model`` removed.

    Excluding ``model`` is the point: it lets the runner prove the base and SFT
    requests are byte-identical apart from which model answers them, so a
    measured difference cannot be an artifact of differing prompts or decoding.
    """
    stripped = {key: value for key, value in request.items() if key != "model"}
    if not stripped:
        raise ProtocolError("request must contain fields other than 'model'")
    return canonical_digest(stripped)


@dataclass(frozen=True)
class RequestIdentity:
    """What makes one prediction unique. Phase is metadata, never identity.

    Because phase is excluded, the 20-row smoke set is genuinely *inside* the
    pilot, and the pilot inside the full suite: a later phase re-encountering an
    identity reuses the earlier canonical response instead of paying twice.
    """

    protocol_id: str
    benchmark_revision: str
    task_name: str
    sample_uid: str
    request_payload_hash: str
    model_ref: str
    model_artifact_digest: str

    def as_tuple(self) -> tuple[str, ...]:
        return (
            self.protocol_id,
            self.benchmark_revision,
            self.task_name,
            self.sample_uid,
            self.request_payload_hash,
            self.model_ref,
            self.model_artifact_digest,
        )

    @property
    def digest(self) -> str:
        """Stable identifier for journal lookup and resume."""
        return canonical_digest(list(self.as_tuple()))


def scoped_sample_uid(task_name: str, uid: object) -> str:
    """Return the task-scoped sample uid.

    Sources may reuse a uid across tasks, so identity is scoped by task; the
    same document evaluated under two task names is two samples, not one.
    """
    return f"{task_name}:{uid}"


# ── hash-chained execution events ─────────────────────────────────────────────
def event_digest(core: Mapping[str, Any], *, domain: bytes = EXECUTION_EVENT_DOMAIN) -> str:
    """Digest an event core. The core must not already carry ``event_digest``."""
    if "event_digest" in core:
        raise ProtocolError("event core must not contain event_digest")
    if "previous_digest" not in core:
        raise ProtocolError("event core must contain previous_digest")
    return hashlib.sha256(domain + canonical_bytes(core)).hexdigest()


def build_event(
    previous_digest: str,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    timestamp: str,
    domain: bytes = EXECUTION_EVENT_DOMAIN,
) -> dict[str, Any]:
    """Build one chained record: core fields plus the digest over them.

    ``timestamp`` is supplied by the caller rather than read from the clock here
    so the encoding stays a pure function of its inputs and stays testable.
    """
    core = {
        "previous_digest": previous_digest,
        "event_type": event_type,
        "timestamp": timestamp,
        "payload": dict(payload),
    }
    record = dict(core)
    record["event_digest"] = event_digest(core, domain=domain)
    return record


def append_event(path: Path, record: Mapping[str, Any]) -> None:
    """Append one canonical record, flushed to disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path: Path, *, domain: bytes = EXECUTION_EVENT_DOMAIN) -> list[dict[str, Any]]:
    """Read and fully verify a chain.

    Every record's digest is recomputed and every link checked against its
    predecessor. A chain that does not verify raises rather than returning
    partial state — a materialized view built from a broken chain would look
    exactly like a good one.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    previous = GENESIS_DIGEST
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ProtocolError(f"{path}:{number} is blank; the chain has no optional records")
        record = loads(line)
        if canonical_bytes(record) != line + b"\n":
            raise ProtocolError(f"{path}:{number} is not in canonical form")
        if not isinstance(record, dict) or "event_digest" not in record:
            raise ProtocolError(f"{path}:{number} has no event_digest")

        core = {key: value for key, value in record.items() if key != "event_digest"}
        recomputed = event_digest(core, domain=domain)
        if recomputed != record["event_digest"]:
            raise ProtocolError(
                f"{path}:{number} digest {record['event_digest']} does not match its content"
            )
        if core["previous_digest"] != previous:
            raise ProtocolError(
                f"{path}:{number} follows {core['previous_digest']} but the chain is at {previous}"
            )
        previous = record["event_digest"]
        records.append(record)
    return records


def chain_head(path: Path, *, domain: bytes = EXECUTION_EVENT_DOMAIN) -> str:
    """Return the digest the next appended record must reference."""
    records = read_events(path, domain=domain)
    return records[-1]["event_digest"] if records else GENESIS_DIGEST


class ExecutionLog:
    """Append-only execution record for one protocol directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.events_path = self.directory / EVENTS_FILENAME

    def append(
        self, event_type: str, payload: Mapping[str, Any], *, timestamp: str
    ) -> dict[str, Any]:
        """Append one event, chained to the current head."""
        record = build_event(chain_head(self.events_path), event_type, payload, timestamp=timestamp)
        append_event(self.events_path, record)
        return record

    def events(self) -> list[dict[str, Any]]:
        """Every verified event, oldest first."""
        return read_events(self.events_path)

    def materialize(self, protocol_id: str) -> dict[str, Any]:
        """Recompute ``execution.json`` from the verified chain.

        Later events for the same key win. This is a view: deleting it loses
        nothing, and editing it changes nothing, because the chain is the
        authority.
        """
        events = self.events()
        state: dict[str, Any] = {}
        for record in events:
            payload = record["payload"]
            if isinstance(payload, Mapping):
                for key, value in payload.items():
                    state[key] = value

        return {
            "protocol_id": protocol_id,
            "benchmark_revision": BENCHMARK_REVISION,
            "event_count": len(events),
            "chain_head": events[-1]["event_digest"] if events else GENESIS_DIGEST,
            **state,
        }

    def write_materialized(self, protocol_id: str) -> Path:
        """Persist the materialized view, overwriting any previous one."""
        path = self.directory / EXECUTION_FILENAME
        atomic_write(path, canonical_bytes(self.materialize(protocol_id)))
        return path


def iter_protocol_dirs(root: Path = OUTPUT_ROOT) -> Iterator[Path]:
    """Yield existing protocol directories under ``root``."""
    base = Path(root)
    if not base.is_dir():
        return
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / PROTOCOL_FILENAME).is_file():
            yield child


def require_all(values: Sequence[Any], message: str) -> None:
    """Raise ``ProtocolError`` unless every value is truthy."""
    if not all(values):
        raise ProtocolError(message)
