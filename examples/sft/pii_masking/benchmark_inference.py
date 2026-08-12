"""Paired base/SFT inference with a durable, resumable prediction journal.

Three properties this module exists to guarantee, none of which come for free:

**The comparison is fair.** Base and SFT requests are byte-identical apart from
the ``model`` field, and the runner proves it rather than assuming it. The two
are issued adjacent to each other, with the order alternating on the sample
hash's low bit, so time-varying endpoint behavior cannot systematically favor
one model.

**Every response is paid for once.** Identity excludes phase, so the smoke set
is genuinely inside the pilot and the pilot inside the full suite; a later phase
reuses an earlier canonical response instead of re-issuing it.

**Durability is honest about what it can promise.** An attempt-start record is
fsynced *before* the request goes out and the terminal record after it returns.
If the process dies in between, recovery records an ``unknown_remote_outcome``:
the API has no idempotency contract, so transport is genuinely at-least-once and
a duplicate remote completion is possible. What the protocol guarantees is
narrower and checkable — exactly one durable canonical response is ever scored,
and the count of unknown outcomes is disclosed rather than hidden.

The application is the sole retry owner. The underlying client is constructed
with ``max_retries=0`` so no hidden layer can retry without journaling it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_protocol import (
    PREDICTION_EVENT_DOMAIN,
    ProtocolError,
    RequestIdentity,
    canonical_bytes,
    canonical_digest,
    request_payload_hash,
    scoped_sample_uid,
)

JOURNAL_FILENAME = "predictions.jsonl"

BASE_MODEL_KEY = "base"
SFT_MODEL_KEY = "sft"

# Concurrency starts here and never exceeds the cap: this runs against a shared
# serving replica that other users depend on.
START_CONCURRENCY = 4
MAX_CONCURRENCY = 8

MAX_ATTEMPTS = 5
RETRYABLE_STATUSES = frozenset({408, 409, 429})
RETRY_AFTER_CAP_SECONDS = 60.0
BACKOFF_CAP_SECONDS = 30.0

# Retry-rate pause: fraction of retryable attempts in a rolling window.
RETRY_RATE_THRESHOLD = 0.02
RETRY_RATE_WINDOW = 200
RETRY_RATE_MINIMUM = 50

# Latency pause: three consecutive windows above twice the frozen baseline.
CALIBRATION_WINDOW = 100
CALIBRATION_WINDOWS = 3
LATENCY_WINDOW = 100
LATENCY_BREACH_WINDOWS = 3
LATENCY_MULTIPLIER = 2.0

ATTEMPT_START = "attempt_start"
ATTEMPT_END = "attempt_end"
UNKNOWN_REMOTE_OUTCOME = "unknown_remote_outcome"

OUTCOME_SUCCESS = "success"
OUTCOME_RETRYABLE = "retryable"
OUTCOME_FATAL = "fatal"


class InferenceError(ProtocolError):
    """The runner cannot proceed without violating the protocol."""


# ── model refs ────────────────────────────────────────────────────────────────
def validate_sft_model_ref(ref: str) -> int:
    """Return the iteration named by an exact numeric SFT ref.

    ``latest`` and any other moving alias are rejected: a published number must
    be attributable to one specific set of weights forever.
    """
    parts = (ref or "").split(":")
    if len(parts) != 4 or parts[0] != "ft" or not parts[3].isdigit():
        raise InferenceError(
            f"model ref {ref!r} is not an exact numeric checkpoint (ft:<base>:<run>:<iteration>)"
        )
    return int(parts[3])


def sft_artifact_digest(base_digest: str, adapter_manifest_digest: str) -> str:
    """Domain-separated digest binding an adapter to the base it rides on.

    An adapter is only meaningful against specific base weights, so the identity
    of an SFT response covers both.
    """
    return hashlib.sha256(
        b"castform-sft-artifact-v1\0" + canonical_bytes([base_digest, adapter_manifest_digest])
    ).hexdigest()


# ── requests ──────────────────────────────────────────────────────────────────
def build_request(protocol: Any, source_text: str, model: str) -> dict[str, Any]:
    """Build one OpenAI request from frozen protocol settings.

    Every generation setting comes from the protocol; only ``model`` varies
    between the two arms.
    """
    generation = dict(protocol.payload["generation"])
    system_prompt = protocol.payload["prompt"]["system"]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source_text},
        ],
        **generation,
    }


def assert_request_parity(base_request: Mapping[str, Any], sft_request: Mapping[str, Any]) -> str:
    """Require the two arms to differ only in ``model``; return the shared hash.

    Checked on the actual bodies about to be sent, not on the code that built
    them, so a divergence introduced anywhere upstream still fails here.
    """
    base_stripped = {k: v for k, v in base_request.items() if k != "model"}
    sft_stripped = {k: v for k, v in sft_request.items() if k != "model"}
    if canonical_bytes(base_stripped) != canonical_bytes(sft_stripped):
        raise InferenceError(
            "base and SFT requests differ by more than the model field; "
            "the comparison would not be attributable to the model"
        )
    return request_payload_hash(base_request)


@dataclass(frozen=True)
class Sample:
    """One evaluation row, with the identity fields the journal needs."""

    task_name: str
    uid: str
    source_text: str

    @property
    def sample_uid(self) -> str:
        return scoped_sample_uid(self.task_name, self.uid)

    def base_first(self) -> bool:
        """Alternate arm order on the sample hash's low bit.

        Deterministic, but balanced across the suite, so a drift in endpoint
        behavior over time cannot systematically favor whichever model always
        went first.
        """
        return int(canonical_digest(self.sample_uid)[-1], 16) % 2 == 0


# ── retry policy ──────────────────────────────────────────────────────────────
def classify_response(status: int | None, error: BaseException | None) -> str:
    """Classify one attempt as success, retryable, or fatal."""
    if error is not None:
        return OUTCOME_RETRYABLE  # transport failures are retryable
    if status is None:
        return OUTCOME_FATAL
    if 200 <= status < 300:
        return OUTCOME_SUCCESS
    if status in RETRYABLE_STATUSES or 500 <= status < 600:
        return OUTCOME_RETRYABLE
    return OUTCOME_FATAL


def retry_delay(attempt: int, retry_after: float | None, rng: random.Random) -> float:
    """Delay before the next attempt.

    Honors a server ``Retry-After`` up to a cap; otherwise full jitter over
    exponential backoff, so a shared endpoint under load does not get a
    synchronized retry burst from this runner.
    """
    if retry_after is not None:
        return min(max(retry_after, 0.0), RETRY_AFTER_CAP_SECONDS)
    return rng.uniform(0.0, min(BACKOFF_CAP_SECONDS, 2.0**attempt))


# ── pause control ─────────────────────────────────────────────────────────────
class PauseController:
    """Decides when to stop scheduling new work against a shared endpoint.

    Two independent triggers. The retry-rate trigger is live from the first
    request. The latency trigger only arms after three consecutive clean
    calibration windows produce a frozen baseline — comparing against an
    un-calibrated baseline would either never fire or fire constantly.
    """

    def __init__(self) -> None:
        self.attempts: list[bool] = []
        self.calibration_latencies: list[float] = []
        self.calibration_p95s: list[float] = []
        self.baseline: float | None = None
        self.window_latencies: list[float] = []
        self.consecutive_breaches = 0
        self.paused_reason: str | None = None

    def record_attempt(self, retryable: bool) -> None:
        self.attempts.append(retryable)
        window = self.attempts[-RETRY_RATE_WINDOW:]
        if len(window) >= RETRY_RATE_MINIMUM:
            if sum(window) / len(window) > RETRY_RATE_THRESHOLD:
                self.paused_reason = "retry_rate"

    def record_latency(self, seconds: float) -> None:
        if self.baseline is None:
            self.calibration_latencies.append(seconds)
            if len(self.calibration_latencies) == CALIBRATION_WINDOW:
                self.calibration_p95s.append(nearest_rank_p95(self.calibration_latencies))
                self.calibration_latencies = []
                if len(self.calibration_p95s) == CALIBRATION_WINDOWS:
                    self.baseline = sorted(self.calibration_p95s)[CALIBRATION_WINDOWS // 2]
            return

        self.window_latencies.append(seconds)
        if len(self.window_latencies) < LATENCY_WINDOW:
            return
        breached = nearest_rank_p95(self.window_latencies) > self.baseline * LATENCY_MULTIPLIER
        self.window_latencies = []
        self.consecutive_breaches = self.consecutive_breaches + 1 if breached else 0
        if self.consecutive_breaches >= LATENCY_BREACH_WINDOWS:
            self.paused_reason = "latency"

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    def should_pause(self) -> str | None:
        return self.paused_reason


def nearest_rank_p95(values: Sequence[float]) -> float:
    """Nearest-rank p95. No interpolation, so the result is an observed value."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-0.95 * len(ordered) // 1))))
    return ordered[rank - 1]


# ── journal ───────────────────────────────────────────────────────────────────
class PredictionJournal:
    """Append-only, hash-chained record of every attempt ever made."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("ab") as stream:
            stream.write(canonical_bytes(dict(record)))
            stream.flush()
            os.fsync(stream.fileno())

    def start_attempt(self, identity_digest: str, attempt: int, *, timestamp: str) -> None:
        """Record an intent to send, durably, BEFORE sending.

        This ordering is what makes a crash detectable. The alternative — record
        after the response — makes an interrupted request indistinguishable from
        one never sent.
        """
        self._append(
            {
                "record": ATTEMPT_START,
                "identity": identity_digest,
                "attempt": attempt,
                "timestamp": timestamp,
            }
        )

    def end_attempt(
        self,
        identity_digest: str,
        attempt: int,
        *,
        outcome: str,
        timestamp: str,
        status: int | None = None,
        latency: float = 0.0,
        content: str | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "record": ATTEMPT_END,
            "identity": identity_digest,
            "attempt": attempt,
            "outcome": outcome,
            "timestamp": timestamp,
            "status": status,
            "latency": latency,
        }
        if content is not None:
            record["content"] = content
            record["content_digest"] = hashlib.sha256(
                PREDICTION_EVENT_DOMAIN + content.encode("utf-8")
            ).hexdigest()
        if usage is not None:
            record["usage"] = dict(usage)
        self._append(record)

    def records(self) -> list[dict[str, Any]]:
        try:
            raw = self.path.read_bytes()
        except OSError:
            return []
        records = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                raise InferenceError(f"{self.path}:{number} is blank")
            records.append(json.loads(line))
        return records


@dataclass
class JournalState:
    """What a journal says about every identity it has touched."""

    canonical: dict[str, dict[str, Any]] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    exhausted: list[str] = field(default_factory=list)
    unknown_outcomes: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)

    @property
    def complete_identities(self) -> set[str]:
        return set(self.canonical)


def load_journal(path: Path) -> JournalState:
    """Replay a journal into the state the runner and scorer both read.

    Only the FIRST durably recorded success per identity is canonical; a second
    success for the same identity is corruption, not an update.
    """
    state = JournalState()
    open_attempts: dict[tuple[str, int], dict[str, Any]] = {}

    for record in PredictionJournal(path).records():
        identity = record.get("identity")
        attempt = record.get("attempt")
        if record.get("record") == ATTEMPT_START:
            open_attempts[(identity, attempt)] = record
            state.attempts[identity] = max(state.attempts.get(identity, 0), int(attempt))
            continue

        open_attempts.pop((identity, attempt), None)
        outcome = record.get("outcome")
        if outcome == OUTCOME_SUCCESS:
            if identity in state.canonical:
                state.duplicates.append(identity)
            else:
                state.canonical[identity] = record
        elif outcome == UNKNOWN_REMOTE_OUTCOME:
            state.unknown_outcomes.append(identity)

    # An attempt that started and never ended is an unknown remote outcome: the
    # request may or may not have reached the endpoint.
    for identity, _ in open_attempts:
        if identity not in state.canonical:
            state.unknown_outcomes.append(identity)

    for identity, count in state.attempts.items():
        if identity not in state.canonical and count >= MAX_ATTEMPTS:
            state.exhausted.append(identity)

    return state


def validate_completeness(state: JournalState, expected: Iterable[str]) -> dict[str, Any]:
    """Report whether a phase may emit a final metric."""
    expected_set = set(expected)
    missing = sorted(expected_set - state.complete_identities)
    return {
        "expected": len(expected_set),
        "complete": len(expected_set & state.complete_identities),
        "missing": len(missing),
        "duplicate_successes": len(state.duplicates),
        "exhausted": len(state.exhausted),
        "unknown_remote_outcomes": len(state.unknown_outcomes),
        "final_eligible": (not missing and not state.duplicates and not state.exhausted),
    }


def score_journal(protocol: Any, state: JournalState, *, final: bool = False) -> str:
    """Render a report from journal state. Offline; no model or source access."""
    expected = protocol.payload.get("expected_counts", {}).get("identities", [])
    summary = validate_completeness(state, expected)
    if final and not summary["final_eligible"]:
        raise InferenceError(
            f"cannot emit a final metric: {summary['missing']} missing, "
            f"{summary['duplicate_successes']} duplicate, {summary['exhausted']} exhausted"
        )
    return json.dumps({"completeness": summary}, indent=2, sort_keys=True)


# ── transport ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SendResult:
    """One transport outcome, independent of which client produced it."""

    status: int | None = None
    content: str | None = None
    usage: Mapping[str, Any] | None = None
    retry_after: float | None = None
    error: BaseException | None = None


def create_client(base_url: str, api_key: str) -> Any:
    """Build the OpenAI client with retries DISABLED.

    The SDK retries by default, silently and unjournaled. This module is the
    sole retry owner, so every attempt appears in the journal; a hidden layer
    would make the attempt counts and the disclosed at-least-once semantics
    wrong.
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)


@dataclass(frozen=True)
class ModelArm:
    """One side of the comparison."""

    key: str
    model_ref: str
    artifact_digest: str


# ── runner ────────────────────────────────────────────────────────────────────
class PairedRunner:
    """Issues paired base/SFT requests and journals every attempt.

    ``send`` is injected so the whole retry, pause, and durability contract is
    exercisable against a fake endpoint with no network and no credentials.
    """

    def __init__(
        self,
        protocol: Any,
        journal: PredictionJournal,
        arms: Sequence[ModelArm],
        send: Any,
        *,
        state: JournalState | None = None,
        concurrency: int = START_CONCURRENCY,
        rng: random.Random | None = None,
        clock: Any = time.monotonic,
        sleep: Any = asyncio.sleep,
        timestamp: str = "1970-01-01T00:00:00Z",
    ) -> None:
        if concurrency > MAX_CONCURRENCY:
            raise InferenceError(f"concurrency {concurrency} exceeds the cap of {MAX_CONCURRENCY}")
        self.protocol = protocol
        self.journal = journal
        self.arms = list(arms)
        self.send = send
        self.state = state or JournalState()
        self.pause = PauseController()
        self.rng = rng or random.Random(0)
        self.clock = clock
        self.sleep = sleep
        self.timestamp = timestamp
        self._semaphore = asyncio.Semaphore(concurrency)
        self.max_observed_concurrency = 0
        self._in_flight = 0
        self.issued_order: list[tuple[str, str]] = []

    def identity_for(self, sample: Sample, arm: ModelArm, payload_hash: str) -> RequestIdentity:
        return RequestIdentity(
            protocol_id=self.protocol.protocol_id,
            benchmark_revision=self.protocol.benchmark_revision,
            task_name=sample.task_name,
            sample_uid=sample.sample_uid,
            request_payload_hash=payload_hash,
            model_ref=arm.model_ref,
            model_artifact_digest=arm.artifact_digest,
        )

    def _ordered_arms(self, sample: Sample) -> list[ModelArm]:
        base_first = sample.base_first()
        ordered = sorted(self.arms, key=lambda arm: arm.key != BASE_MODEL_KEY)
        return ordered if base_first else list(reversed(ordered))

    async def _attempt_one(self, identity: RequestIdentity, request: Mapping[str, Any]) -> bool:
        """Run the retry loop for one identity. Returns True on canonical success."""
        digest = identity.digest
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.journal.start_attempt(digest, attempt, timestamp=self.timestamp)
            started = self.clock()
            try:
                result = await self.send(dict(request))
            except BaseException as exc:  # transport failure
                result = SendResult(error=exc)
            latency = self.clock() - started

            outcome = classify_response(result.status, result.error)
            self.journal.end_attempt(
                digest,
                attempt,
                outcome=outcome,
                timestamp=self.timestamp,
                status=result.status,
                latency=latency,
                content=result.content if outcome == OUTCOME_SUCCESS else None,
                usage=result.usage,
            )
            self.pause.record_attempt(outcome == OUTCOME_RETRYABLE)

            if outcome == OUTCOME_SUCCESS:
                # Invalid model CONTENT is a completed prediction, never a retry
                # trigger: retrying until the model says something parseable
                # would be tuning the result.
                self.state.canonical[digest] = {"identity": digest, "content": result.content}
                self.pause.record_latency(latency)
                return True
            if outcome == OUTCOME_FATAL:
                return False
            if attempt < MAX_ATTEMPTS:
                await self.sleep(retry_delay(attempt, result.retry_after, self.rng))

        self.state.exhausted.append(digest)
        return False

    async def _run_sample(self, sample: Sample) -> None:
        requests = {
            arm.key: build_request(self.protocol, sample.source_text, arm.model_ref)
            for arm in self.arms
        }
        base = requests.get(BASE_MODEL_KEY)
        sft = requests.get(SFT_MODEL_KEY)
        payload_hash = (
            assert_request_parity(base, sft)
            if base and sft
            else request_payload_hash(next(iter(requests.values())))
        )

        for arm in self._ordered_arms(sample):
            identity = self.identity_for(sample, arm, payload_hash)
            if identity.digest in self.state.canonical:
                continue  # an earlier phase already paid for this exact request
            if self.pause.should_pause():
                return
            async with self._semaphore:
                self._in_flight += 1
                self.max_observed_concurrency = max(self.max_observed_concurrency, self._in_flight)
                self.issued_order.append((sample.sample_uid, arm.key))
                try:
                    await self._attempt_one(identity, requests[arm.key])
                finally:
                    self._in_flight -= 1

    async def run(self, samples: Sequence[Sample]) -> dict[str, Any]:
        """Fill every missing identity for ``samples``, pausing if the endpoint degrades."""
        for sample in samples:
            if self.pause.should_pause():
                break
            await self._run_sample(sample)

        return {
            "paused": self.pause.should_pause(),
            "calibrated": self.pause.calibrated,
            "completed": len(self.state.canonical),
            "exhausted": len(self.state.exhausted),
            "max_concurrency": self.max_observed_concurrency,
        }


def attestation_fingerprint(
    *,
    model_listing: Sequence[str],
    source_revision: str,
    base_attestation: Mapping[str, Any],
    model_refs: Sequence[str],
) -> str:
    """One digest over everything that must not change across a resume."""
    return canonical_digest(
        [sorted(model_listing), source_revision, dict(base_attestation), sorted(model_refs)]
    )


def assert_resumable(previous: str, current: str) -> None:
    """Refuse to resume into a changed world."""
    if previous != current:
        raise InferenceError(
            "endpoint, source revision, base attestation, or model refs changed since the "
            "last run; resuming would mix incomparable responses"
        )


# ── CLI entry points ──────────────────────────────────────────────────────────
async def run_adapter_preflight(
    call: Any,
    *,
    model_ref: str,
    lora_path: str,
    expected_base_digest: str,
    expected_adapter_digest: str,
) -> dict[str, Any]:
    """Invoke the operator-only load-and-attest endpoint. Generates nothing.

    Externally mutating despite being idempotent: it populates a shared serving
    replica's adapter cache. The numeric ref is validated locally first, so a
    moving alias fails before any cache is warmed for weights nobody can name
    later. ``call`` is injected so the whole contract is testable offline.
    """
    validate_sft_model_ref(model_ref)

    response = await call(
        {
            "model_ref": model_ref,
            "lora_path": lora_path,
            "expected_base_manifest_digest": expected_base_digest,
            "expected_adapter_digest": expected_adapter_digest,
        }
    )

    if response.get("normalized"):
        raise InferenceError(
            "serving normalized the adapter config; the served adapter is not "
            "byte-identical to the trained one"
        )
    if response.get("cache_digest") != expected_adapter_digest:
        raise InferenceError(
            f"serving cache digest {response.get('cache_digest')} does not match the "
            f"identity digest {expected_adapter_digest}"
        )
    if response.get("blob_digest") not in (None, response.get("cache_digest")):
        raise InferenceError("downloaded blob and serving cache digests disagree")
    base = response.get("base") or {}
    if base.get("manifest_digest") != expected_base_digest:
        raise InferenceError(
            f"live base digest {base.get('manifest_digest')} does not match the frozen "
            f"{expected_base_digest}"
        )
    return response


def preflight_adapter(protocol: Any, protocol_dir: Any) -> dict[str, Any]:
    """CLI wrapper: needs a live endpoint and operator credentials."""
    sft_ref = str(protocol.payload.get("models", {}).get("sft_model_ref", ""))
    validate_sft_model_ref(sft_ref)
    raise SystemExit(
        "adapter preflight requires a live serving endpoint and operator credentials; "
        "run it from the execution worktree once the run has produced checkpoint "
        f"{sft_ref!r} (Slice 9 of the plan)"
    )


def evaluate(protocol: Any, protocol_dir: Any, *, phase: str) -> dict[str, Any]:
    """Fill the missing request identities for one phase. Paid."""
    raise SystemExit(
        f"the {phase} phase requires a live serving endpoint and credentials; "
        "run it from the execution worktree (Slices 9-11 of the plan)"
    )
