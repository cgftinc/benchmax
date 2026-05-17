"""Per-rollout capture of standard-library ``logging`` records.

The training pipeline runs each env method (``init_rollout`` / ``run_tool`` /
``compute_reward`` / ``release_rollout``) inside a :func:`rollout_context`
block that binds the current asyncio task to a rollout_id via a
:class:`~contextvars.ContextVar`. Any ``logging`` emit made from inside that
block — by env code, by libraries the env calls, by the exception path of
``logger.exception`` — is appended to a per-rollout buffer. The trainer
drains the buffer via :func:`pop` once per training batch and emits the lines
as ``rollout_env_logs`` rows alongside the parent ``rollout`` row.

Design notes:

- ContextVar is per-asyncio-task, so concurrent in-flight rollouts in the
  same env-actor process route their log records correctly without locks.
- The handler is installed on the **root** logger and filters by ContextVar
  presence. The user contract is "any logging during an env method body
  becomes an env_log" — that includes third-party libs the env method calls,
  which matches user intent. Threads spawned via ``asyncio.to_thread`` /
  ``loop.run_in_executor`` inherit the ContextVar via Python's standard
  context-copy semantics; raw ``threading.Thread`` does NOT (correct — its
  emits fall through).
- Handler level defaults to ``INFO`` to keep third-party DEBUG noise out of
  the buffer. Users wanting DEBUG capture pass ``level=logging.DEBUG`` to
  :func:`install_capture`. Root logger level is only raised (never lowered)
  so INFO emits actually reach handlers when the process default is WARNING.
- Buffer is bounded by both distinct-rollout count and per-rollout line
  count, plus a per-message byte cap. Excess is dropped silently with
  counters exposed via :func:`get_counters` so the trainer can surface
  loss-of-observability events.
"""

from __future__ import annotations

import logging
import traceback as _tb
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import time as _time

# --- Public surface ---------------------------------------------------------

__all__ = [
    "CapturedRecord",
    "install_capture",
    "rollout_context",
    "current_rollout_id",
    "pop",
    "drop",
    "get_counters",
    "MAX_PENDING_ROLLOUTS",
    "MAX_PENDING_PER_ROLLOUT",
    "MAX_MESSAGE_BYTES",
]

MAX_PENDING_ROLLOUTS = 4096
MAX_PENDING_PER_ROLLOUT = 1000
MAX_MESSAGE_BYTES = 64 * 1024  # 64 KiB per message; truncated past this.


@dataclass(frozen=True)
class CapturedRecord:
    """One captured log line.

    Mirrors the fields ingest's ``rollout_env_logs`` handler stores.
    ``traceback`` is ``None`` unless the originating call set ``exc_info``
    (e.g. ``logger.exception(...)``). ``timestamp`` is wall-clock seconds
    since epoch (``time.time()`` at emit). ``level_no`` is the standard
    Python numeric level (10/20/30/40/50) for callers that want to filter
    without string parsing.
    """

    message: str
    level: str
    level_no: int
    timestamp: float
    traceback: str | None = None


@dataclass
class _Counters:
    evicted_rollouts: int = 0          # whole-rollout LRU evictions
    dropped_records: int = 0           # per-rollout cap hits
    truncated_records: int = 0         # message exceeded MAX_MESSAGE_BYTES
    emit_errors: int = 0               # handler swallowed an emit exception


_counters = _Counters()


# --- Internal state ---------------------------------------------------------

_CURRENT_ROLLOUT_ID: ContextVar[str | None] = ContextVar(
    "benchmax_capture_rollout_id", default=None
)

# OrderedDict for LRU semantics: oldest rollout_id evicted first when the
# distinct-rollout count exceeds MAX_PENDING_ROLLOUTS.
_buffer: "OrderedDict[str, list[CapturedRecord]]" = OrderedDict()
_buffer_lock = Lock()

_installed = False


class _CaptureHandler(logging.Handler):
    """Root-level handler. Routes records into the per-rollout buffer when a
    rollout context is active; otherwise no-op (sibling handlers still see
    the record because we don't ``filter`` it out, just don't store it)."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        rid = _CURRENT_ROLLOUT_ID.get()
        if rid is None:
            return
        try:
            tb_text: str | None = None
            if record.exc_info:
                tb_text = "".join(_tb.format_exception(*record.exc_info)).rstrip()
            # record.getMessage() can raise if the format args don't match
            # the format string. We treat that as a soft failure — bump a
            # counter so the trainer can surface it, but never propagate.
            msg = record.getMessage()
            if len(msg.encode("utf-8", errors="replace")) > MAX_MESSAGE_BYTES:
                # Truncate at byte boundary, leave a marker so consumers know.
                truncated = msg.encode("utf-8", errors="replace")[:MAX_MESSAGE_BYTES]
                msg = truncated.decode("utf-8", errors="replace") + " …[truncated]"
                _counters.truncated_records += 1
            captured = CapturedRecord(
                message=msg,
                level=record.levelname,
                level_no=record.levelno,
                timestamp=record.created,
                traceback=tb_text,
            )
        except Exception:
            _counters.emit_errors += 1
            return
        _append(rid, captured)


def install_capture(*, level: int = logging.INFO) -> None:
    """Install the capture handler on the root logger. Idempotent.

    Args:
        level: minimum log level to capture. Defaults to ``logging.INFO`` to
            avoid pulling in third-party DEBUG chatter (httpx connection
            pool, etc.). Pass ``logging.DEBUG`` to capture everything; the
            per-rollout / per-process bounds still apply.

    Side effect: if the root logger's level is more restrictive than
    ``level`` (e.g. process default WARNING), it is raised to ``level`` so
    emits actually reach handlers. Other handlers in the process will see
    the same set of records as a consequence — this is a deliberate trade
    for ergonomic capture; users who need strict isolation should configure
    their own loggers with explicit ``addHandler`` + ``propagate=False``.
    """
    global _installed
    if _installed:
        return
    handler = _CaptureHandler(level=level)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    _installed = True


def rollout_context(rollout_id: str) -> AbstractContextManager[None]:
    """Bind ``rollout_id`` to the current asyncio task for the duration of
    the ``with`` block. Nested binds for the same task replace the outer
    binding for their scope and restore it on exit (standard ContextVar
    token semantics)."""
    return _RolloutContext(rollout_id)


def current_rollout_id() -> str | None:
    """Return the rollout_id bound to the current asyncio task, or ``None``
    if no rollout context is active. Useful for callers (e.g. error
    decorators) that want to take different paths depending on whether
    a record will be captured per-rollout or fall through to stderr."""
    return _CURRENT_ROLLOUT_ID.get()


def pop(rollout_ids: list[str]) -> dict[str, list[CapturedRecord]]:
    """Drain and return the captured records for each rollout_id. Missing
    rollout_ids return an empty list. Always pops — caller is the source of
    truth for which rollouts are "done"."""
    out: dict[str, list[CapturedRecord]] = {}
    with _buffer_lock:
        for rid in rollout_ids:
            out[rid] = _buffer.pop(rid, [])
    return out


def drop(rollout_ids: list[str]) -> None:
    """Discard pending records for the given rollout_ids (e.g. abandoned
    rerolls). No-op if a rid isn't in the buffer."""
    with _buffer_lock:
        for rid in rollout_ids:
            _buffer.pop(rid, None)


def get_counters() -> dict[str, int]:
    """Snapshot of capture-loss counters.

    Returned keys:
        ``evicted_rollouts``: whole rollouts LRU-evicted before they were
            popped (likely catastrophic for that rollout's logs).
        ``dropped_records``: log lines dropped because the per-rollout cap
            was hit. Other lines from that rollout still landed.
        ``truncated_records``: messages truncated to ``MAX_MESSAGE_BYTES``.
        ``emit_errors``: handler caught an unexpected exception while
            building a CapturedRecord (e.g. ``record.getMessage()`` raised
            on a format-string mismatch).

    Non-zero values indicate observability gaps. Surfacing them through the
    trainer's existing telemetry path is recommended.
    """
    return {
        "evicted_rollouts": _counters.evicted_rollouts,
        "dropped_records": _counters.dropped_records,
        "truncated_records": _counters.truncated_records,
        "emit_errors": _counters.emit_errors,
    }


# --- Internals --------------------------------------------------------------


class _RolloutContext(AbstractContextManager[None]):
    __slots__ = ("_rid", "_tok")

    def __init__(self, rid: str) -> None:
        self._rid = rid

    def __enter__(self) -> None:
        self._tok = _CURRENT_ROLLOUT_ID.set(self._rid)

    def __exit__(self, *exc: object) -> None:
        # Swallow LookupError/ValueError from reset() so we never mask the
        # original exception from the with-body. Token-from-wrong-context
        # only happens in pathological cases (CM reused, task escaped the
        # block) — best-effort clear is the right fallback.
        try:
            _CURRENT_ROLLOUT_ID.reset(self._tok)
        except (LookupError, ValueError):
            _CURRENT_ROLLOUT_ID.set(None)


def _append(rid: str, rec: CapturedRecord) -> None:
    with _buffer_lock:
        bucket = _buffer.get(rid)
        if bucket is None:
            # New rollout — inserts at end (OrderedDict insertion order) so
            # no explicit move_to_end needed.
            bucket = []
            _buffer[rid] = bucket
            # LRU: evict oldest distinct rollout when over cap.
            while len(_buffer) > MAX_PENDING_ROLLOUTS:
                _buffer.popitem(last=False)
                _counters.evicted_rollouts += 1
        else:
            # Touch for LRU recency.
            _buffer.move_to_end(rid)
        if len(bucket) < MAX_PENDING_PER_ROLLOUT:
            bucket.append(rec)
        else:
            _counters.dropped_records += 1


# --- Test-only helpers ------------------------------------------------------


def _reset_for_tests() -> None:
    """Clear all internal state. Tests only — not part of the public API."""
    global _installed, _counters
    with _buffer_lock:
        _buffer.clear()
    # Best-effort clear of the contextvar — fine even if no token is set
    # because tests run in a fresh task per pytest function by default.
    try:
        _CURRENT_ROLLOUT_ID.set(None)
    except Exception:
        pass
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, _CaptureHandler)]
    _installed = False
    _counters = _Counters()


def _peek_buffer_size() -> int:
    """Distinct rollout_id count currently in buffer. Tests only."""
    return len(_buffer)


def _snapshot_pending() -> Iterator[tuple[str, list[CapturedRecord]]]:
    """Snapshot of current buffer state. Tests only."""
    with _buffer_lock:
        return iter(list(_buffer.items()))
