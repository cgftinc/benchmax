"""Tests for :mod:`benchmax.envs.capture`.

Covers the per-rollout capture contract: ContextVar binding, level + traceback
propagation, concurrent-asyncio isolation, LRU and per-rollout bounds, drain
helpers, counters, and edge cases (malformed records, large messages,
threading).
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from benchmax.envs import capture


@pytest.fixture(autouse=True)
def _fresh():
    """Reset capture state between tests."""
    capture._reset_for_tests()
    capture.install_capture(level=logging.DEBUG)  # tests want to see DEBUG too
    yield
    capture._reset_for_tests()


# --- Basic capture ----------------------------------------------------------


def test_log_inside_context_is_captured():
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid-1"):
        log.info("hello %s", "world")
    out = capture.pop(["rid-1"])
    assert len(out["rid-1"]) == 1
    rec = out["rid-1"][0]
    assert rec.message == "hello world"
    assert rec.level == "INFO"
    assert rec.level_no == logging.INFO
    assert rec.timestamp > 0
    assert rec.traceback is None


def test_log_outside_context_is_not_captured():
    log = logging.getLogger("benchmax.envs.test_capture")
    log.info("untagged")  # no rollout_context binding
    assert capture._peek_buffer_size() == 0


def test_levels_round_trip():
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        log.debug("d")
        log.info("i")
        log.warning("w")
        log.error("e")
    out = capture.pop(["rid"])
    levels = [(r.level, r.level_no) for r in out["rid"]]
    assert levels == [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
    ]


def test_exception_traceback_via_log_exception():
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("oops")
    out = capture.pop(["rid"])
    assert len(out["rid"]) == 1
    rec = out["rid"][0]
    assert rec.message == "oops"
    assert rec.level == "ERROR"
    assert rec.traceback is not None
    assert "ValueError: boom" in rec.traceback


def test_exception_traceback_via_exc_info_true():
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        try:
            raise RuntimeError("rt-fail")
        except RuntimeError:
            log.error("explicit exc_info", exc_info=True)
    out = capture.pop(["rid"])
    assert len(out["rid"]) == 1
    rec = out["rid"][0]
    assert "RuntimeError: rt-fail" in (rec.traceback or "")


# --- Concurrency isolation --------------------------------------------------


async def _emit_under_context(rid: str, message: str) -> None:
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context(rid):
        # Yield to event loop so multiple tasks really do interleave.
        await asyncio.sleep(0)
        log.info(message)
        await asyncio.sleep(0)
        log.info("%s-again", message)


async def test_concurrent_asyncio_tasks_route_correctly():
    await asyncio.gather(
        _emit_under_context("rid-a", "from-a"),
        _emit_under_context("rid-b", "from-b"),
        _emit_under_context("rid-c", "from-c"),
    )
    out = capture.pop(["rid-a", "rid-b", "rid-c"])
    assert [r.message for r in out["rid-a"]] == ["from-a", "from-a-again"]
    assert [r.message for r in out["rid-b"]] == ["from-b", "from-b-again"]
    assert [r.message for r in out["rid-c"]] == ["from-c", "from-c-again"]


def test_nested_same_rid_context_preserves_outer():
    """Re-entering rollout_context with the same rid is supported — outer
    binding is restored on exit."""
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        log.info("outer")
        with capture.rollout_context("rid"):
            log.info("inner")
        log.info("outer-again")
    out = capture.pop(["rid"])
    assert [r.message for r in out["rid"]] == ["outer", "inner", "outer-again"]


async def test_asyncio_to_thread_inherits_rollout_context():
    """``asyncio.to_thread`` copies the current Context — logs from the
    threaded function should still route to the current rollout."""
    log = logging.getLogger("benchmax.envs.test_capture")

    def _in_thread() -> None:
        log.info("from-thread")

    with capture.rollout_context("rid"):
        await asyncio.to_thread(_in_thread)
    out = capture.pop(["rid"])
    assert [r.message for r in out["rid"]] == ["from-thread"]


def test_raw_threading_thread_does_not_inherit_context():
    """A raw ``threading.Thread`` starts with a fresh Context — emits are
    not captured. Pinned to document the boundary explicitly."""
    log = logging.getLogger("benchmax.envs.test_capture")

    def _runner():
        log.info("from-raw-thread")

    with capture.rollout_context("rid"):
        t = threading.Thread(target=_runner)
        t.start()
        t.join()
    out = capture.pop(["rid"])
    assert out["rid"] == []


# --- Drain helpers ----------------------------------------------------------


def test_pop_drains():
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        log.info("first")
    assert capture._peek_buffer_size() == 1
    capture.pop(["rid"])
    assert capture._peek_buffer_size() == 0
    # Subsequent pop returns empty list, not KeyError.
    assert capture.pop(["rid"]) == {"rid": []}


def test_drop_clears_without_returning():
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        log.info("first")
    capture.drop(["rid"])
    assert capture._peek_buffer_size() == 0


def test_pop_for_unknown_rid_returns_empty():
    out = capture.pop(["unknown-rid"])
    assert out == {"unknown-rid": []}


# --- Bounds + counters ------------------------------------------------------


def test_lru_evicts_oldest_rollout_at_cap(monkeypatch):
    monkeypatch.setattr(capture, "MAX_PENDING_ROLLOUTS", 3)
    log = logging.getLogger("benchmax.envs.test_capture")
    for i in range(5):
        with capture.rollout_context(f"rid-{i}"):
            log.info("x")
    assert capture._peek_buffer_size() == 3
    out = capture.pop([f"rid-{i}" for i in range(5)])
    assert [k for k, v in out.items() if v] == ["rid-2", "rid-3", "rid-4"]
    assert capture.get_counters()["evicted_rollouts"] == 2


def test_lru_touches_recent_on_subsequent_log(monkeypatch):
    monkeypatch.setattr(capture, "MAX_PENDING_ROLLOUTS", 3)
    log = logging.getLogger("benchmax.envs.test_capture")
    for i in range(3):
        with capture.rollout_context(f"rid-{i}"):
            log.info("x")
    with capture.rollout_context("rid-0"):
        log.info("touch")
    with capture.rollout_context("rid-3"):
        log.info("x")
    out = capture.pop(["rid-0", "rid-1", "rid-2", "rid-3"])
    assert len(out["rid-0"]) == 2
    assert out["rid-1"] == []
    assert len(out["rid-2"]) == 1
    assert len(out["rid-3"]) == 1


def test_per_rollout_cap_drops_excess_and_counts(monkeypatch):
    monkeypatch.setattr(capture, "MAX_PENDING_PER_ROLLOUT", 5)
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        for i in range(20):
            log.info("line %d", i)
    out = capture.pop(["rid"])
    assert len(out["rid"]) == 5
    assert out["rid"][0].message == "line 0"
    assert out["rid"][-1].message == "line 4"
    assert capture.get_counters()["dropped_records"] == 15


def test_large_message_truncated_and_counted(monkeypatch):
    monkeypatch.setattr(capture, "MAX_MESSAGE_BYTES", 100)
    log = logging.getLogger("benchmax.envs.test_capture")
    with capture.rollout_context("rid"):
        log.info("x" * 500)
    out = capture.pop(["rid"])
    msg = out["rid"][0].message
    assert msg.endswith("…[truncated]")
    assert len(msg.encode("utf-8")) <= 100 + len(" …[truncated]".encode("utf-8"))
    assert capture.get_counters()["truncated_records"] == 1


def test_malformed_record_does_not_crash_and_counts():
    """``log.info("%d", "not int")`` raises in ``getMessage()`` — our handler
    must swallow and tick the emit_errors counter.

    We construct the LogRecord directly and feed it to the handler rather
    than going through ``log.info("%d", "x")``, because pytest's caplog
    handler crashes (correctly!) on the bad format string before ours runs.
    The handler-side behavior is what we care about here.
    """
    handler = next(
        h
        for h in logging.getLogger().handlers
        if isinstance(h, capture._CaptureHandler)
    )
    good = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="good", args=(), exc_info=None,
    )
    bad = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="%d", args=("not-an-int",), exc_info=None,
    )
    also_good = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="also good", args=(), exc_info=None,
    )
    with capture.rollout_context("rid"):
        handler.emit(good)
        handler.emit(bad)
        handler.emit(also_good)
    out = capture.pop(["rid"])
    assert [r.message for r in out["rid"]] == ["good", "also good"]
    assert capture.get_counters()["emit_errors"] == 1


# --- Install --------------------------------------------------------------


def test_install_capture_is_idempotent():
    handlers_before = [
        h for h in logging.getLogger().handlers if isinstance(h, capture._CaptureHandler)
    ]
    capture.install_capture()
    handlers_after = [
        h for h in logging.getLogger().handlers if isinstance(h, capture._CaptureHandler)
    ]
    assert len(handlers_before) == 1 == len(handlers_after)


def test_install_raises_root_level_but_does_not_lower():
    """If root is at WARNING, install at INFO raises it. If root is already
    at DEBUG (more permissive), install at INFO leaves it alone."""
    capture._reset_for_tests()
    root = logging.getLogger()
    original_level = root.level
    try:
        root.setLevel(logging.WARNING)
        capture.install_capture(level=logging.INFO)
        assert root.level == logging.INFO
        capture._reset_for_tests()
        root.setLevel(logging.DEBUG)
        capture.install_capture(level=logging.INFO)
        # DEBUG (10) < INFO (20) — leave the more permissive level alone.
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original_level)


def test_handler_does_not_block_other_handlers():
    captured_by_sibling: list[str] = []

    class _SiblingHandler(logging.Handler):
        def emit(self, record):
            captured_by_sibling.append(record.getMessage())

    sibling = _SiblingHandler(level=logging.DEBUG)
    logging.getLogger().addHandler(sibling)
    try:
        log = logging.getLogger("benchmax.envs.test_capture")
        with capture.rollout_context("rid"):
            log.info("seen by both")
        assert "seen by both" in captured_by_sibling
        assert len(capture.pop(["rid"])["rid"]) == 1
    finally:
        logging.getLogger().removeHandler(sibling)


def test_counters_start_at_zero():
    assert capture.get_counters() == {
        "evicted_rollouts": 0,
        "dropped_records": 0,
        "truncated_records": 0,
        "emit_errors": 0,
    }
