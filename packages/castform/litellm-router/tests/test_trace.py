from __future__ import annotations

from datetime import UTC, datetime

from castform_router.trace import append_trace, read_trace


def test_trace_round_trip_and_non_json_values(
    tmp_path: object,
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("CASTFORM_TRACE_DIR", str(tmp_path))

    append_trace(
        "trace-1",
        actor="test",
        stage="test.started",
        summary="A test event.",
        input={"created_at": datetime(2026, 1, 1, tzinfo=UTC)},
    )

    events = read_trace("trace-1")
    assert len(events) == 1
    assert events[0]["stage"] == "test.started"
    assert events[0]["input"]["created_at"].startswith("2026-01-01")


def test_invalid_trace_id_is_ignored(tmp_path: object, monkeypatch: object) -> None:
    monkeypatch.setenv("CASTFORM_TRACE_DIR", str(tmp_path))

    append_trace(
        "../escape",
        actor="test",
        stage="test.invalid",
        summary="This should not be written.",
    )

    assert list(tmp_path.iterdir()) == []
