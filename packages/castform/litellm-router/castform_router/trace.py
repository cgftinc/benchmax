"""Small shared JSONL trace store for the local visualizer.

This is intentionally a local development mechanism. Production should publish
redacted spans to an observability system instead of writing request bodies to
a shared filesystem.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRACE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def _trace_dir() -> Path:
    return Path(os.getenv("CASTFORM_TRACE_DIR", "/tmp/castform-traces"))


def valid_trace_id(trace_id: object) -> str | None:
    if isinstance(trace_id, str) and _TRACE_ID.fullmatch(trace_id):
        return trace_id
    return None


def append_trace(
    trace_id: object,
    *,
    actor: str,
    stage: str,
    summary: str,
    input: Any | None = None,
    output: Any | None = None,
    details: Any | None = None,
) -> None:
    """Append one event using a single O_APPEND write across local processes."""

    normalized_id = valid_trace_id(trace_id)
    if normalized_id is None:
        return

    now_ns = time.time_ns()
    event: dict[str, Any] = {
        "event_id": uuid.uuid4().hex,
        "trace_id": normalized_id,
        "timestamp": datetime.fromtimestamp(now_ns / 1_000_000_000, UTC).isoformat(),
        "timestamp_ns": now_ns,
        "actor": actor,
        "stage": stage,
        "summary": summary,
    }
    if input is not None:
        event["input"] = input
    if output is not None:
        event["output"] = output
    if details is not None:
        event["details"] = details

    directory = _trace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    ).encode()
    file_descriptor = os.open(
        directory / f"{normalized_id}.jsonl",
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(file_descriptor, encoded)
    finally:
        os.close(file_descriptor)


def read_trace(trace_id: object) -> list[dict[str, Any]]:
    normalized_id = valid_trace_id(trace_id)
    if normalized_id is None:
        return []
    path = _trace_dir() / f"{normalized_id}.jsonl"
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return sorted(
        events,
        key=lambda event: (
            int(event.get("timestamp_ns", 0)),
            str(event.get("event_id", "")),
        ),
    )
