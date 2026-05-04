"""Braintrust TraceAdapter implementation."""

from __future__ import annotations

import logging
from typing import Any

from benchmax.traces.adapter import (
    NormalizedTrace,
    TraceCredentials,
    TraceProject,
)
from benchmax.traces.braintrust.message_extraction import extract_messages, extract_scores
from benchmax.traces.http import request_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.braintrust.dev/v1"
_BTQL_URL = "https://api.braintrust.dev/btql"
_MAX_TRACES_PER_FETCH = 1000
_BTQL_COLUMNS = (
    "id, span_id, root_span_id, input, output, scores, metadata, span_attributes, created"
)


class BraintrustTraceAdapter:
    """Adapter for fetching and normalising traces from Braintrust.

    Args:
        api_key: Braintrust API key.
    """

    def __init__(self, api_key: str) -> None:
        self._credentials = TraceCredentials(api_key=api_key)

    def connect(self) -> dict[str, Any]:
        """Validate the API key by listing projects."""
        headers = self._credentials.to_headers()
        resp = request_with_retry("GET", f"{_BASE_URL}/project", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        objects = data.get("objects", []) if isinstance(data, dict) else []
        return {"status": "ok", "projects": len(objects)}

    def list_projects(self) -> list[TraceProject]:
        """List projects visible to the API key."""
        headers = self._credentials.to_headers()
        resp = request_with_retry("GET", f"{_BASE_URL}/project", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        objects = data.get("objects", data) if isinstance(data, dict) else data
        if not isinstance(objects, list):
            return []
        return [
            TraceProject(id=str(p["id"]), name=str(p.get("name", p["id"])))
            for p in objects
            if isinstance(p, dict) and "id" in p
        ]

    def count_traces(self, project_id: str) -> int:
        """Return total root-span count for a project using BTQL.

        BTQL scans at most 1000 rows per query, so a single
        ``SELECT count(*)`` undercounts when > 1000 spans exist.
        We paginate through root spans using timestamp ordering
        and sum across pages.
        """
        headers = {**self._credentials.to_headers(), "Content-Type": "application/json"}
        total = 0
        created_before: str | None = None

        while True:
            where = "WHERE span_id = root_span_id"
            if created_before:
                where += f" AND created < '{created_before}'"
            query = (
                f"SELECT created FROM project_logs('{project_id}') "
                f"{where} "
                f"ORDER BY created DESC "
                f"LIMIT 1000"
            )
            resp = request_with_retry(
                "POST",
                _BTQL_URL,
                headers=headers,
                json={"query": query, "fmt": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                break

            total += len(rows)
            created_before = rows[-1].get("created")

            if len(rows) < 1000:
                break

        return total

    def fetch_traces(
        self,
        project_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[NormalizedTrace], str | None]:
        """Fetch spans from Braintrust, group by root_span_id, and normalise.

        Uses BTQL (higher rate limits, larger page sizes) with automatic
        fallback to the REST API if BTQL fails.  BTQL always returns
        ``cursor=None`` (no cross-call pagination — increase ``limit``
        to fetch more).

        Returns ``(traces, next_cursor)``.
        """
        import httpx

        max_traces = min(limit, _MAX_TRACES_PER_FETCH) if limit else _MAX_TRACES_PER_FETCH

        try:
            trace_trees = self._fetch_via_btql(project_id, max_traces)
        except (httpx.HTTPError, httpx.TimeoutException, httpx.ConnectError, KeyError) as e:
            logger.warning("BTQL fetch failed, falling back to REST: %s", e)
            trace_trees, cursor = self._fetch_via_rest(project_id, max_traces, cursor)

        traces: list[NormalizedTrace] = []
        for trace_id, tree in trace_trees.items():
            traces.append(_normalize_trace(trace_id, tree))
            if len(traces) >= max_traces:
                break

        return traces, cursor

    def _fetch_via_btql(
        self,
        project_id: str,
        max_traces: int,
    ) -> dict[str, dict[str, Any]]:
        """Fetch spans using BTQL, return grouped trace trees.

        BTQL has a 1000-row max per query.  Paginates using
        ``WHERE created < '{last_timestamp}'`` to fetch additional pages.
        ``project_id`` is always from ``list_projects()`` — not user input.
        """
        headers = {**self._credentials.to_headers(), "Content-Type": "application/json"}
        all_rows: list[dict[str, Any]] = []
        created_before: str | None = None
        trace_trees: dict[str, dict[str, Any]] = {}

        while True:
            where = f"WHERE created < '{created_before}'" if created_before else ""
            query = (
                f"SELECT {_BTQL_COLUMNS} "
                f"FROM project_logs('{project_id}') "
                f"{where} "
                f"ORDER BY created DESC "
                f"LIMIT 1000"
            )
            resp = request_with_retry(
                "POST",
                _BTQL_URL,
                headers=headers,
                json={"query": query, "fmt": "json"},
                timeout=60,
            )
            if resp.status_code in (429, 502, 503, 504):
                # Retries exhausted — return what we have so far
                if all_rows:
                    logger.warning(
                        "BTQL pagination stopped at HTTP %d after %d rows",
                        resp.status_code,
                        len(all_rows),
                    )
                    break
                # First page failed — let caller handle the error
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                break

            all_rows.extend(rows)
            created_before = rows[-1].get("created")

            trace_trees = _group_into_traces(all_rows)
            if len(trace_trees) >= max_traces:
                break

            # Last page — fewer than 1000 rows means no more data
            if len(rows) < 1000:
                break

        if not trace_trees:
            trace_trees = _group_into_traces(all_rows)

        logger.info("BTQL fetch: %d spans, %d traces", len(all_rows), len(trace_trees))
        return trace_trees

    def _fetch_via_rest(
        self,
        project_id: str,
        max_traces: int,
        cursor: str | None,
    ) -> tuple[dict[str, dict[str, Any]], str | None]:
        """Fetch spans using REST API, return grouped trace trees. Fallback."""
        headers = {**self._credentials.to_headers(), "Content-Type": "application/json"}
        all_events: list[dict[str, Any]] = []
        page_cursor = cursor
        trace_trees: dict[str, dict[str, Any]] = {}

        while True:
            body: dict[str, Any] = {"limit": 100}
            if page_cursor is not None:
                body["cursor"] = page_cursor

            resp = request_with_retry(
                "POST",
                f"{_BASE_URL}/project_logs/{project_id}/fetch",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

            events = data.get("events", [])
            if not events:
                break

            all_events.extend(events)
            page_cursor = data.get("cursor")

            trace_trees = _group_into_traces(all_events)
            if len(trace_trees) >= max_traces:
                break
            if not page_cursor:
                break

        if not trace_trees:
            trace_trees = _group_into_traces(all_events)

        return trace_trees, page_cursor


def _group_into_traces(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group raw span events by ``root_span_id`` into trace trees.

    Returns ``{trace_id: {"root": root_span, "children": [child_spans]}}``.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        root_id = event.get("root_span_id", event.get("id", ""))
        groups.setdefault(root_id, []).append(event)

    traces: dict[str, dict[str, Any]] = {}
    for trace_id, spans in groups.items():
        root = None
        children: list[dict[str, Any]] = []
        for span in spans:
            if span.get("span_id") == span.get("root_span_id") or span.get("id") == trace_id:
                root = span
            else:
                children.append(span)
        if root is None and spans:
            root = spans[0]
            children = spans[1:]
        if root is not None:
            traces[trace_id] = {"root": root, "children": children}

    return traces


def _normalize_trace(trace_id: str, tree: dict[str, Any]) -> NormalizedTrace:
    """Convert a grouped trace tree into a ``NormalizedTrace``."""
    root = tree["root"]
    children = tree.get("children", [])

    trace_dict = {
        "input": root.get("input"),
        "output": root.get("output"),
        "scores": root.get("scores"),
        "children": [
            {
                "input": c.get("input"),
                "output": c.get("output"),
                "span_attributes": c.get("span_attributes"),
                "created": c.get("created", ""),
                "name": c.get("name", ""),
                "id": c.get("id", ""),
                "span_id": c.get("span_id", ""),
            }
            for c in children
        ],
    }

    errors: list[str] = []
    try:
        messages = extract_messages(trace_dict)
    except Exception as e:
        messages = []
        errors.append(f"Message extraction failed: {e}")

    scores = extract_scores(trace_dict)

    metadata: dict[str, Any] = {}
    if root.get("metadata"):
        metadata = dict(root["metadata"]) if isinstance(root["metadata"], dict) else {}

    return NormalizedTrace(
        id=trace_id,
        messages=messages,
        scores=scores,
        metadata=metadata,
        timestamp=root.get("created"),
        errors=errors if errors else None,
    )
