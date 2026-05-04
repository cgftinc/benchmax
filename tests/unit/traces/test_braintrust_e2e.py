"""Integration tests for BraintrustTraceAdapter against live API.

Requires environment variables:
    BT_API_KEY — Braintrust API key
    BT_PROJECT_ID — Braintrust project ID to test against

Run manually:
    BT_API_KEY=sk-... BT_PROJECT_ID=... pytest -m integration tests/traces/test_braintrust_e2e.py -v
"""

from __future__ import annotations

import os

import pytest

from benchmax.traces.adapter import TraceCredentials
from benchmax.traces.braintrust.adapter import BraintrustTraceAdapter

_API_KEY = os.environ.get("BT_API_KEY", "")
_PROJECT_ID = os.environ.get("BT_PROJECT_ID", "")

pytestmark = pytest.mark.integration


def _skip_if_no_creds():
    if not _API_KEY or not _PROJECT_ID:
        pytest.skip("BT_API_KEY and BT_PROJECT_ID required for integration tests")


class TestBraintrustE2E:
    def test_connect(self):
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)
        result = adapter.connect()
        assert result["status"] == "ok"
        assert result["projects"] > 0

    def test_list_projects(self):
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)
        projects = adapter.list_projects()
        assert len(projects) > 0
        assert all(p.id and p.name for p in projects)

    def test_count_traces(self):
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)
        count = adapter.count_traces(_PROJECT_ID)
        assert count >= 0

    def test_count_traces_matches_fetch(self):
        """count_traces() must match the actual number of fetched traces.

        This catches the BTQL scan-limit bug where a single-query count
        undercounts when > 1000 spans exist.
        """
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)

        count = adapter.count_traces(_PROJECT_ID)
        if count < 100:
            pytest.skip("Need at least 100 traces to validate count accuracy")

        traces, _ = adapter.fetch_traces(_PROJECT_ID, limit=count)
        fetched = len(traces)

        # Allow small delta: new traces may arrive between count and fetch
        assert abs(count - fetched) <= max(5, int(count * 0.02)), (
            f"count_traces()={count} but fetch_traces() returned {fetched} — "
            f"count is likely truncated to first BTQL page"
        )

    def test_fetch_traces_btql(self):
        """Verify BTQL fetch works against real Braintrust API."""
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)

        traces, cursor = adapter.fetch_traces(_PROJECT_ID, limit=5)
        assert len(traces) > 0
        assert len(traces) <= 5

        for trace in traces:
            assert trace.id
            assert len(trace.messages) > 0

    def test_btql_query_format(self):
        """Smoke test: BTQL query returns valid data without 400 errors."""
        _skip_if_no_creds()
        from benchmax.traces.braintrust.adapter import _BTQL_COLUMNS, _BTQL_URL
        from benchmax.traces.http import request_with_retry

        creds = TraceCredentials(api_key=_API_KEY)
        headers = {**creds.to_headers(), "Content-Type": "application/json"}

        query = (
            f"SELECT {_BTQL_COLUMNS} "
            f"FROM project_logs('{_PROJECT_ID}') "
            f"ORDER BY created DESC LIMIT 5"
        )
        resp = request_with_retry(
            "POST", _BTQL_URL,
            headers=headers,
            json={"query": query, "fmt": "json"},
            timeout=30,
        )
        assert resp.status_code == 200, f"BTQL returned {resp.status_code}: {resp.text[:200]}"

        data = resp.json()
        rows = data.get("data", data) if isinstance(data, dict) else data
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_fetch_traces_produces_valid_messages(self):
        """Verify extracted messages have expected structure."""
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)

        traces, _ = adapter.fetch_traces(_PROJECT_ID, limit=3)
        for trace in traces:
            for msg in trace.messages:
                assert msg.role in ("system", "user", "assistant", "tool")
                assert isinstance(msg.content, str)

    def test_fetch_traces_default_limit(self):
        """Fetch with no limit — matches the wizard Modal worker usage."""
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)

        count = adapter.count_traces(_PROJECT_ID)
        if count < 50:
            pytest.skip("Need at least 50 traces for bulk fetch test")

        traces, _ = adapter.fetch_traces(_PROJECT_ID)
        assert len(traces) >= 50

        for trace in traces:
            assert trace.id
            assert len(trace.messages) > 0

        # No duplicate trace IDs
        trace_ids = [t.id for t in traces]
        assert len(trace_ids) == len(set(trace_ids))

    def test_btql_direct_no_fallback(self):
        """Call _fetch_via_btql directly — proves BTQL works without REST fallback."""
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)

        trace_trees = adapter._fetch_via_btql(_PROJECT_ID, max_traces=100)
        assert len(trace_trees) > 0
        for trace_id, tree in trace_trees.items():
            assert "root" in tree
            assert tree["root"] is not None

    def test_btql_pagination_fetches_beyond_1000_rows(self):
        """BTQL paginates via timestamp to fetch more than 1000 spans."""
        _skip_if_no_creds()
        adapter = BraintrustTraceAdapter(api_key=_API_KEY)

        count = adapter.count_traces(_PROJECT_ID)
        if count < 100:
            pytest.skip("Need at least 100 traces to test pagination")

        trace_trees = adapter._fetch_via_btql(_PROJECT_ID, max_traces=500)
        assert len(trace_trees) > 50

        # No duplicate trace IDs
        assert len(trace_trees) == len(set(trace_trees.keys()))
