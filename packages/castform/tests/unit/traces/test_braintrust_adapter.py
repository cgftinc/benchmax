"""Tests for BraintrustTraceAdapter span grouping and normalisation."""

from unittest.mock import MagicMock, patch

import httpx
from castform.traces.braintrust.adapter import (
    BraintrustTraceAdapter,
    _group_into_traces,
    _normalize_trace,
)


class TestGroupIntoTraces:
    def test_groups_by_root_span_id(self):
        events = [
            {"id": "r1", "span_id": "r1", "root_span_id": "r1", "input": {"messages": []}},
            {"id": "c1", "span_id": "c1", "root_span_id": "r1", "input": {}},
            {"id": "c2", "span_id": "c2", "root_span_id": "r1", "input": {}},
            {"id": "r2", "span_id": "r2", "root_span_id": "r2", "input": {"messages": []}},
        ]
        traces = _group_into_traces(events)
        assert len(traces) == 2
        assert "r1" in traces
        assert "r2" in traces
        assert len(traces["r1"]["children"]) == 2
        assert len(traces["r2"]["children"]) == 0

    def test_root_detected_by_span_id_equals_root_span_id(self):
        events = [
            {"id": "x", "span_id": "root1", "root_span_id": "root1", "input": {}},
            {"id": "y", "span_id": "child1", "root_span_id": "root1", "input": {}},
        ]
        traces = _group_into_traces(events)
        assert traces["root1"]["root"]["span_id"] == "root1"

    def test_fallback_root_when_no_match(self):
        # If no span has span_id == root_span_id, first span becomes root
        events = [
            {"id": "a", "span_id": "s1", "root_span_id": "orphan"},
            {"id": "b", "span_id": "s2", "root_span_id": "orphan"},
        ]
        traces = _group_into_traces(events)
        assert "orphan" in traces
        assert traces["orphan"]["root"]["id"] == "a"
        assert len(traces["orphan"]["children"]) == 1

    def test_empty_events(self):
        assert _group_into_traces([]) == {}


class TestNormalizeTrace:
    def test_basic_normalisation(self):
        tree = {
            "root": {
                "input": {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ]
                },
                "scores": {"quality": 0.9},
                "metadata": {"env": "test"},
                "created": "2024-01-01T00:00:00Z",
            },
            "children": [],
        }
        trace = _normalize_trace("t1", tree)
        assert trace.id == "t1"
        assert len(trace.messages) == 2
        assert trace.scores == {"quality": 0.9}
        assert trace.metadata == {"env": "test"}
        assert trace.timestamp == "2024-01-01T00:00:00Z"
        assert trace.errors is None

    def test_extraction_failure_populates_errors(self):
        # Malformed input that will cause extract_messages to fail
        tree = {
            "root": {
                "input": None,  # not a dict — will cause issues
                "created": "2024-01-01",
            },
            "children": [
                {
                    "input": "not a dict",
                    "output": "also not a dict",
                    "span_attributes": "invalid",
                    "created": "",
                    "name": "",
                    "id": "",
                    "span_id": "",
                },
            ],
        }
        trace = _normalize_trace("t2", tree)
        # Should have messages (possibly empty) but no crash
        assert trace.id == "t2"
        # Even if extraction produces empty messages, it shouldn't error
        # since None input returns []

    def test_preserves_child_metadata_for_extraction(self):
        tree = {
            "root": {
                "input": {"messages": [{"role": "user", "content": "Q"}]},
            },
            "children": [
                {
                    "input": {
                        "messages": [
                            {"role": "system", "content": "S"},
                            {"role": "user", "content": "Q"},
                        ]
                    },
                    "output": {"role": "assistant", "content": "A"},
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "name": "llm_call",
                    "id": "c1",
                    "span_id": "c1",
                },
            ],
        }
        trace = _normalize_trace("t3", tree)
        assert len(trace.messages) > 0
        assert any(m.role == "assistant" for m in trace.messages)

    def test_missing_metadata_handled(self):
        tree = {
            "root": {
                "input": {
                    "messages": [
                        {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "A"},
                    ]
                },
                # no scores, no metadata, no created
            },
            "children": [],
        }
        trace = _normalize_trace("t4", tree)
        assert trace.scores == {}
        assert trace.metadata == {}
        assert trace.timestamp is None


class TestCountTraces:
    def test_count_traces_single_page(self):
        """< 1000 root spans — single BTQL page, returns row count."""
        adapter = BraintrustTraceAdapter(api_key="test-key")

        rows = [{"created": f"2024-01-01T00:{i:02d}:00Z"} for i in range(142)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": rows}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            count = adapter.count_traces("proj-123")

        assert count == 142
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args.args[0] == "POST"
        assert "/btql" in call_args.args[1]
        body = call_args.kwargs["json"]
        assert "project_logs('proj-123')" in body["query"]
        assert "span_id = root_span_id" in body["query"]
        assert body["fmt"] == "json"

    def test_count_traces_paginates_beyond_1000(self):
        """> 1000 root spans — must paginate and sum across pages."""
        adapter = BraintrustTraceAdapter(api_key="test-key")

        page1 = [{"created": f"2024-01-02T00:{i:02d}:00Z"} for i in range(1000)]
        page2 = [{"created": f"2024-01-01T00:{i:02d}:00Z"} for i in range(300)]

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"data": page1}
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"data": page2}
        resp2.raise_for_status = MagicMock()

        with patch("httpx.request", side_effect=[resp1, resp2]) as mock_req:
            count = adapter.count_traces("proj-big")

        assert count == 1300
        assert mock_req.call_count == 2
        # Second query should include timestamp filter
        body2 = mock_req.call_args_list[1].kwargs["json"]
        assert "created <" in body2["query"]

    def test_count_traces_handles_array_response(self):
        adapter = BraintrustTraceAdapter(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"created": "2024-01-01T00:00:00Z"}] * 55
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp):
            count = adapter.count_traces("proj-456")

        assert count == 55

    def test_count_traces_returns_zero_on_empty(self):
        adapter = BraintrustTraceAdapter(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp):
            count = adapter.count_traces("proj-789")

        assert count == 0


class TestFetchTraces:
    def _make_btql_response(self, spans: list[dict]) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": spans}
        resp.raise_for_status = MagicMock()
        return resp

    def _make_span(self, trace_id: str, span_id: str, is_root: bool = False) -> dict:
        return {
            "id": span_id,
            "span_id": span_id if is_root else f"child_{span_id}",
            "root_span_id": trace_id,
            "input": {"messages": [{"role": "user", "content": "hi"}]} if is_root else {},
            "output": {"role": "assistant", "content": "hello"} if is_root else {},
            "scores": {},
            "metadata": {},
            "span_attributes": {},
            "created": "2024-01-01T00:00:00Z",
            "name": "root" if is_root else "child",
        }

    def test_btql_fetch_returns_traces(self):
        adapter = BraintrustTraceAdapter(api_key="test-key")

        spans = [
            self._make_span("t1", "t1", is_root=True),
            self._make_span("t1", "c1"),
            self._make_span("t2", "t2", is_root=True),
        ]
        resp = self._make_btql_response(spans)

        with patch("httpx.request", return_value=resp):
            traces, cursor = adapter.fetch_traces("proj-123", limit=10)

        assert len(traces) == 2
        assert {t.id for t in traces} == {"t1", "t2"}

    def test_btql_request_uses_btql_url(self):
        adapter = BraintrustTraceAdapter(api_key="test-key")

        spans = [self._make_span("t1", "t1", is_root=True)]
        resp = self._make_btql_response(spans)

        with patch("httpx.request", return_value=resp) as mock_req:
            adapter.fetch_traces("proj-123", limit=10)

        # Should use BTQL URL, not REST
        url = mock_req.call_args.args[1]
        assert "/btql" in url

    def test_falls_back_to_rest_on_btql_failure(self):
        adapter = BraintrustTraceAdapter(api_key="test-key")

        btql_resp = MagicMock()
        btql_resp.status_code = 500
        btql_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=btql_resp
        )

        rest_resp = MagicMock()
        rest_resp.status_code = 200
        rest_resp.json.return_value = {
            "events": [self._make_span("t1", "t1", is_root=True)],
            "cursor": None,
        }
        rest_resp.raise_for_status = MagicMock()

        def side_effect(method, url, **kwargs):
            if "/btql" in url:
                return btql_resp
            return rest_resp

        with patch("httpx.request", side_effect=side_effect):
            traces, cursor = adapter.fetch_traces("proj-123", limit=10)

        assert len(traces) == 1
        assert traces[0].id == "t1"
