"""Tests for the shared HTTP retry utility."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from castform.traces.http import request_with_retry


def _make_response(status_code: int, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=resp
        )
    return resp


class TestRetryCount:
    def test_max_retries_gives_correct_total_attempts(self):
        """max_retries=3 should make 4 total attempts (1 initial + 3 retries)."""
        resp_429 = _make_response(429)
        with (
            patch("httpx.request", return_value=resp_429) as mock_req,
            patch("time.sleep"),
        ):
            result = request_with_retry("GET", "http://test", max_retries=3)
        assert mock_req.call_count == 4
        assert result.status_code == 429

    def test_succeeds_on_first_try(self):
        resp_200 = _make_response(200)
        with patch("httpx.request", return_value=resp_200) as mock_req:
            result = request_with_retry("GET", "http://test")
        assert mock_req.call_count == 1
        assert result.status_code == 200

    def test_succeeds_after_transient_failure(self):
        resp_429 = _make_response(429)
        resp_200 = _make_response(200)
        with (
            patch("httpx.request", side_effect=[resp_429, resp_429, resp_200]) as mock_req,
            patch("time.sleep"),
        ):
            result = request_with_retry("GET", "http://test", max_retries=5)
        assert mock_req.call_count == 3
        assert result.status_code == 200


class TestLastAttemptNoSleep:
    def test_does_not_sleep_after_last_attempt(self):
        """On the final attempt, should NOT sleep before returning."""
        resp_429 = _make_response(429)
        with (
            patch("httpx.request", return_value=resp_429),
            patch("time.sleep") as mock_sleep,
        ):
            request_with_retry("GET", "http://test", max_retries=2)
        # 3 attempts total (0, 1, 2). Sleep after attempts 0 and 1, NOT after 2.
        assert mock_sleep.call_count == 2


class TestRetryAfterHeader:
    def test_respects_retry_after_on_429(self):
        resp_429 = _make_response(429, headers={"retry-after": "30"})
        resp_200 = _make_response(200)
        with (
            patch("httpx.request", side_effect=[resp_429, resp_200]),
            patch("time.sleep") as mock_sleep,
        ):
            result = request_with_retry("GET", "http://test")
        assert result.status_code == 200
        mock_sleep.assert_called_once_with(30.0)

    def test_falls_back_to_exponential_without_header(self):
        resp_429 = _make_response(429)
        resp_200 = _make_response(200)
        with (
            patch("httpx.request", side_effect=[resp_429, resp_200]),
            patch("time.sleep") as mock_sleep,
        ):
            request_with_retry("GET", "http://test")
        # Attempt 0 backoff: 2^0 + jitter = ~1-2s
        wait = mock_sleep.call_args[0][0]
        assert 1.0 <= wait <= 2.0


class TestStatusCodeHandling:
    def test_retryable_status_returned_after_exhausting_retries(self):
        """429 response should be returned (not raised) after retries exhaust."""
        resp_429 = _make_response(429)
        with (
            patch("httpx.request", return_value=resp_429),
            patch("time.sleep"),
        ):
            result = request_with_retry("GET", "http://test", max_retries=1)
        assert result.status_code == 429

    def test_non_retryable_status_returns_immediately(self):
        """400 should return immediately without retrying."""
        resp_400 = _make_response(400)
        with patch("httpx.request", return_value=resp_400) as mock_req:
            result = request_with_retry("GET", "http://test", max_retries=5)
        assert mock_req.call_count == 1
        assert result.status_code == 400

    def test_retries_on_all_retryable_codes(self):
        for code in (429, 502, 503, 504):
            resp = _make_response(code)
            resp_200 = _make_response(200)
            with (
                patch("httpx.request", side_effect=[resp, resp_200]),
                patch("time.sleep"),
            ):
                result = request_with_retry("GET", "http://test")
            assert result.status_code == 200


class TestNetworkErrors:
    def test_retries_on_timeout(self):
        resp_200 = _make_response(200)
        with (
            patch(
                "httpx.request",
                side_effect=[httpx.TimeoutException("timeout"), resp_200],
            ),
            patch("time.sleep"),
        ):
            result = request_with_retry("GET", "http://test")
        assert result.status_code == 200

    def test_raises_after_exhausting_retries_on_timeout(self):
        with (
            patch("httpx.request", side_effect=httpx.TimeoutException("timeout")),
            patch("time.sleep"),
            pytest.raises(httpx.TimeoutException),
        ):
            request_with_retry("GET", "http://test", max_retries=2)
