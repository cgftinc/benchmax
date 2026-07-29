"""Unit tests for the device authorization flow (RFC 8628 client)."""

from __future__ import annotations

import httpx
import pytest
from castform.platform.device_auth import (
    DeviceAuthError,
    poll_for_token,
    request_device_code,
)


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _post_returning(*responses):
    """A fake httpx.post that yields the given responses in order."""
    it = iter(responses)

    def _post(_url, **_kwargs):
        return next(it)

    return _post


def test_request_device_code_ok(monkeypatch):
    payload = {
        "device_code": "dc",
        "user_code": "ABCD1234",
        "verification_uri": "https://app.x/device",
        "interval": 5,
        "expires_in": 1800,
    }
    monkeypatch.setattr(httpx, "post", _post_returning(_Resp(200, payload)))
    assert request_device_code("https://auth.x")["user_code"] == "ABCD1234"


def test_request_device_code_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", _post_returning(_Resp(500, None, "boom")))
    with pytest.raises(DeviceAuthError, match="device-code request failed"):
        request_device_code("https://auth.x")


def test_poll_pending_then_success(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        _post_returning(
            _Resp(400, {"error": "authorization_pending"}),
            _Resp(400, {"error": "authorization_pending"}),
            _Resp(200, {"access_token": "sess_abc", "expires_in": 604800}),
        ),
    )
    slept: list[float] = []
    tok = poll_for_token("https://auth.x", "dc", interval=5, sleep=slept.append, now=lambda: 0.0)
    assert tok["access_token"] == "sess_abc"
    assert slept == [5, 5]  # slept once between each pending poll


def test_poll_slow_down_backs_off(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        _post_returning(
            _Resp(400, {"error": "slow_down"}),
            _Resp(200, {"access_token": "sess_abc"}),
        ),
    )
    slept: list[float] = []
    poll_for_token("https://auth.x", "dc", interval=5, sleep=slept.append, now=lambda: 0.0)
    assert slept == [10]  # interval bumped 5 -> 10 on slow_down


def test_poll_denied_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", _post_returning(_Resp(400, {"error": "access_denied"})))
    with pytest.raises(DeviceAuthError, match="access_denied"):
        poll_for_token("https://auth.x", "dc", sleep=lambda _s: None, now=lambda: 0.0)


def test_poll_expires_before_approval(monkeypatch):
    monkeypatch.setattr(
        httpx, "post", _post_returning(_Resp(400, {"error": "authorization_pending"}))
    )
    # now(): deadline calc (0), first while-check (0, enters), second check (past)
    times = iter([0.0, 0.0, 9999.0])
    with pytest.raises(DeviceAuthError, match="expired"):
        poll_for_token(
            "https://auth.x",
            "dc",
            interval=5,
            expires_in=1800,
            sleep=lambda _s: None,
            now=lambda: next(times),
        )
