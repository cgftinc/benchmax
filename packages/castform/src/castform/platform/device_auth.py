"""OAuth 2.0 device authorization flow (RFC 8628) against auth-service.

Used by ``castform login``: request a device+user code, then poll the token
endpoint until the user approves in a browser. Interactive only — headless/CI
never reach this (they use ``PLATFORM_API_KEY`` / ``ACT_AS_TOKEN_PATH``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

DEFAULT_CLIENT_ID = "castform-cli"


class DeviceAuthError(RuntimeError):
    """The device flow failed (denied, expired, or auth-service error)."""


def request_device_code(auth_url: str, client_id: str = DEFAULT_CLIENT_ID) -> dict[str, Any]:
    """Start the flow: returns device_code, user_code, verification_uri[_complete],
    expires_in, interval."""
    resp = httpx.post(
        f"{auth_url}/api/auth/device/code",
        json={"client_id": client_id},
        timeout=15.0,
    )
    if resp.status_code != 200:
        raise DeviceAuthError(
            f"device-code request failed: HTTP {resp.status_code} {resp.text[:200]}"
        )
    return resp.json()


def poll_for_token(
    auth_url: str,
    device_code: str,
    *,
    interval: int = 5,
    expires_in: int = 1800,
    client_id: str = DEFAULT_CLIENT_ID,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll ``/device/token`` until the user approves; return the token payload.

    Honors the RFC 8628 poll signals: ``authorization_pending`` (keep waiting),
    ``slow_down`` (back off the interval); anything else (``access_denied``,
    ``expired_token``, …) raises. ``sleep``/``now`` are injectable for tests.
    """
    deadline = now() + expires_in
    while now() < deadline:
        resp = httpx.post(
            f"{auth_url}/api/auth/device/token",
            json={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": client_id,
            },
            timeout=15.0,
        )
        data: dict[str, Any] = {}
        try:
            data = resp.json()
        except Exception:
            pass
        if data.get("access_token"):
            return data
        error = data.get("error") or f"HTTP {resp.status_code}"
        if error == "authorization_pending":
            sleep(interval)
            continue
        if error == "slow_down":
            interval += 5
            sleep(interval)
            continue
        raise DeviceAuthError(f"device authorization failed: {error}")
    raise DeviceAuthError("device code expired before it was approved")
