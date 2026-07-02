"""Device-auth login + session bootstrap for the SDK.

``ensure_session()`` is the single, explicit, TTY-gated auto-login entry. The
per-request credential seam (:func:`platform_bearer`) stays pure and fails
loudly; interactive browser login happens only here — called once up front by a
generated/hand-written script (or via ``castform login``). Library and
data-prep code never log you in: they resolve through the seam and error
clearly if no credential is present.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser

from . import credentials
from .device_auth import poll_for_token, request_device_code


def _maybe_open_browser(url: str) -> None:
    """Best-effort: open the verification URL so the user needn't copy it.

    The link and code are always printed above (the real path); opening is just
    convenience, and done immediately (a blocking prompt would stall the approval
    poll). Skipped when there's no useful local browser to open — a
    non-interactive stdin, an SSH session (the browser would spawn on the remote
    host, not the user's machine), or an explicit ``CASTFORM_NO_BROWSER`` opt-out
    — then the user opens the printed link themselves.
    """
    if (
        not sys.stdin.isatty()
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("CASTFORM_NO_BROWSER")
    ):
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass  # best-effort; the printed link is the fallback


def _login() -> None:
    """Run the device flow and cache the session. Raises DeviceAuthError on failure."""
    from benchmax import config

    auth = config.auth_url()
    dc = request_device_code(auth)
    verification = dc.get("verification_uri_complete") or dc["verification_uri"]
    print(f"\nTo sign in, open this URL in your browser:\n\n    {verification}\n")
    print(f"and confirm this code:  {dc['user_code']}\n")
    _maybe_open_browser(verification)
    print("Waiting for approval…")

    tok = poll_for_token(
        auth,
        dc["device_code"],
        interval=int(dc.get("interval", 5)),
        expires_in=int(dc.get("expires_in", 1800)),
    )

    session: dict[str, object] = {"access_token": tok["access_token"]}
    if tok.get("refresh_token"):
        session["refresh_token"] = tok["refresh_token"]
    if tok.get("expires_in"):
        session["expires_at"] = int(time.time()) + int(tok["expires_in"])
    credentials.write_castform_session(session)


def ensure_session(*, interactive: bool | None = None) -> None:
    """Make sure a platform credential is available; auto-login if interactive.

    No-op when one already resolves (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY`` /
    a valid ``~/.castform`` session). Otherwise, on an interactive TTY (and unless
    ``CASTFORM_NO_AUTO_LOGIN`` is set), run the device flow. Headless callers fall
    through untouched so the downstream request fails with its own loud error.

    Call this ONCE at the top of a script, before any platform interaction
    (corpus upload, QA generation, validate, upload, launch). It is the only
    place that triggers an interactive login.
    """
    try:
        credentials.platform_bearer()
        return
    except RuntimeError:
        pass
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive or os.environ.get("CASTFORM_NO_AUTO_LOGIN"):
        return
    _login()
