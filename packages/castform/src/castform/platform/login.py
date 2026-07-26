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

from . import credentials
from .browser import maybe_open_browser
from .device_auth import poll_for_token, request_device_code


def _login(
    *,
    profile: str | None = None,
    domain: str | None = None,
    platform_url: str | None = None,
    llm_url: str | None = None,
    auth_url: str | None = None,
    app_url: str | None = None,
) -> str:
    """Run device login for one profile and return its name."""
    from castform import config, profile_config

    selected = profile_config.selected_profile_name(profile)
    if any((domain, platform_url, llm_url, auth_url, app_url)):
        profile_config.upsert_profile(
            selected,
            domain=domain,
            platform_url=platform_url,
            llm_url=llm_url,
            auth_url=auth_url,
            app_url=app_url,
        )
    elif profile_config.get_profile(selected) is None:
        raise RuntimeError(
            f"Profile {selected!r} is not configured. Pass --domain or explicit service URLs."
        )

    auth = config.auth_url(selected)
    dc = request_device_code(auth)
    verification = dc.get("verification_uri_complete") or dc["verification_uri"]
    print(f"\nTo sign in, open this URL in your browser:\n\n    {verification}\n")
    print(f"and confirm this code:  {dc['user_code']}\n")
    maybe_open_browser(verification)
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
    credentials.write_castform_session(session, selected, auth_url=auth)
    return selected


def ensure_session(*, interactive: bool | None = None) -> None:
    """Make sure a platform credential is available; auto-login if interactive.

    No-op when one already resolves (``ACT_AS_TOKEN_PATH`` / ``CASTFORM_API_KEY`` /
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
