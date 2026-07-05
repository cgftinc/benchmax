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
from benchmax import profile_config as profiles


def _login(
    *,
    profile: str | None = None,
    domain: str | None = None,
    api_url: str | None = None,
    llm_url: str | None = None,
    auth_url: str | None = None,
    app_url: str | None = None,
) -> str:
    """Run the device flow and cache the session. Raises DeviceAuthError on failure."""
    profile_name = profile or profiles.selected_profile_name()
    existing = profiles.get_profile(profile_name)
    if existing is None and not domain and profile_name != profiles.DEFAULT_PROFILE:
        raise RuntimeError(
            f"Profile {profile_name!r} does not exist yet. Pass --domain when logging in."
        )
    # Make the profile resolvable for this login without writing a partial
    # config before the device flow succeeds.
    login_domain = domain or (existing or {}).get("domain") or profiles.DEFAULT_DOMAIN
    login_profile = dict(existing or {})
    login_profile["domain"] = login_domain
    if api_url:
        login_profile["api_url"] = api_url
    if llm_url:
        login_profile["llm_url"] = llm_url
    if auth_url:
        login_profile["auth_url"] = auth_url
    if app_url:
        login_profile["app_url"] = app_url

    auth = auth_url or login_profile.get("auth_url") or f"https://auth.{login_domain}"
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
    profiles.upsert_profile(
        profile_name,
        domain=login_domain,
        api_url=api_url,
        llm_url=llm_url,
        auth_url=auth_url,
        app_url=app_url,
        session=session,
    )
    return profile_name


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
