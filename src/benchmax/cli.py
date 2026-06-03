"""``castform`` CLI — browser-based login for the SDK.

Commands: ``login`` (device authorization), ``logout``, ``whoami``. After
``castform login`` the SDK resolves its bearer from ``~/.castform`` automatically
(see :mod:`benchmax.platform.credentials`) — no API key or URL needed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

from benchmax.platform import credentials
from benchmax.platform.device_auth import (
    DeviceAuthError,
    poll_for_token,
    request_device_code,
)


def _auth_url(env: str | None) -> str:
    """Auth host for a login. There's no cached session yet to derive from, so
    map the chosen env directly; ``CASTFORM_AUTH_URL`` (local dev) always wins."""
    override = os.environ.get("CASTFORM_AUTH_URL")
    if override:
        return override
    base = os.environ.get("CASTFORM_BASE_DOMAIN") or (
        "castform.dev" if env == "staging" else "castform.com"
    )
    return f"https://auth.{base}"


def _login(env: str | None) -> None:
    """Run the device flow and cache the session. Raises DeviceAuthError on failure."""
    auth = _auth_url(env)
    dc = request_device_code(auth)
    verification = dc.get("verification_uri_complete") or dc["verification_uri"]
    print(f"\nTo sign in, open this URL in your browser:\n\n    {verification}\n")
    print(f"and confirm this code:  {dc['user_code']}\n")
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
    # prod carries no env marker (the default); only internal staging logins set it.
    if env == "staging":
        session["env"] = "staging"
    credentials.write_castform_session(session)


def ensure_session(env: str | None = None, *, interactive: bool | None = None) -> None:
    """Make sure a platform credential is available; auto-login if interactive.

    No-op when one already resolves (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY`` /
    a valid ``~/.castform`` session). Otherwise, on an interactive TTY (and unless
    ``CASTFORM_NO_AUTO_LOGIN`` is set), run the device flow. Headless callers fall
    through untouched so the downstream request fails with its own loud error.
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
    _login(env)


def _cmd_login(args: argparse.Namespace) -> int:
    env = "staging" if args.env == "staging" else None
    try:
        _login(env)
    except DeviceAuthError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    print(f"\n✓ Logged in to {args.env}.")
    return 0


def _cmd_logout(_args: argparse.Namespace) -> int:
    credentials.clear_castform_session()
    print("✓ Logged out.")
    return 0


def _cmd_whoami(_args: argparse.Namespace) -> int:
    session = credentials.read_castform_session()
    if not session:
        print("Not logged in. Run `castform login`.", file=sys.stderr)
        return 1
    env = session.get("env", "prod")
    jwt = credentials._session_jwt()  # mints from the session; None if invalid/expired
    if not jwt:
        print(
            f"Session present (env: {env}) but could not be verified — "
            "run `castform login` again.",
            file=sys.stderr,
        )
        return 1
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        who = claims.get("email") or claims.get("sub", "<unknown>")
    except Exception:
        who = "<unknown>"
    print(f"Logged in as {who} (env: {env}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="castform", description="Castform CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="Sign in via your browser")
    p_login.add_argument(
        "--env",
        choices=["prod", "staging"],
        default="prod",
        help="Environment to sign in to (staging is internal-only)",
    )
    p_login.set_defaults(func=_cmd_login)

    sub.add_parser("logout", help="Clear the cached session").set_defaults(func=_cmd_logout)
    sub.add_parser("whoami", help="Show the current login").set_defaults(func=_cmd_whoami)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
