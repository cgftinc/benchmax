"""``castform`` CLI — browser-based login for the SDK.

Commands: ``login`` (device authorization), ``logout``, ``whoami``. The login
flow + the reusable ``ensure_session`` live in :mod:`benchmax.platform.login`;
this module is the thin argparse wrapper. After ``castform login`` the SDK
resolves its bearer from ``~/.castform`` automatically — no API key or URL.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys

from benchmax.platform import credentials
from benchmax.platform.device_auth import DeviceAuthError
from benchmax.platform.login import _login


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
