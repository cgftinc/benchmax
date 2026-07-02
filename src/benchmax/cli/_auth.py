"""castform auth commands: ``login`` / ``logout`` / ``whoami``.

The device-auth flow + the reusable ``ensure_session`` live in
:mod:`benchmax.platform.login`; these handlers are thin argparse wrappers. After
``castform login`` the SDK resolves its bearer from ``~/.castform`` automatically.
"""

from __future__ import annotations

import argparse
import sys

from benchmax import config
from benchmax.platform import credentials
from benchmax.platform.device_auth import DeviceAuthError
from benchmax.platform.login import _login


def _cmd_login(_args: argparse.Namespace) -> int:
    try:
        _login()
    except DeviceAuthError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    print(f"\n✓ Logged in to {config.base_domain()}.")
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
    jwt = credentials._session_jwt()  # None if invalid/expired/offline
    if not jwt:
        print(
            "Session present, but couldn't reach auth-service to verify it "
            "(offline, or the session expired). If this persists, run "
            "`castform login` again.",
            file=sys.stderr,
        )
        return 1
    claims = credentials._jwt_claims(jwt)
    who = claims.get("email") or claims.get("sub", "<unknown>")
    print(f"Logged in as {who} ({config.base_domain()}).")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach login/logout/whoami to the top-level subparsers."""
    sub.add_parser("login", help="Sign in via your browser").set_defaults(
        func=_cmd_login
    )
    sub.add_parser("logout", help="Clear the cached session").set_defaults(
        func=_cmd_logout
    )
    sub.add_parser("whoami", help="Show the current login").set_defaults(
        func=_cmd_whoami
    )
