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
from benchmax import profile_config as profiles
from benchmax.platform.device_auth import DeviceAuthError
from benchmax.platform.login import _login


def _selected_profile_from_args(args: argparse.Namespace) -> str:
    return getattr(args, "profile", None) or profiles.selected_profile_name()


def _cmd_login(args: argparse.Namespace) -> int:
    try:
        profile_name = _login(
            profile=getattr(args, "profile", None),
            domain=getattr(args, "domain", None),
            api_url=getattr(args, "api_url", None),
            llm_url=getattr(args, "llm_url", None),
            auth_url=getattr(args, "auth_url", None),
            app_url=getattr(args, "app_url", None),
        )
    except (DeviceAuthError, RuntimeError) as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    domain = config.base_domain(profile_name)
    print(f"\n✓ Logged in to profile {profile_name!r} ({domain}).")
    if profile_name != profiles.DEFAULT_PROFILE:
        print(f"Use it with: castform ... --profile {profile_name}")
        print(
            f'To make it default, set default_profile = "{profile_name}" in '
            f"{profiles.config_path()}."
        )
    return 0


def _cmd_logout(args: argparse.Namespace) -> int:
    if getattr(args, "all", False):
        credentials.clear_castform_session(all_profiles=True)
        print("✓ Logged out of all profiles.")
        return 0
    profile_name = _selected_profile_from_args(args)
    credentials.clear_castform_session(profile=profile_name)
    print(f"✓ Logged out of profile {profile_name!r}.")
    return 0


def _cmd_whoami(args: argparse.Namespace) -> int:
    profile_name = _selected_profile_from_args(args)
    session = credentials.read_castform_session(profile_name)
    if not session:
        print(
            f"Profile {profile_name!r} has no saved session. Run "
            f"`castform login --profile {profile_name} --domain <domain>`.",
            file=sys.stderr,
        )
        return 1
    jwt = credentials._session_jwt(profile_name)  # None if invalid/expired/offline
    if not jwt:
        print(
            "Session present, but couldn't reach auth-service to verify it "
            "(offline, or the session expired). If this persists, run "
            f"`castform login --profile {profile_name}` again.",
            file=sys.stderr,
        )
        return 1
    claims = credentials._jwt_claims(jwt)
    who = claims.get("email") or claims.get("sub", "<unknown>")
    print(f"Logged in as {who} (profile {profile_name!r}, {config.base_domain(profile_name)}).")
    return 0


def _cmd_profile_list(_args: argparse.Namespace) -> int:
    selected = profiles.select_profile()
    for name, profile in profiles.stored_profiles().items():
        marker = "*" if name == selected.name else " "
        has_session = "signed in" if profile.get("session") else "no session"
        domain = profile.get("domain", "<missing-domain>")
        overrides = [key for key in profiles.URL_FIELDS if profile.get(key)]
        suffix = f"; overrides: {', '.join(overrides)}" if overrides else ""
        print(f"{marker} {name}\t{domain}\t{has_session}{suffix}")
    return 0


def _cmd_profile_current(_args: argparse.Namespace) -> int:
    selected = profiles.select_profile()
    profile = profiles.get_profile(selected.name)
    if profile is None:
        print(f"{selected.name} ({selected.source}; not configured)")
        return 1
    print(f"{selected.name} ({selected.source}; {profile.get('domain', '<missing-domain>')})")
    return 0


def _cmd_config_show(_args: argparse.Namespace) -> int:
    cfg = profiles.load_config()
    print(f"default_profile = {cfg['default_profile']!r}")
    for name, profile in cfg["profiles"].items():
        print(f"\n[{name}]")
        for key in ("domain", *profiles.URL_FIELDS):
            if value := profile.get(key):
                print(f"{key} = {value}")
        print(f"session = {'present' if profile.get('session') else 'absent'}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach login/logout/whoami to the top-level subparsers."""
    login = sub.add_parser("login", help="Sign in via your browser")
    login.add_argument("--profile", help="Profile to create/update")
    login.add_argument("--domain", help="Base domain for a new/updated profile")
    login.add_argument("--api-url", help="Override API URL for this profile")
    login.add_argument("--llm-url", help="Override LLM URL for this profile")
    login.add_argument("--auth-url", help="Override auth URL for this profile")
    login.add_argument("--app-url", help="Override app URL for this profile")
    login.set_defaults(func=_cmd_login)

    logout = sub.add_parser("logout", help="Clear cached session(s)")
    logout.add_argument("--profile", help="Profile to clear")
    logout.add_argument("--all", action="store_true", help="Clear sessions for all profiles")
    logout.set_defaults(func=_cmd_logout)

    whoami = sub.add_parser("whoami", help="Show the current login")
    whoami.add_argument("--profile", help="Profile to inspect")
    whoami.set_defaults(func=_cmd_whoami)

    profile = sub.add_parser("profile", help="Inspect Castform profiles")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)
    profile_sub.add_parser("list", help="List profiles").set_defaults(
        func=_cmd_profile_list
    )
    profile_sub.add_parser("current", help="Show selected profile").set_defaults(
        func=_cmd_profile_current
    )

    config_p = sub.add_parser("config", help="Inspect Castform config")
    config_sub = config_p.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", help="Show config without secrets").set_defaults(
        func=_cmd_config_show
    )
