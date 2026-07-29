"""Profile-aware Castform login and credential forwarding commands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from castform import config, profile_config
from castform.platform import credentials
from castform.platform.device_auth import DeviceAuthError
from castform.platform.login import _login


def _profile(args: argparse.Namespace) -> str:
    return profile_config.selected_profile_name(getattr(args, "profile", None))


def _cmd_login(args: argparse.Namespace) -> int:
    try:
        selected = _login(
            profile=getattr(args, "profile", None),
            domain=getattr(args, "domain", None),
            platform_url=getattr(args, "platform_url", None),
            llm_url=getattr(args, "llm_url", None),
            auth_url=getattr(args, "auth_url", None),
            app_url=getattr(args, "app_url", None),
        )
    except (DeviceAuthError, RuntimeError) as error:
        print(f"Login failed: {error}", file=sys.stderr)
        return 1
    print(f"\n✓ Logged in to profile {selected!r} ({config.profile_target(selected)}).")
    return 0


def _cmd_logout(args: argparse.Namespace) -> int:
    if getattr(args, "all", False):
        credentials.clear_castform_session(all_profiles=True)
        print("✓ Logged out of all profiles.")
        return 0
    selected = _profile(args)
    credentials.clear_castform_session(selected)
    print(f"✓ Logged out of profile {selected!r}.")
    return 0


def _cmd_whoami(args: argparse.Namespace) -> int:
    selected = _profile(args)
    session = credentials.read_castform_session(selected)
    if not session:
        print(
            f"Profile {selected!r} is not logged in. Run `castform login --profile {selected}`.",
            file=sys.stderr,
        )
        return 1
    jwt = credentials._session_jwt(selected)
    if not jwt:
        print(
            "Session present, but it is expired or auth-service is unavailable. "
            f"Run `castform login --profile {selected}` again.",
            file=sys.stderr,
        )
        return 1
    claims = credentials._jwt_claims(jwt)
    who = claims.get("email") or claims.get("sub", "<unknown>")
    print(f"Logged in as {who} (profile {selected!r}, {config.profile_target(selected)}).")
    return 0


def _cmd_profile_list(_args: argparse.Namespace) -> int:
    active = profile_config.selected_profile_name()
    for name, profile in profile_config.stored_profiles().items():
        marker = "*" if name == active else " "
        try:
            credentials.session_auth_token(name)
        except RuntimeError:
            status = "not logged in"
        else:
            status = "signed in"
        target = profile.get("domain") or profile.get("platform_url") or "<incomplete>"
        print(f"{marker} {name}\t{target}\t{status}")
    return 0


def _cmd_profile_current(_args: argparse.Namespace) -> int:
    selected = profile_config.selected_profile_name()
    profile = profile_config.get_profile(selected)
    if not profile:
        print(f"Profile {selected!r} is not configured.", file=sys.stderr)
        return 1
    target = profile.get("domain") or profile.get("platform_url")
    print(f"{selected}\t{target}")
    return 0


def _cmd_profile_activate(args: argparse.Namespace) -> int:
    try:
        profile_config.activate_profile(args.name)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"✓ Activated profile {args.name!r}.")
    return 0


def _cmd_with_auth(args: argparse.Namespace) -> int:
    """Run a child command with the selected login session in its environment."""
    child_command = list(args.child_command)
    if child_command[:1] == ["--"]:
        child_command = child_command[1:]
    if not child_command:
        print("with-auth requires a command after --", file=sys.stderr)
        return 2
    selected = _profile(args)
    try:
        token = credentials.session_auth_token(selected)
        platform_url = config.platform_url(selected)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    environment = os.environ.copy()
    environment["CASTFORM_PROFILE"] = selected
    environment["CASTFORM_AUTH_TOKEN"] = token
    environment["CASTFORM_PLATFORM_URL"] = platform_url
    try:
        completed = subprocess.run(child_command, env=environment, check=False)
    except FileNotFoundError:
        print(f"Command not found: {child_command[0]}", file=sys.stderr)
        return 127
    return completed.returncode


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        help="Use a named Castform profile for this command",
    )


def register(sub: argparse._SubParsersAction) -> None:
    login = sub.add_parser("login", help="Sign in via your browser")
    _add_profile_argument(login)
    login.add_argument("--domain", help="Domain shortcut, e.g. castform.dev")
    login.add_argument("--api-url", dest="platform_url", help="Platform API URL override")
    login.add_argument("--llm-url", help="LLM API URL override")
    login.add_argument("--auth-url", help="Auth-service URL override")
    login.add_argument("--app-url", help="Web application URL override")
    login.set_defaults(func=_cmd_login)

    logout = sub.add_parser("logout", help="Clear cached session(s)")
    _add_profile_argument(logout)
    logout.add_argument("--all", action="store_true", help="Log out of every profile")
    logout.set_defaults(func=_cmd_logout)

    whoami = sub.add_parser("whoami", help="Show the current login")
    _add_profile_argument(whoami)
    whoami.set_defaults(func=_cmd_whoami)

    profile = sub.add_parser("profile", help="Manage Castform profiles")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)
    profile_sub.add_parser("list", help="List configured profiles").set_defaults(
        func=_cmd_profile_list
    )
    profile_sub.add_parser("current", help="Show the active profile").set_defaults(
        func=_cmd_profile_current
    )
    activate = profile_sub.add_parser("activate", help="Change the active profile")
    activate.add_argument("name")
    activate.set_defaults(func=_cmd_profile_activate)

    with_auth = sub.add_parser("with-auth", help="Run a command with the selected login credential")
    _add_profile_argument(with_auth)
    with_auth.add_argument(
        "child_command",
        nargs=argparse.REMAINDER,
        help="Command following --",
    )
    with_auth.set_defaults(func=_cmd_with_auth)
