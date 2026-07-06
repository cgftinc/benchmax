"""Shared platform-client wiring + error handling for CLI command groups.

Read/control commands resolve their bearer through the credential seam
(:func:`benchmax.platform.credentials.platform_bearer`) — ``ACT_AS_TOKEN_PATH``
→ ``PLATFORM_API_KEY`` → cached ``~/.castform`` session — against the host from
:mod:`benchmax.config`. ``handle_errors`` keeps tracebacks out of normal failures.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable

import httpx

from benchmax import profile_config as profiles
from benchmax.platform.client import TrainerClient
from benchmax.platform.exceptions import AuthenticationError, TrainerError


def trainer_client() -> TrainerClient:
    """A TrainerClient bound to the configured platform host + bearer seam."""
    return TrainerClient()


def _login_hint() -> str:
    try:
        selected = profiles.select_profile()
        domain = "<domain>"
        try:
            profile = profiles.get_profile(selected.name)
        except RuntimeError:
            profile = None
        if isinstance(profile, dict) and isinstance(profile.get("domain"), str):
            domain = profile["domain"]
        return f"castform login --profile {selected.name} --domain {domain}"
    except RuntimeError:
        return "castform login --profile <profile> --domain <domain>"


def handle_errors(func: Callable) -> Callable:
    """Turn client/credential/network failures into a clean stderr line + exit 1."""

    @functools.wraps(func)
    def wrapper(args):
        try:
            return func(args)
        except AuthenticationError:
            print(
                f"Not logged in (or session expired). Run `{_login_hint()}`.",
                file=sys.stderr,
            )
            return 1
        except TrainerError as exc:
            print(f"Error: {exc.message}", file=sys.stderr)
            return 1
        except RuntimeError as exc:  # platform_bearer with no resolvable credential
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"Network error: {exc}", file=sys.stderr)
            return 1

    return wrapper
