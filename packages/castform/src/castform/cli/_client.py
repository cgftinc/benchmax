"""Shared platform-client wiring + error handling for CLI command groups.

Read/control commands resolve their bearer through the credential seam
(:func:`castform.platform.credentials.platform_bearer`) — ``ACT_AS_TOKEN_PATH``
→ ``CASTFORM_API_KEY`` → cached ``~/.castform`` session — against the host from
:mod:`castform.config`. ``handle_errors`` keeps tracebacks out of normal failures.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable

import httpx

from castform.platform.client import TrainerClient
from castform.platform.exceptions import AuthenticationError, TrainerError


def trainer_client() -> TrainerClient:
    """A TrainerClient bound to the configured platform host + bearer seam."""
    return TrainerClient()


def handle_errors(func: Callable) -> Callable:
    """Turn client/credential/network failures into a clean stderr line + exit 1."""

    @functools.wraps(func)
    def wrapper(args):
        try:
            return func(args)
        except AuthenticationError:
            print(
                "Not logged in (or session expired). Run `castform login`.",
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
