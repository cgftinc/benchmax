"""Castform platform credential resolution for RL environments.

The generated env authenticates corpus search (platform-service BFF) and the
LLM judge (llm-proxy) with a single bearer token. Where that token comes from
depends on *who* runs the env, so it is resolved **per call** — never captured
at construction. A captured token expires mid-run and can't be rotated, which
is exactly what broke training when the credential was baked into the bundle.

Precedence (per call):

1. ``ACT_AS_TOKEN_PATH`` — a rotating act-as JWT the trainer's ``token_refresher``
   writes and re-writes before expiry (multi-audience: platform-service +
   llm-proxy-service). Used during training. Re-read each call so rotation is
   picked up. Mirrors ``trainer/auth/ray_auth.py``'s per-request read.
2. ``PLATFORM_API_KEY`` — injected by the rollout worker (playground / eval) or
   set by the user (self-serve / benchmax lib).

Raises if neither is available — fail loudly rather than make an
unauthenticated call.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

# Matches the trainer's token_refresher / ray_auth default.
_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_API_KEY_ENV = "PLATFORM_API_KEY"

TokenProvider = Callable[[], str]


def platform_bearer() -> str:
    """Resolve the Castform platform bearer token. Call once per request.

    See module docstring for precedence. Raises ``RuntimeError`` if no
    credential is available.
    """
    token_path = os.environ.get(_TOKEN_PATH_ENV)
    if token_path:
        try:
            token = Path(token_path).read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if token:
            return token

    env_token = os.environ.get(_API_KEY_ENV)
    if env_token:
        return env_token

    raise RuntimeError(
        f"No Castform platform credential available: set {_API_KEY_ENV}, or "
        f"(in training) ensure {_TOKEN_PATH_ENV} points at the token_refresher "
        f"output ({_TOKEN_PATH_ENV}={token_path!r})."
    )
