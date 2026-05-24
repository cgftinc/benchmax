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

import functools
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


def _read_env_token(env_var: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{env_var} is not set. The launcher must inject it into the "
            f"run process (or set it locally for a self-serve run)."
        )
    return value


def env_token(env_var: str) -> TokenProvider:
    """A per-call provider that reads a static key from ``env_var``.

    For **external-provider** credentials (Turbopuffer, Pinecone, …) — keys we
    neither mint nor rotate. Read at runtime (never baked); raises if unset.
    The launcher injects ``env_var`` into the trainer/worker process; for
    self-serve the user sets it locally. Convention: ``<PROVIDER>_API_KEY``.

    Returned as a ``functools.partial`` over a module-level function so it
    pickles by reference (tiny bundle, no closure) — the var name travels, not
    a secret.
    """
    return functools.partial(_read_env_token, env_var)


def as_token_provider(
    value: str | TokenProvider | None,
    default: TokenProvider,
) -> TokenProvider:
    """Normalize a credential argument into a per-call provider.

    - ``None``     → ``default`` (the runtime seam; nothing baked).
    - ``str``      → a fixed-value provider. Ergonomic, but the literal is
      captured in the closure, so it **bakes** if the env is pickled — the
      discouraged class-A "your own key" carve-out (see
      ``docs/design/env-credential-model.md`` §7.1). Prefer the env-var default.
    - ``callable`` → used as-is.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return lambda: value
    return value
