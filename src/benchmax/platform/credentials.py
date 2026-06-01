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
import warnings
from collections.abc import Callable
from pathlib import Path

# Matches the trainer's token_refresher / ray_auth default.
_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_API_KEY_ENV = "PLATFORM_API_KEY"

TokenProvider = Callable[[], str]


def resolve_judge_key(api_key: str, base_url: str | None = None) -> str | None:
    """Resolve the bearer for an LLM-judge / LLM-clustering call.

    Explicit ``api_key`` wins. Otherwise fall back to the Castform platform
    credential seam (``ACT_AS_TOKEN_PATH`` in training, ``PLATFORM_API_KEY`` in
    playground / self-serve) — the same surface the search clients resolve
    through ``platform_bearer``.

    Failure modes are deliberately *loud*:

    - No platform credential **and** ``OPENAI_API_KEY`` set → return ``None`` so
      the OpenAI SDK picks it up (legitimate direct-customer use).
    - No platform credential and no ``OPENAI_API_KEY`` → re-raise
      ``platform_bearer``'s ``RuntimeError``.
    - Platform credential resolved but ``base_url`` unset → raise. The platform
      token is only valid against our llm-proxy; handing it to api.openai.com
      would leak it off-platform.
    """
    if api_key:
        return api_key
    try:
        token = platform_bearer()
    except RuntimeError:
        if os.environ.get("OPENAI_API_KEY"):
            return None
        raise
    if not base_url:
        raise RuntimeError(
            "Refusing to send the Castform platform credential to the OpenAI "
            "SDK default endpoint: judge base_url is unset. Set base_url to the "
            "llm-proxy, or pass an explicit api_key for direct OpenAI use."
        )
    return token


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
    This is the self-serve / hand-written-env path: the user sets ``env_var``
    locally. NOTE: platform-orchestrated training does NOT reach this — the
    trainer runs the env in a Ray actor that can't read these external secrets
    at runtime, so the platform codegen bakes the key at build instead (passes
    an explicit ``token_provider``; see :func:`as_token_provider`). Until
    first-party runtime injection exists, baking is the platform path.
    Convention: ``<PROVIDER>_API_KEY``.

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

    - ``None``     → ``default`` (the runtime seam; nothing baked). The
      self-serve / hand-written-env path, where ``env_var`` is set at runtime.
    - ``str``      → a fixed-value provider. The literal is captured in the
      closure, so it **bakes** into the pickle if the env is serialized. Don't
      hardcode a first-party key this way — use the env-var default. (A
      third-party provider key is baked deliberately; see ``callable``.)
    - ``callable`` → used as-is. The platform codegen passes a callable over a
      build-time-resolved *third-party* key here on purpose: the
      platform-orchestrated trainer (a Ray actor) can't read that external
      secret from its runtime env, so the key is baked at build. Revisit if
      first-party runtime injection of external keys is added.
    """
    if value is None:
        return default
    if isinstance(value, str):
        warnings.warn(
            "A literal token string was passed where a per-call provider is "
            "expected. It is captured in a closure and will be baked into the "
            "bundle if the env is pickled — a secret written to storage at rest. "
            "For a first-party key prefer the default resolver (platform_bearer) "
            "or an env-var provider (env_token). For a third-party provider key "
            "the codegen baking path passes a callable, not a literal string — "
            "if you are seeing this from generated code, that is a bug.",
            stacklevel=2,
        )
        return lambda: value
    return value
