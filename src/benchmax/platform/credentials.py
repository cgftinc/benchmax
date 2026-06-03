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
3. ``~/.castform/credentials.json`` — the device-auth session cached by
   ``castform login`` (the human self-serve path). Lowest precedence so an
   explicit key/token always wins. Re-read each call.

Raises if none is available — fail loudly rather than make an
unauthenticated call.
"""

from __future__ import annotations

import functools
import json
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path

# Matches the trainer's token_refresher / ray_auth default.
_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_API_KEY_ENV = "PLATFORM_API_KEY"

# Cached device-auth session written by `castform login` (Phase 4). Lowest
# precedence in platform_bearer. Path overridable for tests.
_CRED_PATH_ENV = "CASTFORM_CREDENTIALS_PATH"
_DEFAULT_CRED_PATH = Path.home() / ".castform" / "credentials.json"

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

    session_jwt = _session_jwt()
    if session_jwt:
        return session_jwt

    raise RuntimeError(
        f"No Castform platform credential available: run `castform login`, set "
        f"{_API_KEY_ENV}, or (in training) ensure {_TOKEN_PATH_ENV} points at the "
        f"token_refresher output ({_TOKEN_PATH_ENV}={token_path!r})."
    )


def _credentials_path() -> Path:
    override = os.environ.get(_CRED_PATH_ENV)
    return Path(override) if override else _DEFAULT_CRED_PATH


def read_castform_session() -> dict | None:
    """Read the cached device-auth session, or ``None`` if unusable.

    Returns the parsed ``~/.castform/credentials.json`` dict, or ``None`` when the
    file is absent, malformed, or (on POSIX) looser than ``0600`` — a
    world/group-readable credential is not trusted. Written by ``castform
    login`` (Phase 4); schema::

        {"access_token": str, "refresh_token": str, "expires_at": int,
         "env": "staging"}   # env omitted for prod (the default)

    ``refresh_token`` is reserved for the per-process mint/refresh added with the
    device flow (Phase 3/4); today only ``access_token`` / ``env`` are consumed.
    """
    path = _credentials_path()
    try:
        st = path.stat()
    except OSError:
        return None
    if os.name == "posix" and (st.st_mode & 0o077):
        warnings.warn(
            f"Ignoring {path}: permissions {oct(st.st_mode & 0o777)} are looser "
            f"than 0600. Run `chmod 600 {path}`.",
            stacklevel=2,
        )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_castform_session(session: dict) -> None:
    """Write the device-auth session to ``~/.castform/credentials.json`` (0600).

    Called by ``castform login``. Creates the parent dir and writes the file with
    owner-only permissions; an existing file is replaced atomically-enough for a
    single-user CLI (truncate + write under a 0600 umask).
    """
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with 0600 from the start (don't briefly expose a world-readable file).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(session, fh)
    os.chmod(path, 0o600)  # tighten even if the file pre-existed with looser bits


def clear_castform_session() -> None:
    """Delete the cached session (``castform logout``). No-op if absent."""
    _SESSION_JWT_CACHE["token"] = None
    _SESSION_JWT_CACHE["exp"] = 0.0
    try:
        _credentials_path().unlink()
    except OSError:
        pass


# Per-process cache of the short-lived JWT minted from the cached session, so we
# don't re-mint on every request. Mirrors web-app/src/lib/auth/jwt-fetcher.ts.
_SESSION_JWT_CACHE: dict[str, object] = {"token": None, "exp": 0.0}


def _jwt_exp(token: str) -> float:
    """Read ``exp`` from a JWT payload without verifying (we only need timing)."""
    import base64

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # pad base64url
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return 0.0


def _mint_session_jwt(access_token: str) -> str | None:
    """Exchange the cached session for a short-lived auth-service JWT.

    Hits the jwt plugin's ``GET /api/auth/token`` with the session as Bearer —
    the same endpoint the web-app uses over a cookie. Returns ``None`` on any
    failure (network, or the session no longer valid → caller re-prompts login).
    """
    import httpx

    from benchmax import config

    try:
        resp = httpx.get(
            f"{config.auth_url()}/api/auth/token",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    token = resp.json().get("token")
    return token if isinstance(token, str) and token else None


def _session_jwt() -> str | None:
    """Resolve a short-lived JWT from the cached device-auth session.

    Reads ``~/.castform``; if the session is present and unexpired, returns a
    cached JWT (re-minting ~60s before it expires — the per-process mint from
    D2). Returns ``None`` if there's no usable session (caller falls through to
    the ``castform login`` prompt). Refresh from ``refresh_token`` when the
    *session* itself expires is a future addition; today an expired session just
    prompts re-login.
    """
    session = read_castform_session()
    if not session:
        return None
    access_token = session.get("access_token")
    if not access_token:
        return None
    expires_at = session.get("expires_at")
    if isinstance(expires_at, (int, float)) and expires_at <= time.time():
        return None

    cached = _SESSION_JWT_CACHE["token"]
    if cached and time.time() < float(_SESSION_JWT_CACHE["exp"]) - 60:
        return cached  # type: ignore[return-value]

    jwt = _mint_session_jwt(access_token)
    if jwt:
        _SESSION_JWT_CACHE["token"] = jwt
        _SESSION_JWT_CACHE["exp"] = _jwt_exp(jwt)
    return jwt


def resolve_token_provider(
    api_key: str | None,
    token_provider: TokenProvider | None = None,
) -> TokenProvider:
    """Pick the per-call bearer provider for a control-plane client.

    Precedence:

    1. a **non-empty** ``api_key`` — a fixed-value provider (the caller's
       override). Empty string / ``None`` count as "not provided" (config layers
       commonly default an unset key to ``""``), so they fall through to:
    2. explicit ``token_provider`` — a custom per-call provider (tests / BYO).
    3. :func:`platform_bearer` — the credential seam (``ACT_AS_TOKEN_PATH`` →
       ``PLATFORM_API_KEY`` → cached ``~/.castform`` session).

    The result is called **per request** by the client, so a rotating/expiring
    token is picked up — never frozen at construction.
    """
    if api_key:
        return lambda: api_key
    if token_provider is not None:
        return token_provider
    return platform_bearer


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
