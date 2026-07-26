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
2. ``CASTFORM_AUTH_TOKEN`` — an explicitly forwarded Castform bearer.
3. ``PLATFORM_API_KEY`` — the legacy API-key override used by hosted workers.
4. ``~/.castform/credentials.json`` — the selected profile's device-auth session cached by
   ``castform login`` (the human self-serve path). Lowest precedence so an
   explicit key/token always wins. Re-read each call.

Raises if none is available — fail loudly rather than make an
unauthenticated call.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from castform import profile_config

# Matches the trainer's token_refresher / ray_auth default.
_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_AUTH_TOKEN_ENV = "CASTFORM_AUTH_TOKEN"
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


def platform_bearer(profile: str | None = None) -> str:
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

    for env_var in (_AUTH_TOKEN_ENV, _API_KEY_ENV):
        if env_token := os.environ.get(env_var):
            return env_token

    session_jwt = _session_jwt(profile)
    if session_jwt:
        return session_jwt

    raise RuntimeError(
        f"No Castform platform credential available: run `castform login`, set "
        f"{_AUTH_TOKEN_ENV}, or (in training) ensure {_TOKEN_PATH_ENV} points at the "
        f"token_refresher output ({_TOKEN_PATH_ENV}={token_path!r})."
    )


def _credentials_path() -> Path:
    override = os.environ.get(_CRED_PATH_ENV)
    return Path(override) if override else _DEFAULT_CRED_PATH


def _read_credentials_data() -> dict | None:
    """Read the protected JSON object without selecting a profile."""
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


def _empty_store() -> dict:
    return {"version": 2, "profiles": {}}


def _credentials_store() -> dict:
    """Read the store; any unrecognized format reads as empty (re-login rewrites it)."""
    data = _read_credentials_data()
    if data and data.get("version") == 2 and isinstance(data.get("profiles"), dict):
        return data
    return _empty_store()


def _session_record(profile: str | None = None) -> dict | None:
    selected = profile_config.selected_profile_name(profile)
    record = _credentials_store()["profiles"].get(selected)
    return record if isinstance(record, dict) else None


def read_castform_session(profile: str | None = None) -> dict | None:
    """Read the selected profile's cached device-auth session.

    Sessions are bound to the auth URL used during login. A changed profile
    cannot silently send an old session to another Castform deployment.
    """
    record = _session_record(profile)
    if not record:
        return None
    session = record.get("session")
    if not isinstance(session, dict):
        return None
    stored_auth_url = record.get("auth_url")
    if stored_auth_url:
        from castform import config

        expected = config.auth_url(profile).rstrip("/")
        if stored_auth_url.rstrip("/") != expected:
            warnings.warn(
                "Ignoring Castform session because its auth URL no longer matches "
                f"the selected profile ({stored_auth_url!r} != {expected!r}).",
                stacklevel=2,
            )
            return None
    return session


def _write_credentials_store(store: dict) -> None:
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(store, output)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            Path(temporary).unlink()
        except OSError:
            pass


def write_castform_session(
    session: dict,
    profile: str | None = None,
    *,
    auth_url: str | None = None,
) -> None:
    """Write one profile's device-auth session to the protected JSON store.

    Existing profiles are preserved. Writes use a same-directory temporary file
    and atomic replacement, with owner-only permissions throughout.
    """
    from castform import config

    selected = profile_config.selected_profile_name(profile)
    store = _credentials_store()
    store["profiles"][selected] = {
        "auth_url": (auth_url or config.auth_url(profile)).rstrip("/"),
        "session": session,
    }
    _write_credentials_store(store)


def clear_castform_session(
    profile: str | None = None, *, all_profiles: bool = False
) -> None:
    """Clear one profile's session, or every session when explicitly requested."""
    with _SESSION_JWT_LOCK:
        _SESSION_JWT_CACHE.update(
            {"token": None, "src": None, "exp": 0.0, "profile": None}
        )
    store = _credentials_store()
    if all_profiles:
        store["profiles"] = {}
    else:
        store["profiles"].pop(profile_config.selected_profile_name(profile), None)
    if store["profiles"]:
        _write_credentials_store(store)
        return
    try:
        _credentials_path().unlink()
    except OSError:
        pass


def session_auth_token(profile: str | None = None) -> str:
    """Return the selected profile's unexpired session token for forwarding."""
    session = read_castform_session(profile)
    if not session:
        name = profile_config.selected_profile_name(profile)
        raise RuntimeError(
            f"Profile {name!r} is not logged in. Run `castform login --profile {name}`."
        )
    access_token = session.get("access_token")
    expires_at = session.get("expires_at")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("The selected Castform session has no access token.")
    if expires_at is not None and (
        not isinstance(expires_at, (int, float)) or expires_at <= time.time()
    ):
        raise RuntimeError(
            "The selected Castform session has expired; run `castform login` again."
        )
    return access_token


# Per-process cache of the short-lived JWT minted from the cached session, so we
# don't re-mint on every request. Mirrors web-app/src/lib/auth/jwt-fetcher.ts.
# ``src`` is the session access_token the JWT was minted from, so a re-login (or
# any session swap) within a live process can't keep serving a JWT for the old
# identity. Guarded by ``_SESSION_JWT_LOCK`` (rollouts resolve concurrently).
_SESSION_JWT_LOCK = threading.Lock()
_SESSION_JWT_CACHE: dict[str, object] = {
    "token": None,
    "src": None,
    "exp": 0.0,
    "profile": None,
}

# When a minted token carries no parseable ``exp``, cache it for this long rather
# than re-minting on every request (auth-service mints ~5-minute JWTs).
_MINT_FALLBACK_TTL = 240.0


def _jwt_claims(token: str) -> dict:
    """Decode a JWT payload without verifying — for timing/identity display only.

    Returns the claims dict, or ``{}`` if the token isn't a parseable JWT.
    """
    import base64

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # pad base64url
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _jwt_exp(token: str) -> float:
    """Read ``exp`` from a JWT payload without verifying; 0.0 if absent/unparseable."""
    exp = _jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else 0.0


def _mint_session_jwt(access_token: str, profile: str | None = None) -> str | None:
    """Exchange the cached session for a short-lived auth-service JWT.

    Hits the jwt plugin's ``GET /api/auth/token`` with the session as Bearer —
    the same endpoint the web-app uses over a cookie. Returns ``None`` on any
    failure (network, non-200, or a non-JSON body → caller re-prompts login).
    """
    import httpx

    from castform import config

    try:
        resp = httpx.get(
            f"{config.auth_url(profile)}/api/auth/token",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        token = resp.json().get("token")  # ValueError on a non-JSON 200 body
    except (httpx.HTTPError, ValueError):
        return None
    return token if isinstance(token, str) and token else None


def _session_jwt(profile: str | None = None) -> str | None:
    """Resolve a short-lived JWT from the cached device-auth session.

    Reads ``~/.castform``; if the session is present and unexpired, returns a
    cached JWT (re-minting ~60s before it expires — the per-process mint from
    D2). On a transient mint failure, a still-valid cached JWT is reused rather
    than failing the in-flight request. Returns ``None`` if there's no usable
    session (caller falls through to the ``castform login`` prompt). Refresh from
    ``refresh_token`` when the *session* itself expires is a future addition;
    today an expired session just prompts re-login.
    """
    selected = profile_config.selected_profile_name(profile)
    session = read_castform_session(profile)
    if not session:
        return None
    access_token = session.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        return None
    # Fail closed on a malformed expiry (non-numeric) as well as a past one, so a
    # hand-edited/legacy file can't slip an expired session past the guard.
    expires_at = session.get("expires_at")
    if expires_at is not None and (
        not isinstance(expires_at, (int, float)) or expires_at <= time.time()
    ):
        return None

    with _SESSION_JWT_LOCK:
        cached = _SESSION_JWT_CACHE["token"]
        if (
            cached
            and _SESSION_JWT_CACHE["src"] == access_token
            and _SESSION_JWT_CACHE["profile"] == selected
            and time.time() < float(_SESSION_JWT_CACHE["exp"]) - 60
        ):
            return cached  # type: ignore[return-value]

    jwt = _mint_session_jwt(access_token, profile)
    if jwt:
        exp = _jwt_exp(jwt)
        if exp <= time.time():
            exp = (
                time.time() + _MINT_FALLBACK_TTL
            )  # no parseable exp → don't re-mint per call
        with _SESSION_JWT_LOCK:
            _SESSION_JWT_CACHE.update(
                {"token": jwt, "src": access_token, "exp": exp, "profile": selected}
            )
        return jwt

    # Mint failed (transient): reuse a cached JWT for this same session that is
    # still actually valid, rather than failing a request that had a usable token.
    with _SESSION_JWT_LOCK:
        cached = _SESSION_JWT_CACHE["token"]
        if (
            cached
            and _SESSION_JWT_CACHE["src"] == access_token
            and _SESSION_JWT_CACHE["profile"] == selected
            and time.time() < float(_SESSION_JWT_CACHE["exp"])
        ):
            return cached  # type: ignore[return-value]
    return None


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
