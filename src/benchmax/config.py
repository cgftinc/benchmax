"""Centralized URL configuration for the Castform platform.

All URLs derive from a single base domain. Set ``CASTFORM_BASE_DOMAIN`` to
point at a different environment (e.g. ``staging.castform.com``); individual
URL components may be overridden via their own env vars when needed.

Usage::

    from benchmax import config
    client = httpx.Client(base_url=config.platform_url())
"""

import os

DEFAULT_BASE_DOMAIN = "castform.com"


def base_domain() -> str:
    """Resolve the platform base domain.

    Precedence: explicit ``CASTFORM_BASE_DOMAIN`` → the cached device-auth
    session's ``env`` (``staging`` → ``castform.dev``) → ``prod`` default
    (``castform.com``). The ``env`` claim travels with the credential, so a
    logged-in SDK routes to the same environment it authenticated against —
    URL and credential can't desync. A prod session carries no ``env`` marker
    (``None`` → prod), so only internal staging logins deviate from the default.
    """
    override = os.environ.get("CASTFORM_BASE_DOMAIN")
    if override:
        return override
    if _session_env() == "staging":
        return "castform.dev"
    return DEFAULT_BASE_DOMAIN


def _session_env() -> str | None:
    """The ``env`` from the cached device-auth session, if any.

    Lazy import: ``config`` is a leaf that ``benchmax.platform`` depends on, so a
    top-level import would cycle (platform/__init__ → client → config)."""
    try:
        from benchmax.platform.credentials import read_castform_session

        session = read_castform_session()
    except Exception:
        return None
    if not session:
        return None
    env = session.get("env")
    return env if isinstance(env, str) else None


def platform_url() -> str:
    """Control-plane API (run management, dataset upload, env bundles).

    Returns the API host without the ``/v1`` suffix — clients prepend
    versioned paths (e.g. ``/v1/storage/upload-url``) themselves. The
    user-facing web app lives at ``app.{domain}`` and is not the API.
    """
    return os.environ.get("CASTFORM_PLATFORM_URL") or f"https://api.{base_domain()}"


def web_app_url() -> str:
    """User-facing web app (run dashboard, etc.). Runs are viewable at
    ``{web_app_url}/train/{run_id}``."""
    return os.environ.get("CASTFORM_WEB_APP_URL") or f"https://app.{base_domain()}"


def llm_url() -> str:
    """OpenAI-compatible LLM endpoint hosted by the platform."""
    return os.environ.get("CASTFORM_LLM_URL") or f"https://llm.{base_domain()}/v1"


def auth_url() -> str:
    """Auth-service base URL (device-authorization + JWT mint endpoints).

    Used by ``castform login`` and the per-process session→JWT mint. Derives from
    the same base domain as everything else, so a session minted against ``staging``
    talks to ``auth.castform.dev`` and a ``prod`` session to ``auth.castform.com``.
    """
    return os.environ.get("CASTFORM_AUTH_URL") or f"https://auth.{base_domain()}"
