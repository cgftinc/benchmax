"""Centralized URL configuration for the Castform platform.

All URLs derive from a single base domain, resolved from exactly two places: the
``CASTFORM_BASE_DOMAIN`` env var, or the built-in ``castform.com`` default.
Individual URLs may be overridden via their own env vars
(``CASTFORM_PLATFORM_URL`` / ``CASTFORM_LLM_URL`` / ``CASTFORM_AUTH_URL`` /
``CASTFORM_WEB_APP_URL``) — e.g. point platform at ``http://localhost:3000`` for
local dev while auth keeps talking to the real host.

Usage::

    from castform import config
    client = httpx.Client(base_url=config.platform_url())
"""

import os

DEFAULT_BASE_DOMAIN = "castform.com"


def base_domain() -> str:
    """Resolve the platform base domain: ``CASTFORM_BASE_DOMAIN`` or the
    ``castform.com`` default. To target another environment (e.g. internal
    staging), export ``CASTFORM_BASE_DOMAIN=castform.dev``."""
    return os.environ.get("CASTFORM_BASE_DOMAIN") or DEFAULT_BASE_DOMAIN


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
    the same base domain as everything else, or ``CASTFORM_AUTH_URL`` to override.
    """
    return os.environ.get("CASTFORM_AUTH_URL") or f"https://auth.{base_domain()}"
