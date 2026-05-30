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
    return os.environ.get("CASTFORM_BASE_DOMAIN", DEFAULT_BASE_DOMAIN)


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
