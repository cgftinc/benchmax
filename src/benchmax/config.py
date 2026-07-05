"""Centralized URL configuration for the Castform platform.

URLs resolve through the selected Castform profile:

``--profile`` / explicit param > ``CASTFORM_PROFILE`` > ``default_profile`` in
``~/.castform.toml`` > ``prod``.

Each profile has a domain and optional URL overrides. Individual URL env vars
still win for one-off local development.

Usage::

    from benchmax import config
    client = httpx.Client(base_url=config.platform_url())
"""

import os

from benchmax import profile_config as profiles

DEFAULT_BASE_DOMAIN = "castform.com"
_JOB_TOKEN_ENV = "ACT_AS_TOKEN_PATH"


def _profile(profile: str | None = None) -> dict:
    selected = profiles.get_profile(profile)
    if selected is None:
        name = profile or profiles.selected_profile_name()
        raise RuntimeError(
            f"Castform profile {name!r} is not configured. Run "
            f"`castform login --profile {name} --domain <domain>` or edit "
            f"{profiles.config_path()}."
        )
    return selected


def base_domain(profile: str | None = None) -> str:
    """Resolve the selected profile's base domain."""
    selected = _profile(profile)
    domain = selected.get("domain")
    return domain if isinstance(domain, str) and domain else DEFAULT_BASE_DOMAIN


def platform_url(profile: str | None = None) -> str:
    """Control-plane API (run management, dataset upload, env bundles).

    Returns the API host without the ``/v1`` suffix — clients prepend
    versioned paths (e.g. ``/v1/storage/upload-url``) themselves. The
    user-facing web app lives at ``app.{domain}`` and is not the API.
    """
    env_url = os.environ.get("CASTFORM_PLATFORM_URL")
    if env_url:
        return env_url
    if os.environ.get(_JOB_TOKEN_ENV):
        raise RuntimeError(
            "CASTFORM_PLATFORM_URL must be set when ACT_AS_TOKEN_PATH is set. "
            "Refusing to derive a platform URL inside a training job."
        )
    selected = _profile(profile)
    return selected.get("api_url") or f"https://api.{base_domain(profile)}"


def web_app_url(profile: str | None = None) -> str:
    """User-facing web app (run dashboard, etc.). Runs are viewable at
    ``{web_app_url}/train/{run_id}``."""
    selected = _profile(profile)
    return (
        os.environ.get("CASTFORM_WEB_APP_URL")
        or selected.get("app_url")
        or f"https://app.{base_domain(profile)}"
    )


def llm_url(profile: str | None = None) -> str:
    """OpenAI-compatible LLM endpoint hosted by the platform."""
    env_url = os.environ.get("CASTFORM_LLM_URL")
    if env_url:
        return env_url
    if os.environ.get(_JOB_TOKEN_ENV):
        raise RuntimeError(
            "CASTFORM_LLM_URL must be set when ACT_AS_TOKEN_PATH is set. "
            "Refusing to derive an LLM URL inside a training job."
        )
    selected = _profile(profile)
    return selected.get("llm_url") or f"https://llm.{base_domain(profile)}/v1"


def auth_url(profile: str | None = None) -> str:
    """Auth-service base URL (device-authorization + JWT mint endpoints).

    Used by ``castform login`` and the per-process session→JWT mint.
    """
    selected = _profile(profile)
    return (
        os.environ.get("CASTFORM_AUTH_URL")
        or selected.get("auth_url")
        or f"https://auth.{base_domain(profile)}"
    )
