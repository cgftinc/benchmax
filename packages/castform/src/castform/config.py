"""Profile-aware URL configuration for the Castform platform.

Selection precedence is an explicit profile, ``CASTFORM_PROFILE``, then the
active profile. The built-in profile is ``prod`` at ``castform.com``. Existing
URL environment variables remain the highest precedence for development and CI.
"""

import os

from castform import profile_config

DEFAULT_BASE_DOMAIN = profile_config.DEFAULT_DOMAIN


def _profile(profile: str | None = None) -> dict[str, str]:
    selected = profile_config.get_profile(profile)
    if selected is None:
        name = profile_config.selected_profile_name(profile)
        raise RuntimeError(
            f"Castform profile {name!r} is not configured. Run "
            f"`castform login --profile {name} --domain <domain>`."
        )
    return selected


def base_domain(profile: str | None = None) -> str:
    """Resolve a profile's domain, with the legacy environment override."""
    if override := os.environ.get("CASTFORM_BASE_DOMAIN"):
        return override
    if domain := _profile(profile).get("domain"):
        return domain
    name = profile_config.selected_profile_name(profile)
    raise RuntimeError(
        f"Castform profile {name!r} has no domain; configure the required service URL explicitly."
    )


def profile_target(profile: str | None = None) -> str:
    """Human-readable target for CLI status messages."""
    selected = _profile(profile)
    return (
        selected.get("domain")
        or selected.get("platform_url")
        or selected.get("auth_url")
        or "<incomplete>"
    )


def platform_url(profile: str | None = None) -> str:
    """Control-plane API base URL without the ``/v1`` suffix."""
    return (
        os.environ.get("CASTFORM_PLATFORM_URL")
        or _profile(profile).get("platform_url")
        or f"https://api.{base_domain(profile)}"
    )


def web_app_url(profile: str | None = None) -> str:
    """User-facing web application URL."""
    return (
        os.environ.get("CASTFORM_WEB_APP_URL")
        or _profile(profile).get("app_url")
        or f"https://app.{base_domain(profile)}"
    )


def llm_url(profile: str | None = None) -> str:
    """OpenAI-compatible Castform LLM endpoint."""
    return (
        os.environ.get("CASTFORM_LLM_URL")
        or _profile(profile).get("llm_url")
        or f"https://llm.{base_domain(profile)}/v1"
    )


def auth_url(profile: str | None = None) -> str:
    """Auth-service base URL used for login and session JWT minting."""
    return (
        os.environ.get("CASTFORM_AUTH_URL")
        or _profile(profile).get("auth_url")
        or f"https://auth.{base_domain(profile)}"
    )
