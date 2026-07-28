"""Unit tests for castform.config base-domain / URL resolution."""

from __future__ import annotations

import pytest
from castform import config, profile_config


@pytest.fixture(autouse=True)
def _clear_url_env(monkeypatch, tmp_path):
    """Hermetic defaults: no explicit URL/domain overrides leaking from the env."""
    for var in (
        "CASTFORM_BASE_DOMAIN",
        "CASTFORM_PLATFORM_URL",
        "CASTFORM_LLM_URL",
        "CASTFORM_AUTH_URL",
        "CASTFORM_WEB_APP_URL",
        "CASTFORM_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CASTFORM_CONFIG_PATH", str(tmp_path / "config.toml"))


def test_base_domain_defaults_to_prod():
    assert config.base_domain() == "castform.com"


def test_base_domain_from_env_var(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.dev")
    assert config.base_domain() == "castform.dev"


def test_base_domain_drives_all_service_urls(monkeypatch):
    """One domain knob routes every service URL together."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.dev")
    assert config.platform_url() == "https://api.castform.dev"
    assert config.llm_url() == "https://llm.castform.dev/v1"
    assert config.auth_url() == "https://auth.castform.dev"


def test_per_service_override_wins(monkeypatch):
    """A per-service URL override beats the derived domain — e.g. point platform
    at a local server while auth keeps talking to the real host."""
    monkeypatch.setenv("CASTFORM_PLATFORM_URL", "http://localhost:3000")
    assert config.platform_url() == "http://localhost:3000"
    assert config.auth_url() == "https://auth.castform.com"


def test_named_profile_drives_urls(monkeypatch):
    profile_config.upsert_profile("staging", domain="castform.dev")
    monkeypatch.setenv("CASTFORM_PROFILE", "staging")
    assert config.platform_url() == "https://api.castform.dev"
    assert config.auth_url() == "https://auth.castform.dev"


def test_self_hosted_profile_uses_explicit_urls(monkeypatch):
    profile_config.upsert_profile(
        "acme",
        platform_url="https://control.acme.internal/",
        auth_url="https://login.acme.internal/",
        llm_url="https://models.acme.internal/v1/",
        app_url="https://castform.acme.internal/",
    )
    monkeypatch.setenv("CASTFORM_PROFILE", "acme")
    assert config.platform_url() == "https://control.acme.internal"
    assert config.auth_url() == "https://login.acme.internal"
    assert config.llm_url() == "https://models.acme.internal/v1"
    assert config.web_app_url() == "https://castform.acme.internal"


def test_self_hosted_profile_never_falls_back_to_prod(monkeypatch):
    profile_config.upsert_profile("acme", auth_url="https://login.acme.internal")
    monkeypatch.setenv("CASTFORM_PROFILE", "acme")
    with pytest.raises(RuntimeError, match="no domain"):
        config.platform_url()
