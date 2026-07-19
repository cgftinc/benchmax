"""Unit tests for castform.config base-domain / URL resolution."""

from __future__ import annotations

import pytest

from castform import config


@pytest.fixture(autouse=True)
def _clear_url_env(monkeypatch):
    """Hermetic defaults: no explicit URL/domain overrides leaking from the env."""
    for var in (
        "CASTFORM_BASE_DOMAIN",
        "CASTFORM_PLATFORM_URL",
        "CASTFORM_LLM_URL",
        "CASTFORM_AUTH_URL",
    ):
        monkeypatch.delenv(var, raising=False)


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
