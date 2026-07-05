"""Unit tests for benchmax.config profile / URL resolution."""

from __future__ import annotations

import pytest

from benchmax import config
from benchmax import profile_config as profiles


@pytest.fixture(autouse=True)
def _clear_url_env(monkeypatch, tmp_path):
    """Hermetic defaults: no explicit URL/profile overrides leaking from the env."""
    for var in (
        "CASTFORM_PROFILE",
        "CASTFORM_PLATFORM_URL",
        "CASTFORM_LLM_URL",
        "CASTFORM_AUTH_URL",
        "CASTFORM_WEB_APP_URL",
        "ACT_AS_TOKEN_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CASTFORM_CONFIG_PATH", str(tmp_path / "castform.toml"))


def test_base_domain_defaults_to_prod():
    assert config.base_domain() == "castform.com"


def test_profile_from_env_var(monkeypatch):
    profiles.upsert_profile("staging", domain="castform.dev")
    monkeypatch.setenv("CASTFORM_PROFILE", "staging")
    assert config.base_domain() == "castform.dev"


def test_profile_domain_drives_all_service_urls(monkeypatch):
    """One profile domain routes every service URL together."""
    profiles.upsert_profile("staging", domain="castform.dev")
    monkeypatch.setenv("CASTFORM_PROFILE", "staging")
    assert config.platform_url() == "https://api.castform.dev"
    assert config.llm_url() == "https://llm.castform.dev/v1"
    assert config.auth_url() == "https://auth.castform.dev"


def test_per_service_override_wins(monkeypatch):
    """A per-service URL override beats the derived domain — e.g. point platform
    at a local server while auth keeps talking to the real host."""
    monkeypatch.setenv("CASTFORM_PLATFORM_URL", "http://localhost:3000")
    assert config.platform_url() == "http://localhost:3000"
    assert config.auth_url() == "https://auth.castform.com"


def test_profile_url_override_wins_before_derived(monkeypatch):
    profiles.upsert_profile(
        "local",
        domain="castform.dev",
        api_url="http://localhost:4200",
        llm_url="http://localhost:8000/v1",
    )
    monkeypatch.setenv("CASTFORM_PROFILE", "local")
    assert config.platform_url() == "http://localhost:4200"
    assert config.llm_url() == "http://localhost:8000/v1"
    assert config.auth_url() == "https://auth.castform.dev"


def test_training_job_requires_injected_service_urls(monkeypatch):
    monkeypatch.setenv("ACT_AS_TOKEN_PATH", "/tmp/act-as-token")
    with pytest.raises(RuntimeError, match="CASTFORM_PLATFORM_URL must be set"):
        config.platform_url()
    with pytest.raises(RuntimeError, match="CASTFORM_LLM_URL must be set"):
        config.llm_url()

    monkeypatch.setenv("CASTFORM_PLATFORM_URL", "https://api.castform.dev")
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.castform.dev/v1")
    assert config.platform_url() == "https://api.castform.dev"
    assert config.llm_url() == "https://llm.castform.dev/v1"
