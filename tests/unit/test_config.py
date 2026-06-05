"""Unit tests for benchmax.config base-domain / URL resolution."""

from __future__ import annotations

import json

import pytest

from benchmax import config

_CRED_PATH_ENV = "CASTFORM_CREDENTIALS_PATH"


@pytest.fixture(autouse=True)
def _clear_url_env(monkeypatch, tmp_path):
    """Hermetic defaults: no explicit URL/domain overrides, and the session
    cache points at a non-existent file so a real ~/.castform can't leak in."""
    for var in ("CASTFORM_BASE_DOMAIN", "CASTFORM_PLATFORM_URL", "CASTFORM_LLM_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(_CRED_PATH_ENV, str(tmp_path / "no-creds.json"))


def _write_session(tmp_path, monkeypatch, data):
    f = tmp_path / "credentials.json"
    f.write_text(json.dumps(data))
    f.chmod(0o600)
    monkeypatch.setenv(_CRED_PATH_ENV, str(f))


def test_base_domain_defaults_to_prod():
    assert config.base_domain() == "castform.com"


def test_base_domain_from_staging_session(tmp_path, monkeypatch):
    _write_session(tmp_path, monkeypatch, {"access_token": "x", "env": "staging"})
    assert config.base_domain() == "castform.dev"


def test_prod_session_has_no_env_marker(tmp_path, monkeypatch):
    """A prod session carries no env marker → prod default (no redundant field)."""
    _write_session(tmp_path, monkeypatch, {"access_token": "x"})
    assert config.base_domain() == "castform.com"


def test_explicit_base_domain_overrides_session(tmp_path, monkeypatch):
    _write_session(tmp_path, monkeypatch, {"access_token": "x", "env": "staging"})
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    assert config.base_domain() == "castform.com"


def test_staging_session_drives_both_platform_and_llm_urls(tmp_path, monkeypatch):
    """The env claim travels with the credential, so api + llm both route to the
    same env — structurally preventing the .dev->.com misroute class."""
    _write_session(tmp_path, monkeypatch, {"access_token": "x", "env": "staging"})
    assert config.platform_url() == "https://api.castform.dev"
    assert config.llm_url() == "https://llm.castform.dev/v1"
