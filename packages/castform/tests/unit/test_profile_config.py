from __future__ import annotations

import pytest

from castform import profile_config


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("CASTFORM_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.delenv("CASTFORM_PROFILE", raising=False)


def test_missing_config_defaults_to_prod() -> None:
    assert profile_config.selected_profile_name() == "prod"
    assert profile_config.get_profile() == {"domain": "castform.com"}


def test_activate_and_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    profile_config.upsert_profile("staging", domain="castform.dev")
    profile_config.activate_profile("staging")
    assert profile_config.selected_profile_name() == "staging"

    monkeypatch.setenv("CASTFORM_PROFILE", "prod")
    assert profile_config.selected_profile_name() == "prod"


def test_activate_rejects_unknown_profile() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        profile_config.activate_profile("missing")
