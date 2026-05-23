"""Unit tests for the per-call platform credential resolver."""

from __future__ import annotations

import pytest

from benchmax.platform.credentials import platform_bearer

_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_API_KEY_ENV = "PLATFORM_API_KEY"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(_TOKEN_PATH_ENV, raising=False)
    monkeypatch.delenv(_API_KEY_ENV, raising=False)


def test_reads_token_file_and_strips(tmp_path, monkeypatch):
    f = tmp_path / "act-as-token"
    f.write_text("  jwt-from-file\n")
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(f))
    assert platform_bearer() == "jwt-from-file"


def test_token_file_takes_precedence_over_env(tmp_path, monkeypatch):
    f = tmp_path / "act-as-token"
    f.write_text("jwt-from-file")
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(f))
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    assert platform_bearer() == "jwt-from-file"


def test_falls_back_to_env_when_path_set_but_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    assert platform_bearer() == "sk_env"


def test_falls_back_to_env_when_file_empty(tmp_path, monkeypatch):
    f = tmp_path / "act-as-token"
    f.write_text("   \n")
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(f))
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    assert platform_bearer() == "sk_env"


def test_uses_env_when_no_token_path(monkeypatch):
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    assert platform_bearer() == "sk_env"


def test_raises_when_no_credential(monkeypatch):
    with pytest.raises(RuntimeError, match="No Castform platform credential"):
        platform_bearer()


def test_rotation_is_picked_up_per_call(tmp_path, monkeypatch):
    """A new call re-reads the file, so token_refresher rotation is seen."""
    f = tmp_path / "act-as-token"
    f.write_text("token-1")
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(f))
    assert platform_bearer() == "token-1"
    f.write_text("token-2")
    assert platform_bearer() == "token-2"
