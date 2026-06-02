"""Unit tests for the per-call platform credential resolver."""

from __future__ import annotations

import json
import pickle

import cloudpickle
import pytest

from benchmax.platform.credentials import (
    as_token_provider,
    env_token,
    platform_bearer,
    read_castform_session,
    resolve_token_provider,
)

_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_API_KEY_ENV = "PLATFORM_API_KEY"
_CRED_PATH_ENV = "CASTFORM_CREDENTIALS_PATH"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch, tmp_path):
    monkeypatch.delenv(_TOKEN_PATH_ENV, raising=False)
    monkeypatch.delenv(_API_KEY_ENV, raising=False)
    # Point the session cache at a non-existent file so a real ~/.castform on
    # the dev machine can't leak into tests; cache tests override it.
    monkeypatch.setenv(_CRED_PATH_ENV, str(tmp_path / "no-creds.json"))


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


# ---- ~/.castform session cache (lowest precedence) ----


def _write_session(tmp_path, monkeypatch, data, mode=0o600):
    f = tmp_path / "credentials.json"
    f.write_text(json.dumps(data))
    f.chmod(mode)
    monkeypatch.setenv(_CRED_PATH_ENV, str(f))
    return f


def test_platform_bearer_uses_cached_session_token(tmp_path, monkeypatch):
    _write_session(tmp_path, monkeypatch, {"access_token": "sk_session"})
    assert platform_bearer() == "sk_session"


def test_env_key_takes_precedence_over_session(tmp_path, monkeypatch):
    _write_session(tmp_path, monkeypatch, {"access_token": "sk_session"})
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    assert platform_bearer() == "sk_env"


def test_token_path_takes_precedence_over_session(tmp_path, monkeypatch):
    f = tmp_path / "act-as-token"
    f.write_text("jwt-from-file")
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(f))
    _write_session(tmp_path, monkeypatch, {"access_token": "sk_session"})
    assert platform_bearer() == "jwt-from-file"


def test_session_used_when_not_expired(tmp_path, monkeypatch):
    _write_session(
        tmp_path, monkeypatch, {"access_token": "sk_session", "expires_at": 9999999999}
    )
    assert platform_bearer() == "sk_session"


def test_session_skipped_when_expired(tmp_path, monkeypatch):
    _write_session(
        tmp_path, monkeypatch, {"access_token": "sk_session", "expires_at": 1}
    )
    with pytest.raises(RuntimeError, match="No Castform platform credential"):
        platform_bearer()


def test_session_ignored_when_world_readable(tmp_path, monkeypatch):
    _write_session(tmp_path, monkeypatch, {"access_token": "sk_session"}, mode=0o644)
    with pytest.warns(UserWarning, match="looser than 0600"):
        assert read_castform_session() is None


def test_session_missing_file_is_noop():
    # autouse fixture points _CRED_PATH_ENV at a non-existent file
    assert read_castform_session() is None
    with pytest.raises(RuntimeError, match="No Castform platform credential"):
        platform_bearer()


def test_read_castform_session_malformed_returns_none(tmp_path, monkeypatch):
    f = tmp_path / "credentials.json"
    f.write_text("{not valid json")
    f.chmod(0o600)
    monkeypatch.setenv(_CRED_PATH_ENV, str(f))
    assert read_castform_session() is None


# ---- resolve_token_provider (client bearer precedence) ----


def test_resolve_explicit_api_key_wins(monkeypatch):
    """An explicit api_key beats both a token_provider and the env seam."""
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    provider = resolve_token_provider("sk_explicit", token_provider=lambda: "sk_tp")
    assert provider() == "sk_explicit"


def test_resolve_token_provider_used_when_no_api_key():
    provider = resolve_token_provider(None, token_provider=lambda: "sk_tp")
    assert provider() == "sk_tp"


def test_resolve_falls_back_to_platform_bearer(monkeypatch):
    """No api_key and no token_provider → the platform_bearer seam."""
    monkeypatch.setenv(_API_KEY_ENV, "sk_env")
    provider = resolve_token_provider(None)
    assert provider is platform_bearer
    assert provider() == "sk_env"


def test_resolve_seam_is_per_call(monkeypatch):
    """The fallback provider re-reads the env each call (rotation is seen)."""
    provider = resolve_token_provider(None)
    monkeypatch.setenv(_API_KEY_ENV, "sk_1")
    assert provider() == "sk_1"
    monkeypatch.setenv(_API_KEY_ENV, "sk_2")
    assert provider() == "sk_2"


# ---- env_token (external-provider keys) ----


def test_env_token_reads_var(monkeypatch):
    monkeypatch.setenv("TPUF_API_KEY", "tpuf_abc")
    assert env_token("TPUF_API_KEY")() == "tpuf_abc"


def test_env_token_raises_when_unset(monkeypatch):
    monkeypatch.delenv("TPUF_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TPUF_API_KEY is not set"):
        env_token("TPUF_API_KEY")()


def test_env_token_reads_per_call(monkeypatch):
    provider = env_token("TPUF_API_KEY")
    monkeypatch.setenv("TPUF_API_KEY", "k1")
    assert provider() == "k1"
    monkeypatch.setenv("TPUF_API_KEY", "k2")
    assert provider() == "k2"


def test_env_token_pickles_by_reference_small(monkeypatch):
    """partial-over-module-fn → tiny pickle, var name travels (no secret)."""
    provider = env_token("TPUF_API_KEY")
    data = cloudpickle.dumps(provider)
    assert len(data) < 500
    assert b"tpuf_" not in data  # nothing read at pickle time
    monkeypatch.setenv("TPUF_API_KEY", "tpuf_xyz")
    assert pickle.loads(data)() == "tpuf_xyz"


# ---- as_token_provider (normalization + string sugar) ----


def test_as_token_provider_none_returns_default():
    default = env_token("X")
    assert as_token_provider(None, default) is default


def test_as_token_provider_callable_passthrough():
    fn = env_token("X")
    assert as_token_provider(fn, platform_bearer) is fn


def test_as_token_provider_string_sugar():
    with pytest.warns(UserWarning, match="baked into the bundle"):
        provider = as_token_provider("sk_literal", platform_bearer)
    assert provider() == "sk_literal"


def test_as_token_provider_string_sugar_bakes_secret_when_pickled():
    """The warning's premise: a literal token is captured in the closure and
    travels with the pickled provider — the at-rest leak the default avoids."""
    with pytest.warns(UserWarning):
        provider = as_token_provider("sk_literal", platform_bearer)
    revived = cloudpickle.loads(cloudpickle.dumps(provider))
    assert revived() == "sk_literal"  # secret survived pickling — that's the risk
