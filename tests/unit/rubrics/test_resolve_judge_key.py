"""Unit tests for the judge-key resolver's fail-loud / no-leak semantics.

``_resolve_judge_key`` is the judge half of the platform credential seam (the
search half lives in ``benchmax.platform.credentials``). Its failure modes are
deliberately loud — a missing/misrouted credential must surface as a raise, not
degrade into an opaque downstream 401 or leak the multi-audience platform token
off-platform.
"""

from __future__ import annotations

import pytest

from benchmax.rubrics.rubric import _resolve_judge_key

_TOKEN_PATH_ENV = "ACT_AS_TOKEN_PATH"
_API_KEY_ENV = "PLATFORM_API_KEY"
_OPENAI_ENV = "OPENAI_API_KEY"

_LLM_PROXY = "https://llm.castform.com/v1"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in (_TOKEN_PATH_ENV, _API_KEY_ENV, _OPENAI_ENV):
        monkeypatch.delenv(var, raising=False)


def test_explicit_api_key_wins_and_skips_base_url_guard(monkeypatch):
    # Customer's own key: returned verbatim, no platform resolution, no guard.
    monkeypatch.setenv(_API_KEY_ENV, "platform_tok")
    assert _resolve_judge_key("sk-customer", base_url=None) == "sk-customer"


def test_platform_credential_resolved_for_llm_proxy(monkeypatch):
    monkeypatch.setenv(_API_KEY_ENV, "platform_tok")
    assert _resolve_judge_key("", base_url=_LLM_PROXY) == "platform_tok"


def test_no_credential_with_openai_key_falls_back_to_none(monkeypatch):
    # Direct-customer use: no platform creds but OPENAI_API_KEY set → None so the
    # OpenAI SDK picks it up against api.openai.com.
    monkeypatch.setenv(_OPENAI_ENV, "sk-openai")
    assert _resolve_judge_key("", base_url=None) is None


def test_no_credential_and_no_openai_key_raises_loudly(monkeypatch):
    # Training/eval misconfig: must raise, not return None → opaque 401 later.
    with pytest.raises(RuntimeError, match="No Castform platform credential"):
        _resolve_judge_key("", base_url=_LLM_PROXY)


def test_platform_token_with_missing_base_url_refuses_to_leak(monkeypatch):
    # A resolved multi-aud platform token + no base_url would otherwise be sent
    # to the OpenAI SDK default endpoint (api.openai.com). Refuse.
    monkeypatch.setenv(_API_KEY_ENV, "platform_tok")
    with pytest.raises(RuntimeError, match="judge base_url is unset"):
        _resolve_judge_key("", base_url=None)


def test_act_as_token_path_resolves_in_training(tmp_path, monkeypatch):
    token_file = tmp_path / "act-as-token"
    token_file.write_text("rotating-jwt")
    monkeypatch.setenv(_TOKEN_PATH_ENV, str(token_file))
    assert _resolve_judge_key("", base_url=_LLM_PROXY) == "rotating-jwt"
