"""Unit tests for BenchMax's platform-independent judge-auth contract."""

from __future__ import annotations

import pytest

from benchmax.auth import InjectedAuth, StaticBearerAuth
from benchmax.rubrics._utils import _resolve_judge_auth


def test_missing_judge_auth_fails_with_declarative_remedy():
    with pytest.raises(RuntimeError, match=r"InjectedAuth\('judge'\)"):
        _resolve_judge_auth(None, "", None)


def test_explicit_static_key_is_supported():
    resolved = _resolve_judge_auth(None, "sk-customer", None)
    assert resolved == StaticBearerAuth("sk-customer")


def test_injected_auth_is_preserved_for_call_time_resolution():
    auth = InjectedAuth("judge")
    assert _resolve_judge_auth(auth, "", None) is auth


def test_multiple_auth_sources_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_judge_auth(InjectedAuth("judge"), "sk-customer", None)
