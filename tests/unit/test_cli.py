"""Unit tests for the `castform` CLI (login/logout/whoami + ensure_session)."""

from __future__ import annotations

import argparse
import base64
import json

import pytest

from benchmax import cli, config
from benchmax import profile_config as profiles
from benchmax.platform import browser, credentials, login
from benchmax.platform.device_auth import DeviceAuthError


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CASTFORM_CONFIG_PATH", str(tmp_path / "castform.toml"))
    monkeypatch.delenv("CASTFORM_PROFILE", raising=False)


def _raise_device_auth(*_a, **_k):
    raise DeviceAuthError("nope")


def _raise_runtime(*_a, **_k):
    raise RuntimeError("no credential")


@pytest.fixture
def stub_flow(monkeypatch):
    """Stub the device flow + capture what login would persist."""
    captured: dict = {}
    monkeypatch.setenv("CASTFORM_AUTH_URL", "http://localhost:4300")
    # _login (which _cmd_login calls) lives in benchmax.platform.login and uses
    # request_device_code / poll_for_token imported there.
    monkeypatch.setattr(
        login,
        "request_device_code",
        lambda _auth, **_k: {
            "device_code": "dc",
            "user_code": "ABCD1234",
            "verification_uri": "https://app.x/device",
            "interval": 5,
            "expires_in": 1800,
        },
    )
    monkeypatch.setattr(
        login,
        "poll_for_token",
        lambda _auth, _dc, **_k: {"access_token": "sess_abc", "expires_in": 604800},
    )
    monkeypatch.setattr(
        profiles,
        "upsert_profile",
        lambda name, **kw: captured.update(profile=name, **kw),
    )
    monkeypatch.setattr(
        config,
        "base_domain",
        lambda _profile=None: captured.get("domain") or "castform.com",
    )
    # This test is about session writing, not browser UX — stub the open so it
    # never launches a real browser (e.g. under `pytest -s`, where stdin is a tty).
    monkeypatch.setattr(login, "maybe_open_browser", lambda _u: None)
    return captured


def test_login_writes_session(stub_flow):
    assert cli._cmd_login(argparse.Namespace()) == 0
    s = stub_flow["session"]
    assert stub_flow["profile"] == "prod"
    assert stub_flow["domain"] == "castform.com"
    assert s["access_token"] == "sess_abc"
    assert s["expires_at"] > 0


def test_login_profile_domain_writes_profile_session(stub_flow):
    args = argparse.Namespace(
        profile="staging",
        domain="castform.dev",
        api_url=None,
        llm_url=None,
        auth_url=None,
        app_url=None,
    )
    assert cli._cmd_login(args) == 0
    assert stub_flow["profile"] == "staging"
    assert stub_flow["domain"] == "castform.dev"
    assert stub_flow["session"]["access_token"] == "sess_abc"


def test_login_failure_returns_1(monkeypatch):
    monkeypatch.setenv("CASTFORM_AUTH_URL", "http://localhost:4300")
    monkeypatch.setattr(login, "request_device_code", _raise_device_auth)
    assert cli._cmd_login(argparse.Namespace()) == 1


def test_logout_clears_session(monkeypatch):
    called = {}
    monkeypatch.setattr(
        credentials,
        "clear_castform_session",
        lambda **kw: called.update(kw) or called.setdefault("cleared", True),
    )
    assert cli._cmd_logout(argparse.Namespace()) == 0
    assert called["cleared"]
    assert called["profile"] == "prod"


def test_whoami_not_logged_in(monkeypatch):
    monkeypatch.setattr(credentials, "read_castform_session", lambda _profile=None: None)
    assert cli._cmd_whoami(argparse.Namespace()) == 1


def test_whoami_logged_in_shows_email(monkeypatch, capsys):
    monkeypatch.setattr(
        credentials, "read_castform_session", lambda _profile=None: {"access_token": "x"}
    )
    claims = base64.urlsafe_b64encode(json.dumps({"email": "a@b.com"}).encode()).rstrip(
        b"="
    )
    monkeypatch.setattr(credentials, "_session_jwt", lambda _profile=None: f"h.{claims.decode()}.s")
    assert cli._cmd_whoami(argparse.Namespace()) == 0
    assert "a@b.com" in capsys.readouterr().out


def test_ensure_session_noop_when_credential_resolves(monkeypatch):
    monkeypatch.setattr(credentials, "platform_bearer", lambda: "tok")
    called = {}
    monkeypatch.setattr(login, "_login", lambda: called.setdefault("login", True))
    login.ensure_session()
    assert "login" not in called


def test_ensure_session_skips_when_not_interactive(monkeypatch):
    monkeypatch.delenv("CASTFORM_NO_AUTO_LOGIN", raising=False)
    monkeypatch.setattr(credentials, "platform_bearer", _raise_runtime)
    called = {}
    monkeypatch.setattr(login, "_login", lambda: called.setdefault("login", True))
    login.ensure_session(interactive=False)
    assert "login" not in called


def test_ensure_session_logs_in_when_interactive(monkeypatch):
    monkeypatch.delenv("CASTFORM_NO_AUTO_LOGIN", raising=False)
    monkeypatch.setattr(credentials, "platform_bearer", _raise_runtime)
    called = {}
    monkeypatch.setattr(login, "_login", lambda: called.setdefault("login", True))
    login.ensure_session(interactive=True)
    assert called.get("login")


class _FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_maybe_open_browser_opens_on_tty(monkeypatch):
    monkeypatch.setattr(browser.sys, "stdin", _FakeStdin(True))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("CASTFORM_NO_BROWSER", raising=False)
    opened = {}
    monkeypatch.setattr(browser.webbrowser, "open", lambda u: opened.setdefault("url", u))
    browser.maybe_open_browser("https://app.x/device?user_code=ABCD")
    assert opened["url"] == "https://app.x/device?user_code=ABCD"


def test_maybe_open_browser_skips_over_ssh(monkeypatch):
    monkeypatch.setattr(browser.sys, "stdin", _FakeStdin(True))
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    monkeypatch.delenv("CASTFORM_NO_BROWSER", raising=False)
    opened = {}
    monkeypatch.setattr(browser.webbrowser, "open", lambda u: opened.setdefault("url", u))
    browser.maybe_open_browser("https://app.x/device")
    assert "url" not in opened


def test_maybe_open_browser_skips_non_interactive(monkeypatch):
    monkeypatch.setattr(browser.sys, "stdin", _FakeStdin(False))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("CASTFORM_NO_BROWSER", raising=False)
    opened = {}
    monkeypatch.setattr(browser.webbrowser, "open", lambda u: opened.setdefault("url", u))
    browser.maybe_open_browser("https://app.x/device")
    assert "url" not in opened


def test_maybe_open_browser_opt_out(monkeypatch):
    monkeypatch.setattr(browser.sys, "stdin", _FakeStdin(True))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("CASTFORM_NO_BROWSER", "1")
    opened = {}
    monkeypatch.setattr(browser.webbrowser, "open", lambda u: opened.setdefault("url", u))
    browser.maybe_open_browser("https://app.x/device")
    assert "url" not in opened
