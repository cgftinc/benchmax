"""Unit tests for `castform doctor`."""

from __future__ import annotations

import argparse
import json

from benchmax.cli import doctor


def _run(monkeypatch, *, signed_in: bool, py_ok: bool = True, json_mode: bool = False):
    """Drive _cmd_doctor with the auth + interpreter checks stubbed."""
    monkeypatch.setattr(
        doctor,
        "_auth_check",
        lambda: (signed_in, "you@x (x)" if signed_in else "not signed in"),
    )
    monkeypatch.setattr(doctor, "_py_check", lambda: (py_ok, "3.12.0"))
    monkeypatch.setattr(doctor, "_version", lambda: "0.0.0")
    monkeypatch.setattr(doctor, "extra_is_installed", lambda name: name == "rag")
    return doctor._cmd_doctor(argparse.Namespace(json=json_mode))


def test_ready_returns_zero(monkeypatch, capsys):
    assert _run(monkeypatch, signed_in=True) == 0
    out = capsys.readouterr().out
    assert "ready to `castform validate`" in out
    # An absent extra shows its install hint; a present one says installed.
    assert "castform[turbopuffer]" in out
    assert "installed" in out


def test_not_signed_in_is_blocking(monkeypatch, capsys):
    assert _run(monkeypatch, signed_in=False) == 1
    assert "not ready" in capsys.readouterr().out


def test_wrong_python_is_blocking(monkeypatch):
    assert _run(monkeypatch, signed_in=True, py_ok=False) == 1


def test_json_mode_shape_and_exit(monkeypatch, capsys):
    rc = _run(monkeypatch, signed_in=True, json_mode=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["python_ok"] is True
    assert payload["signed_in"] is True
    assert payload["extras"] == {
        "rag": True,
        "turbopuffer": False,
        "pinecone": False,
        "chroma": False,
    }


def test_json_mode_not_signed_in_exit_one(monkeypatch, capsys):
    rc = _run(monkeypatch, signed_in=False, json_mode=True)
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["signed_in"] is False
