"""Unit tests for `castform stop` (slice 1.3). Offline: fake platform client."""

from __future__ import annotations

import argparse

from castform.cli import control
from castform.platform.exceptions import TrainerError


class _FakeClient:
    def __init__(self, result=None, raise_exc=None):
        self.result = result or {}
        self.raise_exc = raise_exc
        self.cancelled = None

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def cancel_run(self, run_id):
        self.cancelled = run_id
        if self.raise_exc:
            raise self.raise_exc
        return self.result


def _patch(monkeypatch, **kw) -> _FakeClient:
    client = _FakeClient(**kw)
    monkeypatch.setattr(control, "trainer_client", lambda: client)
    return client


def test_stop_launched_run(monkeypatch, capsys):
    client = _patch(monkeypatch, result={"success": True, "message": "Job cancellation requested"})
    assert control._cmd_stop(argparse.Namespace(run_id="r1")) == 0
    assert client.cancelled == "r1"
    assert "Job cancellation requested" in capsys.readouterr().out


def test_stop_no_job_run(monkeypatch, capsys):
    _patch(monkeypatch, result={"success": True, "message": "Marked as complete"})
    assert control._cmd_stop(argparse.Namespace(run_id="r1")) == 0
    assert "Marked as complete" in capsys.readouterr().out


def test_stop_not_owner_forbidden(monkeypatch, capsys):
    _patch(monkeypatch, raise_exc=TrainerError("forbidden", 403))
    assert control._cmd_stop(argparse.Namespace(run_id="r1")) == 1
    assert "forbidden" in capsys.readouterr().err
