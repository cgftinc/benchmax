"""Unit tests for the `castform runs` command group (slice 1.2).

Offline: the platform client is replaced with a fake returning canned JSON, so
these exercise the CLI formatting + mode-selection logic without a network. The
"output matches the web-app view" half of the gate is the staging fixture check.
"""

from __future__ import annotations

import argparse

from benchmax.cli import runs
from benchmax.platform.exceptions import AuthenticationError, TrainerError


class _FakeClient:
    def __init__(self, **canned):
        self.canned = canned
        self.calls: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def list_runs(self, **_k):
        if "raise" in self.canned:
            raise self.canned["raise"]
        return self.canned.get("runs", [])

    def get_run(self, run_id, **_k):
        return self.canned.get("run", {})

    def get_run_details(self, run_id):
        return self.canned.get("details", {})

    def get_run_scalars(self, run_id, mode):
        self.calls["scalars_mode"] = mode
        return self.canned.get("scalars", {})

    def get_environment_logs(self, run_id, **_k):
        return self.canned.get("logs", [])


def _patch(monkeypatch, **canned) -> _FakeClient:
    client = _FakeClient(**canned)
    monkeypatch.setattr(runs, "trainer_client", lambda: client)
    return client


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_runs_list_table(monkeypatch, capsys):
    _patch(
        monkeypatch,
        runs=[
            {
                "id": "r1",
                "name": "alpha",
                "status": "active",
                "createdAt": "2026-06-16T00:00:00Z",
            }
        ],
    )
    assert runs._cmd_runs_list(_ns(json=False)) == 0
    out = capsys.readouterr().out
    assert "ID" in out and "STATUS" in out
    assert "r1" in out and "alpha" in out and "active" in out


def test_runs_list_empty(monkeypatch, capsys):
    _patch(monkeypatch, runs=[])
    assert runs._cmd_runs_list(_ns(json=False)) == 0
    assert "No runs" in capsys.readouterr().out


def test_runs_list_json(monkeypatch, capsys):
    _patch(monkeypatch, runs=[{"id": "r1"}])
    assert runs._cmd_runs_list(_ns(json=True)) == 0
    assert '"r1"' in capsys.readouterr().out


def test_runs_list_not_logged_in(monkeypatch, capsys):
    _patch(monkeypatch, **{"raise": AuthenticationError("nope", 401)})
    assert runs._cmd_runs_list(_ns(json=False)) == 1
    assert "login" in capsys.readouterr().err


def test_runs_list_server_error(monkeypatch, capsys):
    _patch(monkeypatch, **{"raise": TrainerError("boom", 500)})
    assert runs._cmd_runs_list(_ns(json=False)) == 1
    assert "boom" in capsys.readouterr().err


def test_runs_get(monkeypatch, capsys):
    _patch(
        monkeypatch,
        run={"id": "r1", "name": "alpha", "status": "complete", "isOwner": True},
    )
    assert runs._cmd_runs_get(_ns(run_id="r1", config=False, json=False)) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "complete" in out and "/train/r1" in out


def test_runs_status_with_progress(monkeypatch, capsys):
    _patch(
        monkeypatch,
        run={
            "status": "active",
            "totalSteps": 10,
            "latestActivityMessage": "step 4 done",
        },
        details={"latestStep": 4, "errorCount": 0},
    )
    assert runs._cmd_runs_status(_ns(run_id="r1", json=False)) == 0
    out = capsys.readouterr().out
    assert "active" in out and "4 / 9" in out and "step 4 done" in out


def test_runs_scalars_default_mode_prefers_train(monkeypatch, capsys):
    client = _patch(
        monkeypatch,
        details={"modes": ["eval", "train"]},
        scalars={"reward": [{"step": 1, "value": 0.5}]},
    )
    assert runs._cmd_runs_scalars(_ns(run_id="r1", mode=None, json=False)) == 0
    assert client.calls["scalars_mode"] == "train"  # train preferred over eval
    out = capsys.readouterr().out
    assert "mode=train" in out and "reward" in out


def test_runs_scalars_explicit_mode(monkeypatch, capsys):
    client = _patch(monkeypatch, scalars={"loss": [{"step": 2, "value": 0.1}]})
    assert runs._cmd_runs_scalars(_ns(run_id="r1", mode="eval", json=False)) == 0
    assert client.calls["scalars_mode"] == "eval"
    assert "loss" in capsys.readouterr().out


def test_runs_scalars_no_modes(monkeypatch, capsys):
    _patch(monkeypatch, details={"modes": []})
    assert runs._cmd_runs_scalars(_ns(run_id="r1", mode=None, json=False)) == 0
    assert "No scalars yet" in capsys.readouterr().out


def test_runs_logs(monkeypatch, capsys):
    _patch(
        monkeypatch,
        logs=[
            {
                "createdAt": "t0",
                "level": "ERROR",
                "content": "boom",
                "traceback": "Trace\nline",
            }
        ],
    )
    assert runs._cmd_runs_logs(_ns(run_id="r1", rollout_id=None, json=False)) == 0
    out = capsys.readouterr().out
    assert "ERROR" in out and "boom" in out and "Trace" in out
