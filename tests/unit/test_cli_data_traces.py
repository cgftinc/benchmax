"""Offline: `castform data traces` (faked Braintrust adapter + TracesPipeline).

Guards project resolution, the no-key error, and that output lands under the
project-convention filenames — without any network or the real traces lib.
"""

from __future__ import annotations

import argparse
import json

import benchmax.traces as traces_mod
import benchmax.traces.braintrust.adapter as adapter_mod
from benchmax.cli import build_parser, data
from benchmax.traces.adapter import TraceProject


class _FakeAdapter:
    projects: list[TraceProject] = [TraceProject(id="p1", name="proj")]

    def __init__(self, api_key):
        self.api_key = api_key
        self.fetched_project: str | None = None

    def list_projects(self):
        return list(_FakeAdapter.projects)

    def fetch_traces(self, project_id, *, limit=None, cursor=None):
        self.fetched_project = project_id
        _FakeAdapter.last = self  # type: ignore[attr-defined]
        return ([object()], None)  # one (opaque) trace — pipeline is faked


class _FakePipeline:
    def __init__(self, **kw):
        self.kw = kw

    def run(self):
        return {
            "train_dataset": [{"a": 1}],
            "eval_dataset": [{"b": 2}],
            "stats": {"train_count": 1, "eval_count": 1},
            "system_prompt": "SP",
            "tools": [{"name": "search"}],
        }


def _install(monkeypatch, *, projects=None):
    if projects is not None:
        _FakeAdapter.projects = projects
    else:
        _FakeAdapter.projects = [TraceProject(id="p1", name="proj")]
    monkeypatch.setattr(adapter_mod, "BraintrustTraceAdapter", _FakeAdapter)
    monkeypatch.setattr(traces_mod, "TracesPipeline", _FakePipeline)


def _ns(**kw):
    base = dict(
        api_key=None,
        project_id=None,
        project=None,
        limit=None,
        system_prompt=None,
        max_examples=1000,
        out=".",
        json=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_traces_writes_project_filenames(monkeypatch, tmp_path):
    _install(monkeypatch)
    monkeypatch.setenv("BT_API_KEY", "bt-key")
    monkeypatch.setenv("BT_PROJECT_ID", "p1")
    rc = data._cmd_data_traces(_ns(out=str(tmp_path)))
    assert rc == 0
    train = tmp_path / "train_dataset.jsonl"
    eval_ = tmp_path / "eval_dataset.jsonl"
    assert json.loads(train.read_text()) == {"a": 1}
    assert json.loads(eval_.read_text()) == {"b": 2}


def test_traces_no_api_key_is_clean(monkeypatch, capsys):
    _install(monkeypatch)
    monkeypatch.delenv("BT_API_KEY", raising=False)
    assert data._cmd_data_traces(_ns()) == 1
    assert "Braintrust API key" in capsys.readouterr().err


def test_traces_resolves_project_by_name(monkeypatch):
    _install(
        monkeypatch,
        projects=[TraceProject(id="p1", name="alpha"), TraceProject(id="p2", name="beta")],
    )
    monkeypatch.setenv("BT_API_KEY", "bt-key")
    monkeypatch.delenv("BT_PROJECT_ID", raising=False)
    assert data._cmd_data_traces(_ns(project="beta", out=".")) == 0
    assert _FakeAdapter.last.fetched_project == "p2"  # type: ignore[attr-defined]


def test_traces_ambiguous_project_is_clean(monkeypatch, capsys):
    _install(
        monkeypatch,
        projects=[TraceProject(id="p1", name="alpha"), TraceProject(id="p2", name="beta")],
    )
    monkeypatch.setenv("BT_API_KEY", "bt-key")
    monkeypatch.delenv("BT_PROJECT_ID", raising=False)
    assert data._cmd_data_traces(_ns()) == 1
    assert "specify a project" in capsys.readouterr().err


def test_traces_registered_in_parser():
    args = build_parser().parse_args(["data", "traces", "--project", "x"])
    assert args.func is data._cmd_data_traces
    assert args.project == "x"
