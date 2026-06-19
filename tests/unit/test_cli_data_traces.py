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
        dry_run=False,
        preview=False,
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


def test_traces_dry_run_prints_full_prompt_and_writes_nothing(
    monkeypatch, tmp_path, capsys
):
    # --dry-run surfaces the FULL detected prompt (no 200-char "…" truncation) +
    # tools, and writes no datasets — a confirm-before-generate step.
    long_prompt = "You are a meticulous research agent. " * 20  # > 200 chars

    class _LongPipeline(_FakePipeline):
        def run(self):
            return {
                "train_dataset": [{"a": 1}],
                "eval_dataset": [{"b": 2}],
                "stats": {"train_count": 1, "eval_count": 1},
                "system_prompt": long_prompt,
                "tools": [{"name": "search"}, {"name": "fetch"}],
            }

    _install(monkeypatch)
    monkeypatch.setattr(traces_mod, "TracesPipeline", _LongPipeline)
    monkeypatch.setenv("BT_API_KEY", "bt-key")
    monkeypatch.setenv("BT_PROJECT_ID", "p1")

    rc = data._cmd_data_traces(_ns(out=str(tmp_path), dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert long_prompt in out  # full, untruncated
    assert "…" not in out  # the normal-summary ellipsis must NOT appear
    assert "search" in out and "fetch" in out
    assert not (tmp_path / "train_dataset.jsonl").exists()
    assert not (tmp_path / "eval_dataset.jsonl").exists()


def test_traces_dry_run_json_writes_nothing(monkeypatch, tmp_path, capsys):
    # The --json dry-run branch must ALSO write no datasets — guards a regression
    # that hoists the write above the dry-run short-circuit.
    _install(monkeypatch)
    monkeypatch.setenv("BT_API_KEY", "bt-key")
    monkeypatch.setenv("BT_PROJECT_ID", "p1")
    rc = data._cmd_data_traces(_ns(out=str(tmp_path), dry_run=True, json=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert '"system_prompt"' in out and '"tools"' in out  # JSON emitted
    assert not (tmp_path / "train_dataset.jsonl").exists()
    assert not (tmp_path / "eval_dataset.jsonl").exists()


def test_traces_dry_run_in_parser():
    args = build_parser().parse_args(["data", "traces", "--project", "x", "--dry-run"])
    assert args.dry_run is True


# --- data traces --preview --------------------------------------------------

import benchmax.platform.browser as browser_mod  # noqa: E402
from benchmax.traces.adapter import NormalizedTrace, TraceMessage  # noqa: E402


class _PreviewAdapter(_FakeAdapter):
    """Like _FakeAdapter but returns real NormalizedTrace objects for preview."""

    def fetch_traces(self, project_id, *, limit=None, cursor=None):
        self.fetched_project = project_id
        traces = [
            NormalizedTrace(
                id="t-1",
                messages=[TraceMessage(role="user", content="hi"),
                          TraceMessage(role="assistant", content="hello")],
                scores={"helpfulness": 0.9},
                metadata={"provider": "braintrust"},
            )
        ]
        return (traces, None)


def _install_preview(monkeypatch):
    _FakeAdapter.projects = [TraceProject(id="p1", name="proj")]
    monkeypatch.setattr(adapter_mod, "BraintrustTraceAdapter", _PreviewAdapter)
    # If --preview ever fell through to the pipeline, this fake would prove it.
    monkeypatch.setattr(traces_mod, "TracesPipeline", _FakePipeline)


def test_traces_preview_renders_and_skips_dataset(monkeypatch, tmp_path, capsys):
    _install_preview(monkeypatch)
    monkeypatch.setenv("BT_API_KEY", "bt-key")
    monkeypatch.setenv("BT_PROJECT_ID", "p1")
    opened: dict = {}
    monkeypatch.setattr(browser_mod, "maybe_open_browser", lambda u: opened.setdefault("url", u))

    rc = data._cmd_data_traces(_ns(out=str(tmp_path), preview=True))
    assert rc == 0
    snapshot = tmp_path / "traces_preview.jsonl"
    html = tmp_path / "traces_preview.html"
    assert snapshot.exists() and html.exists()
    # Preview must NOT build the train/eval dataset.
    assert not (tmp_path / "train_dataset.jsonl").exists()
    assert not (tmp_path / "eval_dataset.jsonl").exists()
    # Snapshot is the normalized-trace shape; the viewer embeds it as traces.
    assert json.loads(snapshot.read_text().splitlines()[0])["id"] == "t-1"
    body = html.read_text()
    start = body.index('id="data">') + len('id="data">')
    model = json.loads(body[start : body.index("</script>", start)])
    assert model["kind"] == "traces" and model["variant"] == "normalized-trace"
    assert opened["url"] == html.resolve().as_uri()
    assert "not built into a dataset" in capsys.readouterr().out


def test_traces_preview_registered_flag():
    args = build_parser().parse_args(["data", "traces", "--project", "x", "--preview"])
    assert args.preview is True
