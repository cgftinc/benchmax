"""Offline: `castform data view` load + detect + normalize (slice 1).

Fixtures mirror the real on-disk JSONL shapes (derived from the writer code,
not the web-app viewer's post-transform model): qa-gen ``_to_row``, scaffold
``{prompt, ground_truth}``, trace-example ``to_jsonl_dict``, and a
normalized-trace via an adapter round-trip. Guards classification, per-shape
normalization, the ``--type`` override, the row cap, and the generic fallback
(which must never raise on unrecognized input).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from urllib.parse import unquote, urlparse

from benchmax.cli import build_parser, data
from benchmax.cli.dataview import build_view_model, load_view_model, render_html, write_html
from benchmax.platform import browser as browser_mod
from benchmax.traces.adapter import NormalizedTrace, TraceMessage

# --- fixtures (one real line per shape) ------------------------------------

QAGEN = {
    "question": "What does _to_row emit?",
    "answer": "question/answer/qa_type/reference_chunks plus optional fields.",
    "qa_type": "single_hop",
    "reference_chunks": ["doc1#0", "doc1#1"],
    "difficulty_score": 0.4,
    "reasoning_mode": "extractive",
}

SCAFFOLD = {"prompt": "Solve 2+2.", "ground_truth": "4"}

TRACE_EXAMPLE = {
    "prompt_messages": [
        {"role": "user", "content": "weather?", "tool_calls": [], "tool_call_id": "", "name": ""},
    ],
    "ground_truth": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ],
        "tool_call_id": "",
        "name": "",
    },
    "init_rollout_args": {
        "trace_id": "trace-42",
        "turn_index": 2,
        "total_messages": 3,
        "scores": {"helpfulness": 0.9},
        "raw_prompt": "weather?",
    },
}

TRACE_EXAMPLE_NO_GT = {
    "prompt_messages": [
        {"role": "user", "content": "hi", "tool_calls": [], "tool_call_id": "", "name": ""},
    ],
    "ground_truth": {},
    "init_rollout_args": {
        "trace_id": "trace-43",
        "turn_index": 0,
        "total_messages": 1,
        "scores": {},
        "raw_prompt": "hi",
    },
}


def _normalized_trace_record() -> dict:
    nt = NormalizedTrace(
        id="norm-7",
        messages=[
            TraceMessage(role="system", content="be terse"),
            TraceMessage(role="user", content="2+2?"),
        ],
        scores={"accuracy": 1.0},
        metadata={"provider": "langfuse"},
    )
    return nt.to_dict()


def _vm(record, **kw):
    return build_view_model([record], source="f.jsonl", **kw)


# --- classification ---------------------------------------------------------


def test_classify_all_four_shapes():
    assert (_vm(QAGEN)["kind"], _vm(QAGEN)["variant"]) == ("rag", "qa-gen")
    assert (_vm(SCAFFOLD)["kind"], _vm(SCAFFOLD)["variant"]) == ("rag", "scaffold")
    assert (_vm(TRACE_EXAMPLE)["kind"], _vm(TRACE_EXAMPLE)["variant"]) == (
        "traces",
        "trace-example",
    )
    nt = _normalized_trace_record()
    assert (_vm(nt)["kind"], _vm(nt)["variant"]) == ("traces", "normalized-trace")


# --- RAG normalization ------------------------------------------------------


def test_qagen_preserves_reference_chunks_and_rolls_extra():
    row = _vm(QAGEN)["rows"][0]
    assert row["question"] == QAGEN["question"]
    assert row["answer"] == QAGEN["answer"]
    assert row["qa_type"] == "single_hop"
    assert row["reference_chunks"] == ["doc1#0", "doc1#1"]
    # optional fields roll into extra; the four named fields are excluded.
    assert row["extra"] == {"difficulty_score": 0.4, "reasoning_mode": "extractive"}


def test_scaffold_maps_prompt_ground_truth_to_question_answer():
    row = _vm(SCAFFOLD)["rows"][0]
    assert row["question"] == "Solve 2+2."
    assert row["answer"] == "4"
    assert row["qa_type"] is None
    assert row["reference_chunks"] is None
    assert row["extra"] is None  # canonical 2-key scaffold has no leftovers


# --- trace normalization ----------------------------------------------------


def test_trace_example_pulls_init_rollout_args_and_ground_truth():
    row = _vm(TRACE_EXAMPLE)["rows"][0]
    assert row["id"] == "trace-42"
    assert row["turn_index"] == 2
    assert row["scores"] == {"helpfulness": 0.9}
    # raw_prompt/total_messages land in metadata, not the top-level fields.
    assert row["metadata"]["raw_prompt"] == "weather?"
    assert row["metadata"]["total_messages"] == 3
    assert len(row["prompt_messages"]) == 1
    # ground_truth (a non-empty dict) becomes the single completion message,
    # and its tool-call arguments survive as a JSON string for the JS to parse.
    assert len(row["completion_messages"]) == 1
    tc = row["completion_messages"][0]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == '{"city": "NYC"}'


def test_trace_example_empty_ground_truth_yields_no_completion():
    row = _vm(TRACE_EXAMPLE_NO_GT)["rows"][0]
    assert row["completion_messages"] == []


def test_normalized_trace_all_messages_are_prompt_messages():
    row = _vm(_normalized_trace_record())["rows"][0]
    assert row["id"] == "norm-7"
    assert row["turn_index"] is None
    assert row["scores"] == {"accuracy": 1.0}
    assert row["metadata"]["provider"] == "langfuse"
    assert [m["role"] for m in row["prompt_messages"]] == ["system", "user"]
    assert row["completion_messages"] == []


# --- --type override --------------------------------------------------------


def test_type_override_forces_classification():
    # A qa-gen row forced to generic / traces.
    assert _vm(QAGEN, type_override="generic")["kind"] == "generic"
    assert _vm(QAGEN, type_override="traces")["kind"] == "traces"
    # A trace row forced to rag still sub-detects a sane variant.
    forced = _vm(TRACE_EXAMPLE, type_override="rag")
    assert forced["kind"] == "rag"
    assert forced["variant"] in {"qa-gen", "scaffold"}


# --- row cap ----------------------------------------------------------------


def test_cap_marks_capped_and_shows_subset():
    model = build_view_model([QAGEN] * 5, source="f.jsonl", limit=2)
    assert model["total"] == 5
    assert model["shown"] == 2
    assert model["capped"] is True
    assert len(model["rows"]) == 2


def test_limit_zero_shows_all_uncapped():
    model = build_view_model([QAGEN] * 5, source="f.jsonl", limit=0)
    assert model["shown"] == 5 and model["capped"] is False


# --- generic fallback (never errors) ----------------------------------------


def test_unrecognized_shape_is_generic_with_raw_rows_and_key_union():
    a = {"foo": 1, "bar": 2}
    b = {"foo": 3, "baz": 4}
    model = build_view_model([a, b], source="f.jsonl")
    assert model["kind"] == "generic"
    assert model["variant"] == "unknown"
    assert model["rows"] == [a, b]  # raw records preserved verbatim
    # union of top-level keys, "foo" first (appears in both records).
    assert model["columns"][0] == "foo"
    assert set(model["columns"]) == {"foo", "bar", "baz"}


def test_type_generic_forces_generic_on_a_known_shape():
    model = _vm(QAGEN, type_override="generic")
    assert model["kind"] == "generic"
    assert model["rows"][0] == QAGEN  # not normalized — raw
    assert "columns" in model


# --- file-level loading -----------------------------------------------------


def test_load_empty_file_does_not_error(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    model = load_view_model(p)
    assert model["kind"] == "generic" and model["total"] == 0 and model["rows"] == []


def test_load_skips_blank_and_unparseable_lines(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text("\n".join(["", json.dumps(QAGEN), "{not json", json.dumps(QAGEN), ""]))
    model = load_view_model(p)
    assert model["kind"] == "rag" and model["total"] == 2
    assert model["source"] == "mixed.jsonl"


# --- chunks (corpus browse) -------------------------------------------------

CHUNK = {
    "id": "chunk-1",
    "content": "The capital of France is Paris.",
    "metadata": {"source": "geo.md", "page": 3},
    "score": 0.83,
}
CHUNK_NO_SCORE = {"id": "chunk-2", "content": "Berlin is the capital of Germany.", "metadata": {}}


def test_chunks_detected_and_normalized():
    model = _vm(CHUNK)
    assert (model["kind"], model["variant"]) == ("chunks", "corpus-chunk")
    row = model["rows"][0]
    assert row["id"] == "chunk-1"
    assert row["content"] == "The capital of France is Paris."
    assert row["score"] == 0.83
    assert row["metadata"] == {"source": "geo.md", "page": 3}
    assert row["extra"] is None


def test_chunk_without_score_is_tolerated():
    row = _vm(CHUNK_NO_SCORE)["rows"][0]
    assert row["score"] is None and row["metadata"] == {}


def test_chunk_extra_fields_roll_into_extra():
    row = _vm({**CHUNK, "embedding_model": "text-embedding-3-large"})["rows"][0]
    assert row["extra"] == {"embedding_model": "text-embedding-3-large"}


def test_type_chunks_forces_classification():
    # A generic record forced to chunks still normalizes to the chunk row shape.
    model = _vm({"foo": 1}, type_override="chunks")
    assert model["kind"] == "chunks"
    assert model["rows"][0]["content"] == "" and model["rows"][0]["extra"] == {"foo": 1}


def test_chunks_render_embeds_kind():
    embedded = json.loads(_payload(render_html(_vm(CHUNK))))
    assert embedded["kind"] == "chunks"


# --- Claude Code session transcript ----------------------------------------

CLAUDE_TRANSCRIPT = [
    {"type": "mode", "sessionId": "s1", "mode": "default"},  # meta — ignored
    {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": "fix the bug"}},
    {"type": "assistant", "sessionId": "s1", "message": {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "let me look"},
        {"type": "text", "text": "I'll search."},
        {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "bug"}},
    ]}},
    {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "found at line 5"},
    ]}},
    {"type": "ai-title", "sessionId": "s1", "title": "x"},  # meta — ignored
    {"type": "assistant", "sessionId": "s1", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "Fixed it."},
    ]}},
]


def test_claude_transcript_collapses_to_one_conversation():
    model = build_view_model(CLAUDE_TRANSCRIPT, source="s.jsonl")
    assert model["kind"] == "traces" and model["variant"] == "claude-transcript"
    assert model["total"] == 1 and len(model["rows"]) == 1
    msgs = model["rows"][0]["prompt_messages"]
    # user → (thinking, assistant+toolcall) → tool result → assistant
    assert [m["role"] for m in msgs] == ["user", "thinking", "assistant", "tool", "assistant"]
    # tool_use → a tool call with JSON-parseable arguments
    asst = msgs[2]
    assert asst["tool_calls"][0]["function"]["name"] == "grep"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"q": "bug"}
    # tool_result → a tool message carrying the originating tool_use id
    assert msgs[3]["content"] == "found at line 5" and msgs[3]["tool_call_id"] == "t1"


def test_claude_type_override_forces_reshape():
    one = [{"type": "user", "sessionId": "s", "message": {"role": "user", "content": "hi"}}]
    model = build_view_model(one, source="s.jsonl", type_override="claude")
    assert model["variant"] == "claude-transcript"
    assert model["rows"][0]["prompt_messages"][0]["content"] == "hi"


def test_normalized_trace_not_misdetected_as_claude():
    # A normalized-trace file (no sessionId / type events) must stay itself.
    model = build_view_model([_normalized_trace_record()], source="t.jsonl")
    assert model["variant"] == "normalized-trace"


# --- HTML render (slice 2) --------------------------------------------------


def _payload(html: str) -> str:
    """Extract the embedded JSON text from the data <script> block."""
    marker = 'id="data">'
    start = html.index(marker) + len(marker)
    return html[start : html.index("</script>", start)]


def test_render_embeds_parseable_escaped_payload():
    # A row whose content would break out of the <script> block if embedded raw.
    rec = {"question": "</script><img src=x onerror=alert(1)>", "answer": "a & b < c > d"}
    model = build_view_model([rec], source="x.jsonl")
    html = render_html(model)
    payload = _payload(html)
    # Round-trips to the exact model → embedded AND parseable AND escaping is
    # lossless (< decodes back to the original "<").
    assert json.loads(payload) == model
    # No breakout: the dangerous bytes are unicode-escaped, not literal.
    assert "</script>" not in payload
    assert "<" not in payload and ">" not in payload
    assert "\\u003c" in payload


def test_render_reflects_cap():
    model = build_view_model([QAGEN] * 5, source="x.jsonl", limit=2)
    embedded = json.loads(_payload(render_html(model)))
    assert embedded["capped"] is True
    assert embedded["shown"] == 2 and embedded["total"] == 5


def test_render_injects_initial_view_mode():
    model = build_view_model([QAGEN], source="x.jsonl")
    assert 'data-view="raw"' in render_html(model, view="raw")
    # An unknown mode falls back to auto.
    assert 'data-view="auto"' in render_html(model, view="bogus")


def test_write_html_creates_nonempty_file(tmp_path):
    model = build_view_model([TRACE_EXAMPLE], source="t.jsonl")
    out = write_html(model, tmp_path / "view.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and 'id="data"' in text and len(text) > 1000


# --- `data view` command e2e (slice 3) --------------------------------------


class _FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def _view_ns(file, **kw):
    base = dict(file=str(file), type=None, view="auto", out=None, limit=1000, no_open=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _capture_open(monkeypatch):
    """Stub the browser open so tests never launch a real browser; capture URL."""
    opened: dict = {}
    monkeypatch.setattr(browser_mod, "maybe_open_browser", lambda u: opened.setdefault("url", u))
    return opened


def test_view_e2e_all_shapes(tmp_path, monkeypatch, capsys):
    opened = _capture_open(monkeypatch)
    cases = [
        ("qa.jsonl", [QAGEN], "rag", "qa-gen"),
        ("scaffold.jsonl", [SCAFFOLD], "rag", "scaffold"),
        ("te.jsonl", [TRACE_EXAMPLE], "traces", "trace-example"),
        ("nt.jsonl", [_normalized_trace_record()], "traces", "normalized-trace"),
        ("gen.jsonl", [{"foo": 1, "bar": 2}], "generic", "unknown"),
    ]
    for name, rows, kind, variant in cases:
        opened.clear()
        src = _write(tmp_path, name, rows)
        out = tmp_path / (name + ".html")
        rc = data._cmd_data_view(_view_ns(src, out=str(out)))
        assert rc == 0, name
        assert out.exists() and out.stat().st_size > 0
        printed = capsys.readouterr().out
        assert str(out) in printed and f"{kind}/{variant}" in printed
        # Opened with a file:// URI pointing at the written file.
        assert opened["url"] == out.resolve().as_uri()


def test_view_no_open_skips_browser(tmp_path, monkeypatch, capsys):
    opened = _capture_open(monkeypatch)
    src = _write(tmp_path, "q.jsonl", [QAGEN])
    out = tmp_path / "q.html"
    rc = data._cmd_data_view(_view_ns(src, out=str(out), no_open=True))
    assert rc == 0 and out.exists()
    assert "url" not in opened  # --no-open → open never called
    assert str(out) in capsys.readouterr().out  # path still printed


def test_view_non_tty_suppresses_open_but_prints_path(tmp_path, monkeypatch, capsys):
    # Real maybe_open_browser runs; a non-tty stdin makes it a no-op (no browser
    # launched even under `pytest -s`) while the handler still prints the path.
    monkeypatch.setattr(browser_mod.sys, "stdin", _FakeStdin(False))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("CASTFORM_NO_BROWSER", raising=False)
    opened: dict = {}
    monkeypatch.setattr(browser_mod.webbrowser, "open", lambda u: opened.setdefault("url", u))
    src = _write(tmp_path, "q.jsonl", [QAGEN])
    out = tmp_path / "q.html"
    rc = data._cmd_data_view(_view_ns(src, out=str(out)))
    assert rc == 0 and "url" not in opened
    assert str(out) in capsys.readouterr().out


def test_view_no_browser_env_suppresses_open_but_prints_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(browser_mod.sys, "stdin", _FakeStdin(True))  # tty, so env is the cause
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("CASTFORM_NO_BROWSER", "1")
    opened: dict = {}
    monkeypatch.setattr(browser_mod.webbrowser, "open", lambda u: opened.setdefault("url", u))
    src = _write(tmp_path, "q.jsonl", [QAGEN])
    out = tmp_path / "q.html"
    rc = data._cmd_data_view(_view_ns(src, out=str(out)))
    assert rc == 0 and "url" not in opened
    assert str(out) in capsys.readouterr().out


def test_view_type_and_view_flags_honored(tmp_path, monkeypatch):
    _capture_open(monkeypatch)
    src = _write(tmp_path, "q.jsonl", [QAGEN])
    out = tmp_path / "q.html"
    rc = data._cmd_data_view(_view_ns(src, out=str(out), type="generic", view="raw"))
    assert rc == 0
    html = out.read_text()
    start = html.index('id="data">') + len('id="data">')
    embedded = json.loads(html[start : html.index("</script>", start)])
    assert embedded["kind"] == "generic"  # --type generic forced classification
    assert 'data-view="raw"' in html  # --view raw injected as the initial mode


def test_view_default_out_is_temp_html(tmp_path, monkeypatch, capsys):
    opened = _capture_open(monkeypatch)
    src = _write(tmp_path, "q.jsonl", [QAGEN])
    rc = data._cmd_data_view(_view_ns(src))  # no --out → temp file
    assert rc == 0
    url = opened["url"]
    assert url.startswith("file://") and url.endswith(".html")
    written = pathlib.Path(unquote(urlparse(url).path))
    tmp_root = pathlib.Path(tempfile.gettempdir()).resolve()  # macOS /var → /private/var
    assert tmp_root in written.parents and written.name.startswith("castform-view-")
    # The printed path is the raw (un-resolved) temp path; match on the basename.
    assert written.exists() and written.name in capsys.readouterr().out
    written.unlink()  # tidy up the persistent temp file


def test_view_missing_file_returns_1(tmp_path, capsys):
    rc = data._cmd_data_view(_view_ns(tmp_path / "nope.jsonl"))
    assert rc == 1
    assert "file not found" in capsys.readouterr().err


def test_view_parser_exposes_flags():
    ns = build_parser().parse_args(
        ["data", "view", "f.jsonl", "--type", "traces", "--view", "messages",
         "--limit", "5", "--no-open", "--out", "o.html"]
    )
    assert ns.command == "data" and ns.data_command == "view"
    assert ns.file == "f.jsonl" and ns.type == "traces" and ns.view == "messages"
    assert ns.limit == 5 and ns.no_open is True and ns.out == "o.html"
    assert ns.func is data._cmd_data_view
