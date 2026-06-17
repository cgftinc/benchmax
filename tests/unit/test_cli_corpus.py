"""Offline: `castform corpus ingest` (mocked PostgresChunkSource).

Guards the non-interactive contract: the verb must forward ``on_limit="error"`` to
``populate_from_folder`` (so the lib never reaches its ``input()`` prompt at the
5-corpus cap) and surface ``CorpusLimitError`` as a clean message.
"""

from __future__ import annotations

import argparse

import benchmax.rag.corpus.postgres.source as src_mod
from benchmax.cli import corpus
from benchmax.rag.corpus.postgres.exceptions import CorpusLimitError


class _Corpus:  # minimal stub: cap message reads .name off existing_corpora
    def __init__(self, name: str) -> None:
        self.name = name


def _install(monkeypatch, *, raise_exc=None, corpus_id="corpus_abc123"):
    """Patch the source seam; return a dict capturing how the verb called it."""
    captured: dict = {}

    class _Fake:
        def __init__(self, corpus_name, api_key="", base_url=None):
            captured["corpus_name"] = corpus_name

        def populate_from_folder(self, docs_path, **kwargs):
            captured["docs_path"] = docs_path
            captured["populate_kwargs"] = kwargs
            if raise_exc is not None:
                raise raise_exc

        @property
        def corpus_id(self):
            return corpus_id

    monkeypatch.setattr(src_mod, "PostgresChunkSource", _Fake)
    return captured


def _ns(folder, **kw):
    base = dict(folder=folder, name=None, json=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_ingest_ok_forwards_on_limit_error(monkeypatch, tmp_path, capsys):
    captured = _install(monkeypatch)
    assert corpus._cmd_corpus_ingest(_ns(str(tmp_path))) == 0
    # The non-interactive guarantee: input() is never reachable.
    assert captured["populate_kwargs"]["on_limit"] == "error"
    assert captured["docs_path"] == str(tmp_path)
    assert "corpus_abc123" in capsys.readouterr().out


def test_ingest_default_name_is_folder_basename(monkeypatch, tmp_path):
    captured = _install(monkeypatch)
    assert corpus._cmd_corpus_ingest(_ns(str(tmp_path))) == 0
    assert captured["corpus_name"] == tmp_path.resolve().name


def test_ingest_custom_name(monkeypatch, tmp_path):
    captured = _install(monkeypatch)
    assert corpus._cmd_corpus_ingest(_ns(str(tmp_path), name="my-docs")) == 0
    assert captured["corpus_name"] == "my-docs"


def test_ingest_json(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, corpus_id="corpus_xyz")
    assert corpus._cmd_corpus_ingest(_ns(str(tmp_path), json=True)) == 0
    out = capsys.readouterr().out
    assert "corpus_xyz" in out and "corpus_id" in out


def test_ingest_not_a_folder(monkeypatch, capsys):
    _install(monkeypatch)
    assert corpus._cmd_corpus_ingest(_ns("/tmp/does-not-exist-xyz-123")) == 1
    assert "not a folder" in capsys.readouterr().err


def test_ingest_cap_error_is_clean(monkeypatch, tmp_path, capsys):
    exc = CorpusLimitError(existing_corpora=[_Corpus("posthog"), _Corpus("karpathy")])
    _install(monkeypatch, raise_exc=exc)
    assert corpus._cmd_corpus_ingest(_ns(str(tmp_path))) == 1
    err = capsys.readouterr().err
    assert "5-corpus cap" in err and "posthog" in err and "karpathy" in err


def test_ingest_missing_rag_dep_hints_install(monkeypatch, tmp_path, capsys):
    exc = ModuleNotFoundError("No module named 'langchain_text_splitters'")
    _install(monkeypatch, raise_exc=exc)
    assert corpus._cmd_corpus_ingest(_ns(str(tmp_path))) == 1
    assert "pip install castform[rag]" in capsys.readouterr().err
