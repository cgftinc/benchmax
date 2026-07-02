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


# --- corpus list / delete ---------------------------------------------------

import benchmax.rag.corpus.postgres.client as client_mod  # noqa: E402


class _Row:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name
        from datetime import datetime, timezone

        self.created_at = datetime(2026, 6, 1, tzinfo=timezone.utc)


class _FakeClient:
    rows: list = []
    deleted: list = []
    delete_ok = True

    def __init__(self, base_url=None):
        pass

    def list_corpora(self):
        return list(_FakeClient.rows)

    def delete_corpus(self, corpus_id):
        _FakeClient.deleted.append(corpus_id)
        return _FakeClient.delete_ok


def _install_client(monkeypatch, *, rows=None, delete_ok=True):
    _FakeClient.rows = rows if rows is not None else []
    _FakeClient.deleted = []
    _FakeClient.delete_ok = delete_ok
    monkeypatch.setattr(client_mod, "CorpusClient", _FakeClient)


def _ns_list(**kw):
    base = dict(json=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _ns_del(corpus_, **kw):
    base = dict(corpus=corpus_, yes=False, json=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_corpus_list_shows_names_and_cap(monkeypatch, capsys):
    _install_client(monkeypatch, rows=[_Row("id1", "alpha"), _Row("id2", "beta")])
    assert corpus._cmd_corpus_list(_ns_list()) == 0
    out = capsys.readouterr().out
    assert "2/20 corpora" in out and "alpha" in out and "beta" in out


def test_corpus_list_empty(monkeypatch, capsys):
    _install_client(monkeypatch, rows=[])
    assert corpus._cmd_corpus_list(_ns_list()) == 0
    assert "No corpora yet" in capsys.readouterr().out


def test_corpus_list_json(monkeypatch, capsys):
    _install_client(monkeypatch, rows=[_Row("id1", "alpha")])
    assert corpus._cmd_corpus_list(_ns_list(json=True)) == 0
    out = capsys.readouterr().out
    assert "id1" in out and "alpha" in out


def test_corpus_delete_requires_yes(monkeypatch, capsys):
    _install_client(monkeypatch, rows=[_Row("id1", "alpha")])
    assert corpus._cmd_corpus_delete(_ns_del("alpha")) == 1
    assert "without --yes" in capsys.readouterr().err
    assert _FakeClient.deleted == []  # nothing deleted


def test_corpus_delete_by_name_with_yes(monkeypatch, capsys):
    _install_client(monkeypatch, rows=[_Row("id1", "alpha"), _Row("id2", "beta")])
    assert corpus._cmd_corpus_delete(_ns_del("alpha", yes=True)) == 0
    assert _FakeClient.deleted == ["id1"]
    assert "Deleted corpus 'alpha'" in capsys.readouterr().out


def test_corpus_delete_by_id_with_yes(monkeypatch):
    _install_client(monkeypatch, rows=[_Row("id1", "alpha")])
    assert corpus._cmd_corpus_delete(_ns_del("id1", yes=True)) == 0
    assert _FakeClient.deleted == ["id1"]


def test_corpus_delete_not_found(monkeypatch, capsys):
    _install_client(monkeypatch, rows=[_Row("id1", "alpha")])
    assert corpus._cmd_corpus_delete(_ns_del("nope", yes=True)) == 1
    assert "no corpus matching" in capsys.readouterr().err
    assert _FakeClient.deleted == []


# --- corpus search ----------------------------------------------------------

import benchmax.rag.corpus.postgres.search as search_mod  # noqa: E402


class _FakeSearch:
    hits: list = []

    def __init__(self, corpus, base_url=None):
        self.corpus = corpus

    def search(self, query, top_k=5):
        _FakeSearch.last = (query, top_k)
        return list(_FakeSearch.hits)


def _ns_search(corpus_, query, **kw):
    base = dict(corpus=corpus_, query=query, top_k=5, json=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_corpus_search_prints_hits(monkeypatch, capsys):
    _FakeSearch.hits = [{"score": 2.6, "source": "a.md", "content": "hello\nworld"}]
    monkeypatch.setattr(search_mod, "PostgresSearch", _FakeSearch)
    assert corpus._cmd_corpus_search(_ns_search("karpathy", "q")) == 0
    out = capsys.readouterr().out
    assert "1 hit(s)" in out and "a.md" in out and "2.60" in out
    assert _FakeSearch.last == ("q", 5)


def test_corpus_search_empty(monkeypatch, capsys):
    _FakeSearch.hits = []
    monkeypatch.setattr(search_mod, "PostgresSearch", _FakeSearch)
    assert corpus._cmd_corpus_search(_ns_search("karpathy", "q")) == 0
    assert "No results" in capsys.readouterr().out
