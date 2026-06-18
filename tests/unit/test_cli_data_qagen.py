"""Offline: `castform data qa-gen` config-building (mocked run_pipeline).

Guards the `--fast` lever (quality_gate-only filters, refinement off, entity
patterns off) and that output lands under the project-convention filenames — both
without running the real (LLM-backed) pipeline.
"""

from __future__ import annotations

import argparse

import pytest

import benchmax.rag.qa_generation.pipeline as pipeline_mod
from benchmax.cli import build_parser, data

_FULL_FILTERS = [
    "quality_gate",
    "retrieval_too_easy_llm",
    "grounding_llm",
    "hop_count_validity",
]


def _install(monkeypatch, result=None):
    """Patch run_pipeline; return a dict capturing the cfg the verb built."""
    captured: dict = {}

    def _fake(cfg, **kwargs):
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        return result or {
            "output_paths": {
                "train_jsonl": "train_dataset.jsonl",
                "eval_jsonl": "eval_dataset.jsonl",
            },
            "stats": {"train": 5, "eval": 1, "total": 6},
        }

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake)
    return captured


def _ns(**kw):
    base = dict(
        corpus_name=None,
        corpus_id=None,
        provider=None,
        samples=50,
        min_chunk_chars=None,
        fast=False,
        out=".",
        json=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_qa_gen_fast_config(monkeypatch):
    captured = _install(monkeypatch)
    rc = data._cmd_data_qa_gen(
        _ns(corpus_name="karpathy", samples=6, fast=True, out="/tmp/x")
    )
    assert rc == 0
    cfg = captured["cfg"]
    assert cfg.filtering.filters == ["quality_gate"]
    assert cfg.refinement.enabled is False
    assert cfg.corpus_context.generate_entity_patterns is False
    assert cfg.corpus_context.enabled is True  # linker profile stays on
    assert cfg.targets.total_samples == 6
    assert cfg.corpus.corpus_name == "karpathy"
    # Project-convention filenames written to --out.
    assert cfg.output.train_jsonl == "train_dataset.jsonl"
    assert cfg.output.eval_jsonl == "eval_dataset.jsonl"
    assert cfg.output.dir == "/tmp/x"


def test_qa_gen_default_full_filters(monkeypatch):
    captured = _install(monkeypatch)
    assert data._cmd_data_qa_gen(_ns(corpus_name="karpathy")) == 0
    cfg = captured["cfg"]
    assert cfg.filtering.filters == _FULL_FILTERS
    assert cfg.refinement.enabled is True
    assert cfg.corpus_context.generate_entity_patterns is True


def test_qa_gen_corpus_id_never_uses_docs_path(monkeypatch):
    captured = _install(monkeypatch)
    assert data._cmd_data_qa_gen(_ns(corpus_id="abc-123")) == 0
    cfg = captured["cfg"]
    assert cfg.corpus.corpus_id == "abc-123"
    assert cfg.corpus.docs_path == ""  # never the interactive get_or_create branch


def test_qa_gen_prints_project_filenames(monkeypatch, capsys):
    _install(monkeypatch)
    assert data._cmd_data_qa_gen(_ns(corpus_name="karpathy", fast=True)) == 0
    out = capsys.readouterr().out
    assert "train_dataset.jsonl" in out and "eval_dataset.jsonl" in out


def test_qa_gen_corpus_not_found_is_clean(monkeypatch, capsys):
    def _boom(cfg, **kw):
        raise ValueError("Could not find existing corpus named 'nope'.")

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _boom)
    assert data._cmd_data_qa_gen(_ns(corpus_name="nope")) == 1
    assert "Could not find existing corpus" in capsys.readouterr().err


def test_qa_gen_requires_a_corpus():
    # The mutually-exclusive group is required: neither flag → argparse exits.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "qa-gen"])


def test_qa_gen_min_chunk_chars_overrides_corpus_floor(monkeypatch):
    # lower the eligibility floor for small docs; unset → lib default (not None).
    captured = _install(monkeypatch)
    assert data._cmd_data_qa_gen(_ns(corpus_name="x", min_chunk_chars=120)) == 0
    assert captured["cfg"].corpus.min_chunk_chars == 120
    captured2 = _install(monkeypatch)
    assert data._cmd_data_qa_gen(_ns(corpus_name="x")) == 0
    assert captured2["cfg"].corpus.min_chunk_chars == 400  # lib default preserved


def test_qa_gen_provider_passes_source_factory(monkeypatch):
    # --provider reads DATA_* env and hands run_pipeline a source_factory; the
    # corpus label comes from the provider's resource id (turbopuffer → namespace).
    captured = _install(monkeypatch)
    monkeypatch.setenv("DATA_api_key", "tpuf-key")
    monkeypatch.setenv("DATA_namespace", "myns")
    rc = data._cmd_data_qa_gen(_ns(provider="turbopuffer"))
    assert rc == 0
    assert callable(captured["kwargs"]["source_factory"])
    assert captured["cfg"].corpus.corpus_name == "myns"


def test_qa_gen_provider_missing_env_is_clean(monkeypatch, capsys):
    # chroma needs DATA_collection_name; absent → a clean Error, no traceback.
    monkeypatch.delenv("DATA_collection_name", raising=False)
    assert data._cmd_data_qa_gen(_ns(provider="chroma")) == 1
    assert "DATA_collection_name" in capsys.readouterr().err


def test_qa_gen_provider_is_mutually_exclusive_with_corpus():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["data", "qa-gen", "--corpus-name", "x", "--provider", "chroma"]
        )
