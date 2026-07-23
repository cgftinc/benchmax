"""Slice 6 — qa-gen wiring to Neon + the two endpoint/fallback pins.

Covers:
- the DI seam (an injected Neon ``source_factory`` bypasses the Postgres default),
- NB5 (a bare ``DATABASE_URL`` never satisfies a Neon DSN),
- NB2 (the Neon qa-gen path pins the LLM endpoint to ``llm.castform.dev``).
"""

from __future__ import annotations

import pytest
from fakes.neon import make_neon_source

from castform.rag.corpus.neon.credentials import (
    READ_DSN_ENV_VAR,
    WRITE_DSN_ENV_VAR,
    resolve_read_dsn_provider,
    resolve_write_dsn_provider,
)
from castform.rag.corpus.neon.source import NeonChunkSource
from castform.rag.corpus.postgres.source import PostgresChunkSource
from castform.rag.corpus.source import ChunkSource
from castform.rag.qa_generation import pipeline as pipeline_mod
from castform.rag.qa_generation.neon_entrypoint import (
    neon_llm_url,
    neon_source_factory,
    run_qa_gen_on_neon,
)
from castform.rag.qa_generation.pipeline import Pipeline
from castform.rag.qa_generation.pipeline_config import CorpusConfig, PipelineConfig
from castform.platform.config import PlatformConfig

PINNED_LLM_URL = "https://llm.castform.dev/v1"


def _cfg(corpus_name: str = "mycorpus") -> PipelineConfig:
    return PipelineConfig(
        platform=PlatformConfig(),
        corpus=CorpusConfig(corpus_name=corpus_name),
    )


# --- item 1: qa-gen DI seam ------------------------------------------------


def test_neon_source_factory_builds_neon_source_not_postgres() -> None:
    source = neon_source_factory()(_cfg("wiki"))
    assert isinstance(source, NeonChunkSource)
    assert not isinstance(source, PostgresChunkSource)
    assert source._logical_name == "wiki"
    # Structural conformance to the ChunkSource protocol the pipeline expects.
    assert isinstance(source, ChunkSource)


def test_injected_factory_bypasses_postgres_default(monkeypatch) -> None:
    # A fake Neon source (real NeonChunkSource, fake collaborators, no DB).
    fake = make_neon_source(logical_name="wiki")
    monkeypatch.setattr(
        pipeline_mod,
        "_load_source",
        lambda cfg: pytest.fail("Postgres default path was taken"),
    )

    pipeline = Pipeline(_cfg("wiki"), source_factory=lambda cfg: fake)

    # The seam _prepare_context resolves the source through: never _load_source.
    assert pipeline.source_factory is not pipeline_mod._load_source
    assert pipeline.source_factory(pipeline.cfg) is fake
    assert isinstance(pipeline.source_factory(pipeline.cfg), NeonChunkSource)


def test_default_pipeline_falls_back_to_postgres_load_source() -> None:
    # Guards the flip side: no factory ⇒ the Postgres default is bound.
    assert Pipeline(_cfg()).source_factory is pipeline_mod._load_source


# --- item 2 (NB5): DATABASE_URL must not satisfy a Neon DSN -----------------


def test_bare_database_url_does_not_satisfy_read_dsn(monkeypatch) -> None:
    monkeypatch.delenv(READ_DSN_ENV_VAR, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://someone@wrong-db/whatever")
    provider = resolve_read_dsn_provider()
    with pytest.raises(RuntimeError, match=READ_DSN_ENV_VAR):
        provider()


def test_bare_database_url_does_not_satisfy_write_dsn(monkeypatch) -> None:
    monkeypatch.delenv(WRITE_DSN_ENV_VAR, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://someone@wrong-db/whatever")
    provider = resolve_write_dsn_provider()
    with pytest.raises(RuntimeError, match=WRITE_DSN_ENV_VAR):
        provider()


def test_explicit_neon_dsn_still_resolves(monkeypatch) -> None:
    # The narrowing rejects only the generic fallback, not the real seam.
    monkeypatch.setenv("DATABASE_URL", "postgresql://noise@host/db")
    monkeypatch.setenv(READ_DSN_ENV_VAR, "postgresql://ro@neon/db")
    assert resolve_read_dsn_provider()() == "postgresql://ro@neon/db"


# --- item 3 (NB2): pin the platform endpoint domain ------------------------


def test_neon_llm_url_pinned_regardless_of_ambient_domain(monkeypatch) -> None:
    # The pin ignores the ambient CASTFORM_BASE_DOMAIN default (which is .com).
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    assert neon_llm_url() == PINNED_LLM_URL


def test_default_embed_fn_pinned_to_castform_dev() -> None:
    source = neon_source_factory()(_cfg())
    # PlatformEmbedFn resolves config at call time; the base_url is pinned here.
    assert source._embed_fn._base_url == PINNED_LLM_URL


def test_run_qa_gen_on_neon_pins_llm_base_url(monkeypatch) -> None:
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    captured: dict = {}

    def _fake_run_pipeline(cfg, *, source_factory, rollout_client_factory=None):
        captured["cfg"] = cfg
        captured["source"] = source_factory(cfg)
        return {"ok": True}

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_run_pipeline)

    cfg = _cfg("wiki")
    result = run_qa_gen_on_neon(cfg)

    assert result == {"ok": True}
    assert captured["cfg"].platform.llm_base_url == PINNED_LLM_URL
    assert isinstance(captured["source"], NeonChunkSource)
    assert captured["source"]._embed_fn._base_url == PINNED_LLM_URL


def test_pin_overwrites_pre_resolved_generator_and_judge_urls(monkeypatch) -> None:
    # A config as load_pipeline_config() would hand us: the generator/judge base
    # URLs are ALREADY populated with .com, so resolve_api_keys (fill-if-unset)
    # would never overwrite them. The pin must, or generation/judge traffic
    # silently goes to castform.com.
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    cfg = _cfg("wiki")
    cfg.platform.llm_base_url = "https://llm.castform.com/v1"
    cfg.resolve_api_keys()
    assert cfg.generation.llm_direct.base_url == "https://llm.castform.com/v1"
    assert cfg.filtering.grounding_llm.judge_base_url == "https://llm.castform.com/v1"

    captured: dict = {}

    def _fake_run_pipeline(cfg, *, source_factory, rollout_client_factory=None):
        captured["cfg"] = cfg
        return {}

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_run_pipeline)

    run_qa_gen_on_neon(cfg)

    c = captured["cfg"]
    assert c.generation.llm_direct.base_url == PINNED_LLM_URL
    assert c.filtering.retrieval_llm.judge_base_url == PINNED_LLM_URL
    assert c.filtering.grounding_llm.judge_base_url == PINNED_LLM_URL
    assert c.filtering.hop_count_validity.judge_base_url == PINNED_LLM_URL
    assert "castform.com" not in c.generation.llm_direct.base_url
