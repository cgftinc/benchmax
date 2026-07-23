"""Thin programmatic entrypoint for running qa-gen against a Neon corpus.

The qa-gen pipeline needs no change to target Neon: it already accepts an
injectable ``source_factory`` (``run_pipeline(cfg, source_factory=...)``), and
the default ``_load_source`` hard-codes ``PostgresChunkSource`` only as the
fallback when no factory is supplied. This module supplies the Neon factory.

Two pins live here on purpose (Slice 6):

* **Endpoint domain (NB2).** The platform LLM endpoint is ``llm.castform.dev``
  (judge + generator, and the ``text-embedding-3-large`` embeddings), but the
  ambient ``CASTFORM_BASE_DOMAIN`` default resolves to ``castform.com``. Both
  the generation/judge base URL (via ``cfg.platform.llm_base_url``) and the
  embeddings base URL (via the ``embed_fn``) are pinned to ``base_domain``
  (default ``castform.dev``) so the Neon path hits the correct host regardless
  of the ambient default.
* **DSN resolution (NB5).** Ingest/reads use only the explicit Neon DSN seam
  (``NEON_CORPUS_DSN_RO`` / ``NEON_CORPUS_DSN_RW`` via ``read_dsn_provider`` /
  ``write_dsn_provider``); a bare ``DATABASE_URL`` never satisfies a Neon DSN
  (see ``rag/corpus/neon/credentials.py``).

Ingestion stays programmatic (``ChunkSource.populate_from_*``); the factory here
binds a reader to the already-active corpus version by ``logical_name`` and does
not populate — the corpus is expected to exist before qa-gen runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from castform.platform.credentials import TokenProvider

if TYPE_CHECKING:
    from castform.rag.corpus.neon.source import NeonChunkSource
    from castform.rag.qa_generation.pipeline_config import PipelineConfig

# The platform LLM endpoint domain (llm.<domain>/v1). Pinned so the Neon qa-gen
# path reaches llm.castform.dev, not the ambient CASTFORM_BASE_DOMAIN default.
PLATFORM_BASE_DOMAIN = "castform.dev"


def neon_llm_url(base_domain: str = PLATFORM_BASE_DOMAIN) -> str:
    """Return the pinned OpenAI-compatible LLM base URL for the Neon qa-gen path.

    Independent of ``config.llm_url()`` / ``CASTFORM_BASE_DOMAIN`` on purpose:
    the Neon path must reach ``llm.castform.dev`` even when the ambient default
    resolves elsewhere (NB2).
    """
    return f"https://llm.{base_domain}/v1"


def neon_source_factory(
    *,
    read_dsn_provider: str | TokenProvider | None = None,
    write_dsn_provider: str | TokenProvider | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    base_domain: str = PLATFORM_BASE_DOMAIN,
    schema: str | None = None,
) -> Callable[[PipelineConfig], NeonChunkSource]:
    """Build a ``source_factory`` that binds qa-gen to a Neon corpus.

    The returned callable maps a resolved ``PipelineConfig`` to a
    :class:`NeonChunkSource` on the active version of ``cfg.corpus.corpus_name``.
    When ``embed_fn`` is not supplied it defaults to a platform ``embed_fn`` with
    its base URL pinned to ``base_domain`` (so vector/hybrid retrieval hits the
    correct embeddings host); pass an explicit ``embed_fn`` to override.

    Args:
        read_dsn_provider: Read-only Neon DSN seam; ``None`` reads
            ``NEON_CORPUS_DSN_RO`` from the environment per connection.
        write_dsn_provider: Read-write Neon DSN seam (ingest); ``None`` reads
            ``NEON_CORPUS_DSN_RW``.
        embed_fn: Embedding function for vector/hybrid modes; defaults to a
            platform ``embed_fn`` pinned to ``base_domain``.
        base_domain: Platform endpoint domain for the default ``embed_fn``.
        schema: Postgres schema the corpus lives in; ``None`` uses the source
            default.
    """

    def _factory(cfg: PipelineConfig) -> NeonChunkSource:
        from castform.rag.corpus.embed import platform_embed_fn
        from castform.rag.corpus.neon.source import NeonChunkSource

        fn = embed_fn
        if fn is None:
            fn = platform_embed_fn(
                base_url=neon_llm_url(base_domain),
                api_key=cfg.platform.llm_api_key or cfg.platform.api_key,
            )
        kwargs: dict[str, Any] = {
            "embed_fn": fn,
            "read_dsn_provider": read_dsn_provider,
            "write_dsn_provider": write_dsn_provider,
        }
        if schema is not None:
            kwargs["schema"] = schema
        return NeonChunkSource(cfg.corpus.corpus_name, **kwargs)

    return _factory


def run_qa_gen_on_neon(
    cfg: PipelineConfig,
    *,
    read_dsn_provider: str | TokenProvider | None = None,
    write_dsn_provider: str | TokenProvider | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    base_domain: str = PLATFORM_BASE_DOMAIN,
    schema: str | None = None,
    rollout_client_factory: Callable[[PipelineConfig], Any] | None = None,
) -> dict[str, Any]:
    """Run the qa-gen pipeline against a Neon corpus.

    Pins the generation/judge LLM base URL to ``base_domain`` (NB2) and injects a
    :func:`neon_source_factory`, so the pipeline never touches the Postgres
    default path. All other config comes from ``cfg``.
    """
    from castform.rag.qa_generation.pipeline import run_pipeline

    cfg.platform.llm_base_url = neon_llm_url(base_domain)
    factory = neon_source_factory(
        read_dsn_provider=read_dsn_provider,
        write_dsn_provider=write_dsn_provider,
        embed_fn=embed_fn,
        base_domain=base_domain,
        schema=schema,
    )
    return run_pipeline(
        cfg,
        source_factory=factory,
        rollout_client_factory=rollout_client_factory,
    )


__all__ = [
    "PLATFORM_BASE_DOMAIN",
    "neon_llm_url",
    "neon_source_factory",
    "run_qa_gen_on_neon",
]
