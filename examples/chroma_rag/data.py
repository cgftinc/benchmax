"""Developer-side data pipeline for the Chroma RAG example."""

from __future__ import annotations

import os
from pathlib import Path

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.chroma.source import ChromaChunkSource
from castform.rag.example_data import (
    DEFAULT_QUESTION_COUNT,
    RagDataModelConfig,
    RagExampleData,
    SyncOpenAIEmbedder,
)
from chroma_rag_env import COLLECTION_NAME

ROOT = Path(__file__).parent
_DRIVER = RagExampleData(name="chroma-rag", root=ROOT, env_prefix="CHROMA_RAG")
DATA_DIR = _DRIVER.data_dir


def require_dataset_files() -> dict[str, Path]:
    return _DRIVER.require_dataset_files()


def build_chunks() -> ChunkCollection:
    return _DRIVER.build_chunks()


def connection_args() -> dict[str, object]:
    tenant = os.environ.get("CHROMA_TENANT", "").strip()
    database = os.environ.get("CHROMA_DATABASE", "").strip()
    host = os.environ.get("CHROMA_HOST", "").strip()
    if tenant and database:
        return {"tenant": tenant, "database": database}
    if host:
        return {
            "host": host,
            "port": int(os.environ.get("CHROMA_PORT", "8000")),
            "ssl": os.environ.get("CHROMA_SSL", "").lower() in {"1", "true", "yes"},
        }
    raise RuntimeError("configure CHROMA_TENANT and CHROMA_DATABASE for Cloud, or CHROMA_HOST")


def ingest_corpus(chunks: ChunkCollection, config: RagDataModelConfig) -> None:
    args = connection_args()
    api_key = os.environ.get("CHROMA_API_KEY", "").strip() or None
    if args.get("tenant") and not api_key:
        raise RuntimeError("Chroma Cloud ingestion requires CHROMA_API_KEY")
    source = ChromaChunkSource(
        collection_name=COLLECTION_NAME,
        api_key=api_key,
        embed_fn=SyncOpenAIEmbedder(config, request_id="chroma-rag-ingest"),
        enable_bm25=False,
        **args,
    )
    source.populate_from_chunks(chunks)


def prepare_data(
    *,
    force: bool = False,
    question_count: int = DEFAULT_QUESTION_COUNT,
) -> dict[str, Path]:
    return _DRIVER.prepare(
        ingest_corpus,
        force=force,
        target_questions=question_count,
    )
