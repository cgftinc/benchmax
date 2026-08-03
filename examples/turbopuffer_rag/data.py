"""Developer-side data pipeline for the TurboPuffer RAG example."""

from __future__ import annotations

import os
from pathlib import Path

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.turbopuffer.source import TpufChunkSource
from castform.rag.example_data import (
    DEFAULT_QUESTION_COUNT,
    RagDataModelConfig,
    RagExampleData,
    SyncOpenAIEmbedder,
)
from turbopuffer_rag_env import NAMESPACE, REGION

ROOT = Path(__file__).parent
_DRIVER = RagExampleData(name="turbopuffer-rag", root=ROOT, env_prefix="TURBOPUFFER_RAG")
DATA_DIR = _DRIVER.data_dir


def require_dataset_files() -> dict[str, Path]:
    return _DRIVER.require_dataset_files()


def build_chunks() -> ChunkCollection:
    return _DRIVER.build_chunks()


def ingest_corpus(chunks: ChunkCollection, config: RagDataModelConfig) -> None:
    api_key = os.environ.get("TPUF_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ingestion requires TPUF_API_KEY")
    embedder = SyncOpenAIEmbedder(config, request_id="turbopuffer-rag-ingest")
    source = TpufChunkSource(
        api_key=api_key,
        namespace=NAMESPACE,
        region=REGION,
        embed_fn=embedder,
    )
    source.populate_from_chunks(chunks, embed_fn=embedder)


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
