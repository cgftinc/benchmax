"""Developer-side data pipeline for the Pinecone RAG example."""

from __future__ import annotations

import os
from pathlib import Path

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.pinecone.source import PineconeChunkSource
from castform.rag.example_data import (
    DEFAULT_QUESTION_COUNT,
    RagDataModelConfig,
    RagExampleData,
    SyncOpenAIEmbedder,
)
from pinecone_rag_env import INDEX_NAME, NAMESPACE

ROOT = Path(__file__).parent
_DRIVER = RagExampleData(name="pinecone-rag", root=ROOT, env_prefix="PINECONE_RAG")
DATA_DIR = _DRIVER.data_dir


def require_dataset_files() -> dict[str, Path]:
    return _DRIVER.require_dataset_files()


def build_chunks() -> ChunkCollection:
    return _DRIVER.build_chunks()


def index_host() -> str:
    value = os.environ.get("PINECONE_INDEX_HOST", "").strip()
    if not value:
        raise RuntimeError("configure PINECONE_INDEX_HOST")
    return value


def ingest_corpus(chunks: ChunkCollection, config: RagDataModelConfig) -> None:
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ingestion requires PINECONE_API_KEY")
    source = PineconeChunkSource(
        api_key=api_key,
        index_name=INDEX_NAME,
        index_host=index_host(),
        namespace=NAMESPACE,
        embed_fn=SyncOpenAIEmbedder(config, request_id="pinecone-rag-ingest"),
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
