"""Developer-side ingestion and dataset generation for the Neon RAG example."""

from __future__ import annotations

from pathlib import Path

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.example_data import (
    DEFAULT_QUESTION_COUNT,
    RagDataModelConfig,
    RagExampleData,
    SyncOpenAIEmbedder,
)
from neon_backend.config import CORPUS_NAME
from neon_backend.source import NeonChunkSource

ROOT = Path(__file__).parent
_DRIVER = RagExampleData(name="neon-rag", root=ROOT, env_prefix="NEON_RAG")
DATA_DIR = _DRIVER.data_dir


def model_config() -> RagDataModelConfig:
    return _DRIVER.model_config()


def dataset_files() -> dict[str, Path]:
    return _DRIVER.dataset_files()


def require_dataset_files() -> dict[str, Path]:
    return _DRIVER.require_dataset_files()


def build_chunks() -> ChunkCollection:
    return _DRIVER.build_chunks()


def ingest_corpus(
    chunks: ChunkCollection,
    config: RagDataModelConfig,
    *,
    data_preparation_database_url: str,
    batch_size: int = 64,
) -> None:
    """Publish a fresh, atomically-versioned Neon corpus from ``chunks``."""

    source = NeonChunkSource(
        CORPUS_NAME,
        embed_fn=SyncOpenAIEmbedder(config, request_id="neon-rag-ingest"),
        data_preparation_database_url=data_preparation_database_url,
    )
    source.populate_from_chunks(chunks, batch_size=batch_size)


def prepare_data(
    *,
    data_preparation_database_url: str,
    force: bool = False,
    question_count: int = DEFAULT_QUESTION_COUNT,
) -> dict[str, Path]:
    def ingest(chunks: ChunkCollection, config: RagDataModelConfig) -> None:
        ingest_corpus(
            chunks,
            config,
            data_preparation_database_url=data_preparation_database_url,
        )

    return _DRIVER.prepare(
        ingest,
        force=force,
        target_questions=question_count,
    )
