"""Shared Castform test fixtures."""

import importlib.util
import os
import tempfile
import uuid
from pathlib import Path

import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_HAS_RAG_EXTRA = all(
    _has_module(name)
    for name in (
        "keybert",
        "langchain_text_splitters",
        "numpy",
        "ruamel",
    )
)
_HAS_CHROMA_EXTRA = _has_module("chromadb")
_RAG_EXTRA_TESTS = {
    Path("tests/unit/rag/qa_generation"),
    Path("tests/unit/rag/test_auto_tune.py"),
    Path("tests/unit/rag/test_chunkers.py"),
    Path("tests/unit/rag/test_corpus_profile.py"),
    Path("tests/unit/rag/test_dedup.py"),
    Path("tests/unit/rag/test_deterministic_guards.py"),
    Path("tests/unit/rag/test_entity_quality.py"),
    Path("tests/unit/rag/test_hop_count_validity.py"),
    Path("tests/unit/rag/test_keybert_extraction.py"),
    Path("tests/unit/rag/test_metadata_linker.py"),
    Path("tests/unit/rag/test_micro_batch.py"),
    Path("tests/unit/rag/test_relabel_qa_types.py"),
    Path("tests/unit/rag/test_wiki_builder.py"),
    Path("tests/unit/rag/test_wiki_chunk_linker.py"),
    Path("tests/unit/test_cli_data_qagen.py"),
}


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    rel = Path(collection_path).relative_to(Path(__file__).parent.parent)
    if not _HAS_RAG_EXTRA:
        for path in _RAG_EXTRA_TESTS:
            if rel == path or path in rel.parents:
                return True
    if not _HAS_CHROMA_EXTRA and rel.parts[:5] == (
        "tests",
        "unit",
        "rag",
        "corpus",
        "chroma",
    ):
        return True
    return None

@pytest.fixture
def unique_rollout_id() -> str:
    """Generate a unique rollout ID for testing."""
    return f"test-rollout-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_sync_dir(tmp_path: Path) -> Path:
    """Temporary directory for mocking syncdir (unit tests only)."""
    sync_dir = tmp_path / "sync"
    os.mkdir(sync_dir)
    return sync_dir


@pytest.fixture(scope="session")
def session_tmp_path() -> Path:
    """Temporary directory for test session."""
    return Path(tempfile.mkdtemp(prefix="benchmax_test_session_"))
