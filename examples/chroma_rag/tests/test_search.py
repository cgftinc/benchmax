from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmax.rag.search import SearchClient

_SEARCH_PATH = Path(__file__).parents[1] / "search.py"
_SPEC = importlib.util.spec_from_file_location("chroma_rag_example_search", _SEARCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ChromaSearch = _MODULE.ChromaSearch


@pytest.mark.asyncio
async def test_vector_search_uses_explicit_embeddings_and_current_query_api() -> None:
    collection = SimpleNamespace(
        query=lambda **kwargs: {
            "documents": [["answer"]],
            "metadatas": [[{"file_path": "guide.md"}]],
            "distances": [[0.25]],
        }
    )
    embed = AsyncMock(return_value=[[0.1, 0.2]])
    search = ChromaSearch("docs", host="chroma.local", embed_fn=embed)
    search._collection = collection

    results = await search.search("question", top_k=3)

    assert isinstance(search, SearchClient)
    assert results == [
        {
            "content": "answer",
            "source": "guide.md",
            "metadata": {"file_path": "guide.md"},
            "score": 0.8,
        }
    ]


def test_available_modes_is_static_without_initializing_cloud_client() -> None:
    search = ChromaSearch(
        "docs",
        tenant="tenant",
        database="database",
        api_key="test-key",
        embed_fn=AsyncMock(),
    )

    assert search.available_modes == ["vector"]
    assert search._client is None


@pytest.mark.asyncio
async def test_lexical_mode_is_rejected_before_connecting() -> None:
    search = ChromaSearch("docs", host="chroma.local", embed_fn=AsyncMock())

    with pytest.raises(ValueError, match="only vector"):
        await search.search("question", mode="lexical")
    assert search._client is None
