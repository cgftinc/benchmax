from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmax.rag.search import SearchClient

_SEARCH_PATH = Path(__file__).parents[1] / "search.py"
_SPEC = importlib.util.spec_from_file_location("pinecone_rag_example_search", _SEARCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
PineconeSearch = _MODULE.PineconeSearch


@pytest.mark.asyncio
async def test_search_uses_current_data_plane_query_shape() -> None:
    calls: list[dict] = []
    index = SimpleNamespace(
        query=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                matches=[
                    SimpleNamespace(
                        score=0.91,
                        metadata={"content": "answer", "file_path": "guide.md"},
                    )
                ]
            )
        )
    )
    embed = AsyncMock(return_value=[[0.1, 0.2]])
    search = PineconeSearch(
        "https://index.example",
        api_key="test-key",
        embed_fn=embed,
    )
    search._index = index

    results = await search.search("question", top_k=3)

    assert isinstance(search, SearchClient)
    assert calls == [
        {
            "namespace": "",
            "vector": [0.1, 0.2],
            "top_k": 3,
            "include_metadata": True,
            "include_values": False,
        }
    ]
    assert results == [
        {
            "content": "answer",
            "source": "guide.md",
            "metadata": {"file_path": "guide.md"},
            "score": 0.91,
        }
    ]


def test_available_modes_does_not_initialize_index() -> None:
    search = PineconeSearch(
        "https://index.example",
        api_key="test-key",
        embed_fn=AsyncMock(),
    )

    assert search.available_modes == ["vector"]
    assert search._index is None


@pytest.mark.asyncio
async def test_lexical_mode_is_rejected_before_connecting() -> None:
    search = PineconeSearch(
        "https://index.example",
        api_key="test-key",
        embed_fn=AsyncMock(),
    )

    with pytest.raises(ValueError, match="only vector"):
        await search.search("question", mode="lexical")
    assert search._index is None
