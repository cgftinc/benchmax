from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmax.rag.search import SearchClient

_SEARCH_PATH = Path(__file__).parents[1] / "search.py"
_SPEC = importlib.util.spec_from_file_location("turbopuffer_rag_example_search", _SEARCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
TurbopufferSearch = _MODULE.TurbopufferSearch


def _row(row_id: int, content: str, file: str = "guide.md") -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id,
        model_extra={"content": content, "file": file, "$dist": 0.2},
    )


@pytest.mark.asyncio
async def test_vector_search_uses_current_query_shape() -> None:
    namespace = SimpleNamespace(query=AsyncMock())
    namespace.query = lambda **kwargs: SimpleNamespace(rows=[_row(1, "answer")])
    embed = AsyncMock(return_value=[[0.1, 0.2]])
    search = TurbopufferSearch("docs", api_key="test-key", embed_fn=embed)
    search._client = namespace

    results = await search.search("question", mode="vector", top_k=3)

    assert isinstance(search, SearchClient)
    assert results == [
        {
            "content": "answer",
            "source": "guide.md",
            "metadata": {"file": "guide.md"},
            "score": 1.0,
        }
    ]


def test_available_modes_is_pure_and_does_not_initialize_client() -> None:
    search = TurbopufferSearch(
        "docs",
        api_key="test-key",
        embed_fn=AsyncMock(),
    )

    assert search.available_modes == ["hybrid", "lexical", "vector"]
    assert search._client is None


@pytest.mark.asyncio
async def test_invalid_mode_fails_before_resolving_key() -> None:
    search = TurbopufferSearch(
        "docs",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="unavailable"):
        await search.search("question", mode="vector")
