from __future__ import annotations

import asyncio
import pickle
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from benchmax.rag.search import SearchClient
from castform.rag.corpus.search_schema.search_exceptions import UnsupportedFilterError
from castform.rag.corpus.search_schema.search_types import FieldOperator, FieldPredicate
from neon_backend import reader as reader_module
from neon_backend.filter_mapper import to_neon_filters
from neon_backend.query import NeonQueryRequest, QueryRow, surfaced_score
from neon_backend.reader import AsyncNeonReader
from neon_backend.search import NeonSearch


def _row(*, content: str = "answer", rank: int = 0) -> QueryRow:
    return QueryRow(
        chunk_id="chunk-1",
        content=content,
        metadata={"file": "guide.md"},
        source_file="guide.md",
        chunk_index=0,
        surfaced_score=surfaced_score(rank),
        native_score=0.9,
        rank=rank,
    )


class _FakeReader:
    def __init__(self, rows: list[QueryRow]) -> None:
        self.rows = rows
        self.requests: list[NeonQueryRequest] = []

    async def query(self, request: NeonQueryRequest) -> list[QueryRow]:
        self.requests.append(request)
        return self.rows


@pytest.mark.asyncio
async def test_search_awaits_embedding_and_returns_search_client_shape() -> None:
    calls: list[list[str]] = []

    async def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.25, 0.75]]

    search = NeonSearch(
        "docs",
        database_url="postgresql://ro@host/db",
        embed_fn=embed,
    )
    reader = _FakeReader([_row()])
    search._client = reader

    results = await search.search("what changed?", mode="hybrid", top_k=3)

    assert isinstance(search, SearchClient)
    assert calls == [["what changed?"]]
    assert reader.requests == [
        NeonQueryRequest(
            mode="hybrid",
            top_k=3,
            text="what changed?",
            vector=(0.25, 0.75),
        )
    ]
    assert results == [
        {
            "content": "answer",
            "source": "guide.md",
            "metadata": {"file": "guide.md"},
            "score": 1 / 60,
        }
    ]


@pytest.mark.asyncio
async def test_lexical_search_does_not_call_embedder() -> None:
    embed = AsyncMock(side_effect=AssertionError("lexical search must not embed"))
    search = NeonSearch(
        "docs",
        database_url="postgresql://ro@host/db",
        embed_fn=embed,
    )
    reader = _FakeReader([])
    search._client = reader

    assert await search.search("keyword", mode="lexical") == []
    embed.assert_not_awaited()
    assert reader.requests[0].vector is None


def test_search_pickle_drops_live_reader_and_carries_database_url() -> None:
    secret = "postgresql://benchmax_ro:secret@host/db"
    search = NeonSearch("docs", database_url=secret)
    search._client = _FakeReader([])

    payload = pickle.dumps(search)
    restored = pickle.loads(payload)

    assert secret.encode() in payload
    assert restored._client is None
    assert restored.get_params() == {
        "backend": "neon",
        "table": "docs",
        "schema": "benchmax_corpus",
    }


def test_neon_public_filter_surface_matches_shared_operator_model() -> None:
    predicate = FieldPredicate(field="year", op="gte", value=2025)
    sql, params = to_neon_filters(predicate)
    assert "::numeric >=" in sql
    assert params == {"k0": "year", "v0": 2025}

    neon_only_op = FieldPredicate(
        field="year",
        op=cast(FieldOperator, "gt"),
        value=2025,
    )
    with pytest.raises(UnsupportedFilterError, match="field operator 'gt' is not supported"):
        to_neon_filters(neon_only_op)


@pytest.mark.asyncio
async def test_async_reader_opens_a_fresh_connection_for_each_concurrent_query(
    monkeypatch,
) -> None:
    connections: list[_FakeConnection] = []

    class _FakeAsyncConnection:
        @staticmethod
        async def connect(dsn: str, *, prepare_threshold: None) -> _FakeConnection:
            assert dsn == "postgresql://ro@host/db"
            assert prepare_threshold is None
            connection = _FakeConnection()
            connections.append(connection)
            return connection

    class _FakePsycopg:
        AsyncConnection = _FakeAsyncConnection

    async def run_query(
        connection: _FakeConnection,
        composer: Any,
        request: NeonQueryRequest,
        **kwargs: Any,
    ) -> list[QueryRow]:
        del composer, kwargs
        await asyncio.sleep(0)
        return [_row(content=request.text or "")]

    monkeypatch.setitem(__import__("sys").modules, "psycopg", _FakePsycopg())
    monkeypatch.setattr(reader_module, "run_query_async", run_query)
    monkeypatch.setattr(AsyncNeonReader, "_register_vector", AsyncMock())
    reader = AsyncNeonReader(
        "postgresql://ro@host/db",
        logical_name="docs",
        schema="benchmax_corpus",
        text_search_config="english",
    )

    first, second = await asyncio.gather(
        reader.query(NeonQueryRequest(mode="lexical", text="first")),
        reader.query(NeonQueryRequest(mode="lexical", text="second")),
    )

    assert [first[0].content, second[0].content] == ["first", "second"]
    assert len(connections) == 2
    assert connections[0] is not connections[1]
    assert all(connection.closed for connection in connections)


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True
