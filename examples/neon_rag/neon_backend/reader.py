"""Async read-only Neon transport for this example's rollouts."""

from __future__ import annotations

from typing import Any

from neon_backend.client import NeonClient
from neon_backend.query import NeonQueryRequest, QueryRow, run_query_async


class AsyncNeonReader:
    """Run each query on a fresh read-only async connection.

    The per-operation connection makes grouped rollouts concurrency-safe and
    naturally reconnects after Neon autosuspend. ``NeonClient`` is used only
    as a SQL composer; its synchronous connection path is never opened here.
    """

    def __init__(
        self,
        database_url: str,
        *,
        logical_name: str,
        schema: str,
        text_search_config: str,
    ) -> None:
        if not database_url:
            raise ValueError("database_url must be non-empty")
        self._database_url = database_url
        self._logical_name = logical_name
        self._schema = schema
        self._text_search_config = text_search_config
        self._composer = NeonClient(database_url)

    async def query(self, request: NeonQueryRequest) -> list[QueryRow]:
        """Execute ``request`` in one read transaction and close the socket.

        A connection killed while Neon is suspended is retried once. Queries are
        read-only and the failed transaction is closed before the retry, so this
        cannot duplicate a mutation.
        """

        import psycopg

        for attempt in range(2):
            connection = None
            try:
                connection = await psycopg.AsyncConnection.connect(
                    self._database_url,
                    prepare_threshold=None,
                )
                await self._register_vector(connection)
                return await run_query_async(
                    connection,
                    self._composer,
                    request,
                    logical_name=self._logical_name,
                    schema=self._schema,
                    text_search_config=self._text_search_config,
                )
            except (psycopg.OperationalError, psycopg.InterfaceError):
                if attempt:
                    raise
            finally:
                if connection is not None:
                    await connection.close()
        raise RuntimeError("unreachable Neon query retry state")

    @staticmethod
    async def _register_vector(connection: Any) -> None:
        from pgvector.psycopg import register_vector_async

        await register_vector_async(connection)


__all__ = ["AsyncNeonReader"]
