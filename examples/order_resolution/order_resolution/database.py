"""Bounded database access, URL policy, retries, and command reconciliation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from order_resolution.schema import command_receipts

type JsonObject = dict[str, Any]
type CommandWork = Callable[[AsyncConnection, str], Awaitable[JsonObject]]
type ReadWork = Callable[[AsyncConnection], Awaitable[Any]]
type TransactionWork = Callable[[AsyncConnection], Awaitable[Any]]


class DatabaseConfigurationError(ValueError):
    """A database URL violates the direct/runtime connection contract."""


class AmbiguousCommitError(RuntimeError):
    """A commit acknowledgement was lost and must be reconciled by receipt."""


def validate_database_url(database_url: str, *, purpose: str) -> str:
    """Validate a direct admin or pooled runtime PostgreSQL URL without logging it."""

    if purpose not in {"admin", "runtime"}:
        raise ValueError("purpose must be 'admin' or 'runtime'")
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql", "postgresql+psycopg"}:
        raise DatabaseConfigurationError("database URL must use PostgreSQL")
    if not parsed.hostname or not parsed.path.lstrip("/"):
        raise DatabaseConfigurationError("database URL requires a host and database name")
    if "options" in parse_qs(parsed.query):
        raise DatabaseConfigurationError("database URL must not use session options")

    hostname = parsed.hostname.lower()
    if hostname.endswith(".neon.tech"):
        is_pooler = "-pooler." in hostname
        if purpose == "admin" and is_pooler:
            raise DatabaseConfigurationError("migrations require a direct Neon endpoint")
        if purpose == "runtime" and not is_pooler:
            raise DatabaseConfigurationError("runtime DML requires a pooled Neon endpoint")
    return database_url


def _async_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    return urlunsplit(("postgresql+psycopg", parsed.netloc, parsed.path, parsed.query, ""))


def canonical_request_hash(command_name: str, payload: Mapping[str, Any]) -> str:
    """Hash a command and normalized payload for idempotent replay."""

    body = json.dumps(
        {"command": command_name, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(body.encode()).hexdigest()


class Database:
    """Shared async engine with explicit concurrency and retry boundaries."""

    def __init__(
        self,
        runtime_url: str,
        *,
        max_concurrency: int = 64,
        pool_size: int = 16,
        pool_timeout_seconds: float = 15.0,
        after_commit: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        validate_database_url(runtime_url, purpose="runtime")
        if max_concurrency < 1 or pool_size < 1:
            raise ValueError("database concurrency and pool size must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._engine: AsyncEngine = create_async_engine(
            _async_url(runtime_url),
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=0,
            pool_timeout=pool_timeout_seconds,
            connect_args={
                "connect_timeout": 10,
                "prepare_threshold": None,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
                "tcp_user_timeout": 60_000,
            },
        )
        self._after_commit = after_commit

    async def aclose(self) -> None:
        """Close all pooled connections."""

        await self._engine.dispose()

    async def read(self, work: ReadWork) -> Any:
        """Run a safe read with one bounded reconnect retry."""

        for attempt in range(2):
            try:
                async with self._semaphore, self._engine.connect() as connection:
                    return await work(connection)
            except (OperationalError, DBAPIError) as error:
                if attempt or not _connection_failure(error):
                    raise
                await self._engine.dispose()
        raise AssertionError("unreachable read retry state")

    async def transaction(self, work: TransactionWork) -> Any:
        """Run environment-owned setup or cleanup in one bounded transaction."""

        async with self._semaphore, self._engine.connect() as connection:
            async with connection.begin():
                return await work(connection)

    async def execute_command(
        self,
        *,
        world_id: str,
        command_name: str,
        payload: Mapping[str, Any],
        work: CommandWork,
    ) -> JsonObject:
        """Commit one business command and reconcile any ambiguous commit by receipt."""

        request_hash = canonical_request_hash(command_name, payload)
        request_id = f"cmd_{request_hash[:24]}"
        for attempt in range(2):
            try:
                async with self._semaphore, self._engine.connect() as connection:
                    async with connection.begin():
                        existing = await _receipt(
                            connection,
                            world_id=world_id,
                            command_name=command_name,
                            request_hash=request_hash,
                        )
                        if existing is not None:
                            return existing
                        result = await work(connection, request_id)
                        await connection.execute(
                            sa.insert(command_receipts).values(
                                world_id=world_id,
                                receipt_id=request_id,
                                command_name=command_name,
                                request_hash=request_hash,
                                result=result,
                                created_at=sa.func.now(),
                            )
                        )
                if self._after_commit is not None:
                    await self._after_commit(request_id)
                return result
            except IntegrityError:
                reconciled = await self._reconcile(
                    world_id=world_id,
                    command_name=command_name,
                    request_hash=request_hash,
                )
                if reconciled is not None:
                    return reconciled
                raise
            except (AmbiguousCommitError, OperationalError, DBAPIError) as error:
                if not isinstance(error, AmbiguousCommitError) and not _connection_failure(error):
                    raise
                reconciled = await self._reconcile(
                    world_id=world_id,
                    command_name=command_name,
                    request_hash=request_hash,
                )
                if reconciled is not None:
                    return reconciled
                if attempt:
                    raise AmbiguousCommitError(
                        "command receipt is absent after two bounded attempts"
                    ) from error
                await self._engine.dispose()
        raise AssertionError("unreachable command retry state")

    async def _reconcile(
        self,
        *,
        world_id: str,
        command_name: str,
        request_hash: str,
    ) -> JsonObject | None:
        async def read_receipt(connection: AsyncConnection) -> JsonObject | None:
            return await _receipt(
                connection,
                world_id=world_id,
                command_name=command_name,
                request_hash=request_hash,
            )

        return await self.read(read_receipt)


async def _receipt(
    connection: AsyncConnection,
    *,
    world_id: str,
    command_name: str,
    request_hash: str,
) -> JsonObject | None:
    value = await connection.scalar(
        sa.select(command_receipts.c.result).where(
            command_receipts.c.world_id == world_id,
            command_receipts.c.command_name == command_name,
            command_receipts.c.request_hash == request_hash,
        )
    )
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("command receipt result must be a JSON object")
    return value


def _connection_failure(error: BaseException) -> bool:
    if isinstance(error, OperationalError):
        return True
    return isinstance(error, DBAPIError) and error.connection_invalidated


__all__ = [
    "AmbiguousCommitError",
    "Database",
    "DatabaseConfigurationError",
    "canonical_request_hash",
    "validate_database_url",
]
