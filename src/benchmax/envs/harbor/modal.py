from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Awaitable, TypeVar, override

from harbor.environments.modal import ModalEnvironment

logger = logging.getLogger(__name__)

__all__ = ["BoundedModalEnvironment"]

ResultT = TypeVar("ResultT")
_SLOW_FILESYSTEM_OPERATION_SECS = 10.0


class BoundedModalEnvironment(ModalEnvironment):
    """Modal environment with deadlines around provider filesystem operations."""

    def __init__(
        self,
        *args: Any,
        transfer_timeout_secs: float = 120,
        **kwargs: Any,
    ) -> None:
        if transfer_timeout_secs <= 0:
            raise ValueError("transfer_timeout_secs must be positive")
        self._transfer_timeout_secs = float(transfer_timeout_secs)
        super().__init__(*args, **kwargs)

    @override
    async def upload_file(
        self,
        source_path: Path | str,
        target_path: str,
    ) -> None:
        upload = super().upload_file
        await self._bounded_filesystem_operation(
            lambda: upload(source_path, target_path),
            operation="upload_file",
            source=str(source_path),
            target=target_path,
        )

    @override
    async def upload_dir(
        self,
        source_dir: Path | str,
        target_dir: str,
    ) -> None:
        upload = super().upload_dir
        await self._bounded_filesystem_operation(
            lambda: upload(source_dir, target_dir),
            operation="upload_dir",
            source=str(source_dir),
            target=target_dir,
        )

    @override
    async def download_file(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        download = super().download_file
        await self._bounded_filesystem_operation(
            lambda: download(source_path, target_path),
            operation="download_file",
            source=source_path,
            target=str(target_path),
        )

    @override
    async def download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
    ) -> None:
        download = super().download_dir
        await self._bounded_filesystem_operation(
            lambda: download(source_dir, target_dir),
            operation="download_dir",
            source=source_dir,
            target=str(target_dir),
        )

    @override
    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        if self.is_main_service(service):
            await super().service_download_file(
                source_path,
                target_path,
                service=service,
            )
            return
        download = super().service_download_file
        await self._bounded_filesystem_operation(
            lambda: download(source_path, target_path, service=service),
            operation="download_file",
            source=source_path,
            target=str(target_path),
            service=service,
        )

    @override
    async def service_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        if self.is_main_service(service):
            await super().service_download_dir(
                source_dir,
                target_dir,
                service=service,
            )
            return
        download = super().service_download_dir
        await self._bounded_filesystem_operation(
            lambda: download(source_dir, target_dir, service=service),
            operation="download_dir",
            source=source_dir,
            target=str(target_dir),
            service=service,
        )

    @override
    async def service_download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
        service: str | None = None,
    ) -> None:
        download = super().service_download_dir_with_exclusions
        await self._bounded_filesystem_operation(
            lambda: download(
                source_dir=source_dir,
                target_dir=target_dir,
                exclude=exclude,
                service=service,
            ),
            operation="download_dir",
            source=source_dir,
            target=str(target_dir),
            service=service,
        )

    @override
    async def is_dir(
        self,
        path: str,
        user: str | int | None = None,
    ) -> bool:
        is_dir = super().is_dir
        return await self._bounded_filesystem_operation(
            lambda: is_dir(path, user=user),
            operation="is_dir",
            source=path,
            target="-",
        )

    async def _bounded_filesystem_operation(
        self,
        operation_call: Callable[[], Awaitable[ResultT]],
        *,
        operation: str,
        source: str,
        target: str,
        service: str | None = None,
    ) -> ResultT:
        started_at = time.monotonic()
        operation_logger = getattr(self, "logger", logger)
        try:
            async with asyncio.timeout(self._transfer_timeout_secs):
                result = await operation_call()
        except TimeoutError:
            operation_logger.warning(
                "Harbor Modal filesystem operation timed out after %.1fs "
                "operation=%s source=%s target=%s service=%s",
                self._transfer_timeout_secs,
                operation,
                source,
                target,
                service or "main",
            )
            raise
        except Exception as exc:
            operation_logger.warning(
                "Harbor Modal filesystem operation failed after %.1fs "
                "operation=%s source=%s target=%s service=%s error=%s",
                time.monotonic() - started_at,
                operation,
                source,
                target,
                service or "main",
                type(exc).__name__,
            )
            raise

        elapsed = time.monotonic() - started_at
        if elapsed >= _SLOW_FILESYSTEM_OPERATION_SECS:
            operation_logger.warning(
                "Harbor Modal filesystem operation was slow elapsed=%.1fs "
                "operation=%s source=%s target=%s service=%s",
                elapsed,
                operation,
                source,
                target,
                service or "main",
            )
        return result
