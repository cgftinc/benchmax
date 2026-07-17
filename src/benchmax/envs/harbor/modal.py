from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Awaitable, override

from harbor.environments.modal import ModalEnvironment

logger = logging.getLogger(__name__)

__all__ = ["BoundedModalEnvironment"]


class BoundedModalEnvironment(ModalEnvironment):
    """Modal environment with deadlines around provider filesystem transfers."""

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
        await self._bounded_transfer(
            lambda: upload(source_path, target_path),
            operation="upload_file",
            source=str(source_path),
            target=target_path,
            attempts=2,
        )

    @override
    async def upload_dir(
        self,
        source_dir: Path | str,
        target_dir: str,
    ) -> None:
        upload = super().upload_dir
        await self._bounded_transfer(
            lambda: upload(source_dir, target_dir),
            operation="upload_dir",
            source=str(source_dir),
            target=target_dir,
            attempts=2,
        )

    @override
    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        download = super().service_download_file
        await self._bounded_transfer(
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
        download = super().service_download_dir
        await self._bounded_transfer(
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
        await self._bounded_transfer(
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

    async def _bounded_transfer(
        self,
        transfer: Callable[[], Awaitable[None]],
        *,
        operation: str,
        source: str,
        target: str,
        service: str | None = None,
        attempts: int = 1,
    ) -> None:
        for attempt in range(1, attempts + 1):
            try:
                async with asyncio.timeout(self._transfer_timeout_secs):
                    await transfer()
                return
            except TimeoutError:
                logger.warning(
                    "Harbor Modal transfer timed out after %.1fs "
                    "operation=%s source=%s target=%s service=%s attempt=%d/%d",
                    self._transfer_timeout_secs,
                    operation,
                    source,
                    target,
                    service or "main",
                    attempt,
                    attempts,
                )
                if attempt == attempts:
                    raise
