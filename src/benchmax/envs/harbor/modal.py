from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, override

from harbor.environments.modal import ModalEnvironment

logger = logging.getLogger(__name__)

__all__ = ["BoundedModalEnvironment"]


class BoundedModalEnvironment(ModalEnvironment):
    """Modal environment that cannot stall indefinitely while exporting artifacts."""

    def __init__(
        self,
        *args: Any,
        artifact_transfer_timeout_secs: float = 120,
        **kwargs: Any,
    ) -> None:
        if artifact_transfer_timeout_secs <= 0:
            raise ValueError("artifact_transfer_timeout_secs must be positive")
        self._artifact_transfer_timeout_secs = float(artifact_transfer_timeout_secs)
        super().__init__(*args, **kwargs)

    @override
    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        await self._bounded_artifact_transfer(
            super().service_download_file(
                source_path,
                target_path,
                service=service,
            ),
            source=source_path,
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
        await self._bounded_artifact_transfer(
            super().service_download_dir(
                source_dir,
                target_dir,
                service=service,
            ),
            source=source_dir,
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
        await self._bounded_artifact_transfer(
            super().service_download_dir_with_exclusions(
                source_dir=source_dir,
                target_dir=target_dir,
                exclude=exclude,
                service=service,
            ),
            source=source_dir,
            service=service,
        )

    async def _bounded_artifact_transfer(
        self,
        transfer: Awaitable[None],
        *,
        source: str,
        service: str | None,
    ) -> None:
        try:
            async with asyncio.timeout(self._artifact_transfer_timeout_secs):
                await transfer
        except TimeoutError:
            logger.warning(
                "Harbor Modal artifact transfer timed out after %.1fs "
                "source=%s service=%s",
                self._artifact_transfer_timeout_secs,
                source,
                service or "main",
            )
            raise
