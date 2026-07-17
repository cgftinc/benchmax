from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.environments.modal import ModalEnvironment

from benchmax.envs.harbor.modal import BoundedModalEnvironment


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "download"])
async def test_modal_filesystem_transfers_have_a_deadline(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer_cancelled = asyncio.Event()
    attempts = 0

    async def hang() -> None:
        nonlocal attempts
        attempts += 1
        try:
            await asyncio.Event().wait()
        finally:
            transfer_cancelled.set()

    async def hanging_upload(
        self: ModalEnvironment,
        source_dir: Path | str,
        target_dir: str,
    ) -> None:
        await hang()

    async def hanging_download(
        self: ModalEnvironment,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        await hang()

    environment = object.__new__(BoundedModalEnvironment)
    environment._transfer_timeout_secs = 0.01

    with pytest.raises(TimeoutError):
        if operation == "upload":
            monkeypatch.setattr(ModalEnvironment, "upload_dir", hanging_upload)
            await environment.upload_dir(tmp_path, "/app")
        else:
            monkeypatch.setattr(
                ModalEnvironment,
                "service_download_dir",
                hanging_download,
            )
            await environment.service_download_dir(
                "/logs/artifacts",
                tmp_path,
                service=None,
            )

    assert transfer_cancelled.is_set()
    assert attempts == (2 if operation == "upload" else 1)
