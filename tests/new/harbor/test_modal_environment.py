from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.environments.modal import ModalEnvironment

from benchmax.envs.harbor.modal import BoundedModalEnvironment


@pytest.mark.asyncio
async def test_modal_artifact_directory_transfer_has_a_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer_cancelled = asyncio.Event()

    async def hanging_download(
        self: ModalEnvironment,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            transfer_cancelled.set()

    monkeypatch.setattr(ModalEnvironment, "service_download_dir", hanging_download)
    environment = object.__new__(BoundedModalEnvironment)
    environment._artifact_transfer_timeout_secs = 0.01

    with pytest.raises(TimeoutError):
        await environment.service_download_dir(
            "/logs/artifacts",
            tmp_path,
            service=None,
        )

    assert transfer_cancelled.is_set()
