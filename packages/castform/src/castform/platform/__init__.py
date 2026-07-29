"""Castform platform clients (storage, training runs, rollout)."""

from .client import RolloutClient, StorageClient, TrainerClient
from .config import PlatformConfig
from .credentials import platform_bearer
from .exceptions import SftDatasetInvalidError
from .login import ensure_session
from .training_run import (
    UploadedSftRun,
    UploadedTrainingRun,
    upload_sft_run,
    upload_training_run,
)

__all__ = [
    "RolloutClient",
    "PlatformConfig",
    "SftDatasetInvalidError",
    "StorageClient",
    "TrainerClient",
    "UploadedSftRun",
    "UploadedTrainingRun",
    # The seam token-getter: generated scripts pass it to a raw OpenAI client
    # (e.g. the traces pivot), so it's part of the public surface alongside
    # ensure_session — not just an internal credentials helper.
    "platform_bearer",
    "ensure_session",
    "upload_sft_run",
    "upload_training_run",
]
