"""Castform platform clients (storage, training runs, rollout)."""

from .client import RolloutClient, StorageClient, TrainerClient
from .config import PlatformConfig
from .credentials import platform_bearer

# Imported last: login depends on credentials/device_auth (siblings), so this
# stays cycle-free as long as those are already loaded by the imports above.
from .login import ensure_session
from .training_run import UploadedTrainingRun, upload_training_run

__all__ = [
    "RolloutClient",
    "PlatformConfig",
    "StorageClient",
    "TrainerClient",
    "UploadedTrainingRun",
    # The seam token-getter: generated scripts pass it to a raw OpenAI client
    # (e.g. the traces pivot), so it's part of the public surface alongside
    # ensure_session — not just an internal credentials helper.
    "platform_bearer",
    "ensure_session",
    "upload_training_run",
]
