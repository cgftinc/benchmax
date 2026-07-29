"""Castform platform clients (storage, training runs, rollout)."""

from .client import RolloutClient, StorageClient, TrainerClient
from .config import PlatformConfig
from .credentials import platform_bearer
from .environment_assets import UploadedEnvironmentAssets, upload_assets

# Imported last: login depends on credentials/device_auth (siblings), so this
# stays cycle-free as long as those are already loaded by the imports above.
from .login import ensure_session

__all__ = [
    "RolloutClient",
    "PlatformConfig",
    "StorageClient",
    "TrainerClient",
    "UploadedEnvironmentAssets",
    # The seam token-getter: generated scripts pass it to a raw OpenAI client
    # (e.g. the traces pivot), so it's part of the public surface alongside
    # ensure_session — not just an internal credentials helper.
    "platform_bearer",
    "ensure_session",
    "upload_assets",
]
