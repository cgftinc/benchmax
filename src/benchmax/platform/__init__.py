"""Castform platform clients (storage, training runs, rollout)."""

from .client import RolloutClient, StorageClient, TrainerClient
from .training_run import UploadedTrainingRun, upload_training_run
from .validation import ValidationReport, validate_env

# Imported last: login depends on credentials/device_auth (siblings), so this
# stays cycle-free as long as those are already loaded by the imports above.
from .login import ensure_session

__all__ = [
    "RolloutClient",
    "StorageClient",
    "TrainerClient",
    "UploadedTrainingRun",
    "ValidationReport",
    "ensure_session",
    "upload_training_run",
    "validate_env",
]
