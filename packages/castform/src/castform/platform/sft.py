"""Typed SFT dataset upload and launch transport.

Separate from environment-bundle uploads on purpose: SFT has no environment,
so this surface accepts only a fully validated
:class:`benchmax.sft.SftDataset` and never optionalizes bundle fields. The
platform owns the fixed model/LoRA/topology policy; only the genuine v1
choices appear in :class:`SftTrainingConfig`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from benchmax.sft import SFT_DATASET_FORMAT, SftDataset

from castform import config

from .client import StorageClient
from .environment_assets import _validate_blob_path

__all__ = ["SftTrainingConfig", "UploadedSftAssets", "upload_sft_assets"]


@dataclass(frozen=True)
class UploadedSftAssets:
    """Immutable reference to one uploaded SFT dataset prefix.

    ``dataset_path`` is the blob prefix holding ``train.jsonl``;
    ``dataset_format`` is the literal public format identifier; and
    ``content_digest`` is the full SHA-256 hex digest of the canonical JSONL
    bytes. Pass the whole object to ``TrainerClient.launch_sft_run``.
    """

    dataset_path: str
    dataset_format: str
    content_digest: str


@dataclass(frozen=True)
class SftTrainingConfig:
    """The genuine caller choices for a v1 SFT run.

    Everything else — model, LoRA rank/alpha, batch size, GPU topology,
    optimizer, checkpoint layout, stop policy — is platform-owned and not
    accepted here. Ranges mirror the platform's public argument table so bad
    values fail before any request; the server remains authoritative.
    """

    num_epochs: int = 1
    learning_rate: float = 1e-5
    max_context_tokens: int = 8192
    save_interval: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        self._require_int("num_epochs", self.num_epochs, 1, 100)
        rate = self.learning_rate
        if isinstance(rate, bool) or not isinstance(rate, int | float):
            raise ValueError("learning_rate must be a number")
        if not math.isfinite(rate) or rate <= 0 or rate > 0.1:
            raise ValueError("learning_rate must be finite, greater than 0, and at most 0.1")
        self._require_int("max_context_tokens", self.max_context_tokens, 256, 8192)
        self._require_int("save_interval", self.save_interval, 1, 10_000)
        self._require_int("seed", self.seed, 0, 2_147_483_647)

    @staticmethod
    def _require_int(name: str, value: object, minimum: int, maximum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    def as_args(self) -> dict[str, int | float]:
        """The resolved public ``args`` object for the SFT launch request."""

        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "max_context_tokens": self.max_context_tokens,
            "save_interval": self.save_interval,
            "seed": self.seed,
        }


def upload_sft_assets(
    *,
    dataset: SftDataset,
    run_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    storage_client: StorageClient | None = None,
) -> UploadedSftAssets:
    """Upload a validated SFT dataset's canonical bytes as ``train.jsonl``.

    The blob layout is content-addressed like RL datasets:
    ``datasets/<run_name>/<digest16>/train.jsonl`` where ``digest16`` is the
    first 16 hex chars of the canonical bytes' SHA-256. Auth, retry, and
    path-safety ride the same :class:`StorageClient` mechanism as RL uploads.

    Args:
        dataset: A fully constructed :class:`benchmax.sft.SftDataset`. Invalid
            data cannot reach here — construction already failed loudly.
        run_name: Asset namespace; must satisfy the platform blob-path charset.
        api_key: Platform API key; omitted means the per-request credential
            seam resolves the bearer.
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        storage_client: BYOC for connection reuse or test fakes.

    Returns:
        UploadedSftAssets carrying the prefix, literal format, and digest.
    """

    if not isinstance(dataset, SftDataset):
        raise TypeError(
            f"dataset must be a benchmax.sft.SftDataset, got {type(dataset).__name__}; "
            "construct one with SftDataset.from_jsonl(...) or SftDataset.from_rows(...)"
        )
    payload = dataset.to_jsonl_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    prefix = f"datasets/{run_name}/{digest[:16]}"
    _validate_blob_path(prefix, source="run_name")

    if storage_client is None:
        storage_client = StorageClient(
            api_key=api_key,
            base_url=base_url or config.platform_url(),
        )
    storage_client.upload_file(f"{prefix}/train.jsonl", payload, "application/jsonl")
    return UploadedSftAssets(
        dataset_path=prefix,
        dataset_format=SFT_DATASET_FORMAT,
        content_digest=digest,
    )
