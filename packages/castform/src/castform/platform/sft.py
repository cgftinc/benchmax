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

__all__ = [
    "MAX_EVAL_ROWS",
    "SftTrainingConfig",
    "UploadedSftAssets",
    "sft_assets_digest",
    "upload_sft_assets",
]

_LR_DECAY_STYLES = frozenset({"constant", "cosine"})

# Rank 128 trains but fused QKV expands it past serving MAX_LORA_RANK caps, so
# it stays platform-only; these mirror the server's public set.
_PUBLIC_LORA_RANKS = frozenset({32, 64})

# Megatron needs global_batch_size divisible by the data-parallel width, which
# is 4 on the fixed SFT topology. Mirrors the server; the server is authority.
_DATA_PARALLEL_SIZE = 4

# Eval rows the platform accepts. Mirrored here so an oversized set fails
# before the upload rather than at launch; the server is the authority and
# re-counts the real bytes.
MAX_EVAL_ROWS = 2048


@dataclass(frozen=True)
class UploadedSftAssets:
    """Immutable reference to one uploaded SFT dataset prefix.

    ``dataset_path`` is the blob prefix holding ``train.jsonl``;
    ``dataset_format`` is the literal public format identifier; and
    ``content_digest`` is the full SHA-256 hex digest of the canonical JSONL
    bytes. Pass the whole object to ``TrainerClient.launch_sft_run``.

    The eval fields are set only when ``upload_sft_assets`` received an
    ``eval_dataset``; their presence is what makes the launch send the eval
    marker, so an ``eval.jsonl`` that merely exists at the prefix stays inert.
    """

    dataset_path: str
    dataset_format: str
    content_digest: str
    row_count: int | None = None
    eval_digest: str | None = None
    eval_row_count: int | None = None


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
    # v1.1 knobs. ``None`` means "not sent": the field is omitted from
    # ``as_args()`` entirely, so the platform stamps nothing and the trainer
    # keeps the model config's value. An untouched config therefore still
    # produces exactly the v1 five-key payload.
    lr_decay_style: str | None = None
    min_lr: float | None = None
    warmup_ratio: float | None = None
    adam_beta2: float | None = None
    grad_clip: float | None = None
    lora_rank: int | None = None
    global_batch_size: int | None = None
    # Only meaningful with an eval set: the launch rejects an interval sent
    # without the eval marker rather than silently ignoring it. Unset means the
    # platform derives the cadence from save_interval.
    eval_interval: int | None = None

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

        if self.lr_decay_style is not None and self.lr_decay_style not in _LR_DECAY_STYLES:
            raise ValueError(f"lr_decay_style must be one of {', '.join(sorted(_LR_DECAY_STYLES))}")
        self._require_optional_number("min_lr", self.min_lr, 0.0, None)
        if self.min_lr is not None and self.min_lr >= rate:
            raise ValueError("min_lr must be less than learning_rate")
        self._require_optional_number("warmup_ratio", self.warmup_ratio, 0.0, 0.5)
        self._require_optional_number("adam_beta2", self.adam_beta2, 0.9, 0.999)
        self._require_optional_number("grad_clip", self.grad_clip, None, 10.0, exclusive_minimum=0.0)
        if self.lora_rank is not None and self.lora_rank not in _PUBLIC_LORA_RANKS:
            raise ValueError(
                f"lora_rank must be one of {', '.join(str(r) for r in sorted(_PUBLIC_LORA_RANKS))}"
            )
        if self.global_batch_size is not None:
            self._require_int("global_batch_size", self.global_batch_size, _DATA_PARALLEL_SIZE, 64)
            if self.global_batch_size % _DATA_PARALLEL_SIZE:
                raise ValueError(
                    f"global_batch_size must be a multiple of {_DATA_PARALLEL_SIZE} "
                    "(the data-parallel width of the fixed SFT topology)"
                )

    @staticmethod
    def _require_int(name: str, value: object, minimum: int, maximum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    @staticmethod
    def _require_optional_number(
        name: str,
        value: object,
        minimum: float | None,
        maximum: float | None,
        *,
        exclusive_minimum: float | None = None,
    ) -> None:
        """Range-check a v1.1 knob, leaving ``None`` (not sent) untouched."""

        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            raise ValueError(f"{name} must be greater than {exclusive_minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be at most {maximum}")

    def as_args(self) -> dict[str, int | float | str]:
        """The resolved public ``args`` object for the SFT launch request.

        Unset v1.1 knobs are omitted entirely rather than serialized as null,
        so an untouched config still produces the exact v1 five-key payload.
        """

        args: dict[str, int | float | str] = {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "max_context_tokens": self.max_context_tokens,
            "save_interval": self.save_interval,
            "seed": self.seed,
        }
        for name in (
            "lr_decay_style",
            "min_lr",
            "warmup_ratio",
            "adam_beta2",
            "grad_clip",
            "lora_rank",
            "global_batch_size",
            "eval_interval",
        ):
            value = getattr(self, name)
            if value is not None:
                args[name] = value
        return args


def sft_assets_digest(train_digest: str, eval_digest: str | None = None) -> str:
    """The identity digest for a launch's full asset set.

    With no eval set this is the train digest, which keeps every previously
    uploaded prefix byte-stable. With one it is a domain-separated hash over
    fixed-width raw component digests::

        sha256(b"castform-sft-assets-v1\\0" + b"train\\0" + raw32(train)
               + b"eval\\0" + raw32(eval))

    Tags and fixed-width components make the framing unambiguous; hashing the
    raw payloads concatenated would not be, so a prefix would stop pinning
    which bytes were which. Because the eval digest feeds the prefix, adding
    or changing an eval set produces a NEW prefix — an overwrite at the old
    one cannot swap eval sets under a launched run.

    The platform re-derives this and rejects a prefix that disagrees, so this
    construction must stay byte-identical to
    ``platform-service/src/lib/sft-eval-assets.ts``.
    """
    if eval_digest is None:
        return train_digest
    hasher = hashlib.sha256()
    hasher.update(b"castform-sft-assets-v1\0")
    hasher.update(b"train\0")
    hasher.update(bytes.fromhex(train_digest))
    hasher.update(b"eval\0")
    hasher.update(bytes.fromhex(eval_digest))
    return hasher.hexdigest()


def upload_sft_assets(
    *,
    dataset: SftDataset,
    run_name: str,
    eval_dataset: SftDataset | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    storage_client: StorageClient | None = None,
) -> UploadedSftAssets:
    """Upload a validated SFT dataset, and optionally an eval set beside it.

    The blob layout is content-addressed like RL datasets:
    ``datasets/<run_name>/<digest16>/train.jsonl`` where ``digest16`` is the
    first 16 hex chars of :func:`sft_assets_digest` — the train digest alone
    when there is no eval set, so v1 prefixes are unchanged. An eval set is
    written to ``eval.jsonl`` under the same prefix. Auth, retry, and
    path-safety ride the same :class:`StorageClient` mechanism as RL uploads.

    Args:
        dataset: A fully constructed :class:`benchmax.sft.SftDataset`. Invalid
            data cannot reach here — construction already failed loudly.
        run_name: Asset namespace; must satisfy the platform blob-path charset.
        eval_dataset: Optional held-out set to score during training.
        api_key: Platform API key; omitted means the per-request credential
            seam resolves the bearer.
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        storage_client: BYOC for connection reuse or test fakes.

    Returns:
        UploadedSftAssets carrying the prefix, literal format, both digests,
        and both row counts.

    Raises:
        ValueError: If the eval set exceeds :data:`MAX_EVAL_ROWS`. A local
            mirror of the platform's bound — a convenience, never the
            enforcement point.
    """

    if not isinstance(dataset, SftDataset):
        raise TypeError(
            f"dataset must be a benchmax.sft.SftDataset, got {type(dataset).__name__}; "
            "construct one with SftDataset.from_jsonl(...) or SftDataset.from_rows(...)"
        )
    if eval_dataset is not None and not isinstance(eval_dataset, SftDataset):
        raise TypeError(
            f"eval_dataset must be a benchmax.sft.SftDataset, got {type(eval_dataset).__name__}"
        )

    payload = dataset.to_jsonl_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    eval_payload: bytes | None = None
    eval_digest: str | None = None
    eval_row_count: int | None = None
    if eval_dataset is not None:
        if len(eval_dataset.rows) > MAX_EVAL_ROWS:
            raise ValueError(
                f"eval_dataset has {len(eval_dataset.rows)} rows, above the {MAX_EVAL_ROWS}-row limit"
            )
        eval_payload = eval_dataset.to_jsonl_bytes()
        eval_digest = hashlib.sha256(eval_payload).hexdigest()
        eval_row_count = len(eval_dataset.rows)

    prefix = f"datasets/{run_name}/{sft_assets_digest(digest, eval_digest)[:16]}"
    _validate_blob_path(prefix, source="run_name")

    if storage_client is None:
        storage_client = StorageClient(
            api_key=api_key,
            base_url=base_url or config.platform_url(),
        )
    storage_client.upload_file(f"{prefix}/train.jsonl", payload, "application/jsonl")
    if eval_payload is not None:
        storage_client.upload_file(f"{prefix}/eval.jsonl", eval_payload, "application/jsonl")
    return UploadedSftAssets(
        dataset_path=prefix,
        dataset_format=SFT_DATASET_FORMAT,
        content_digest=digest,
        row_count=len(dataset.rows),
        eval_digest=eval_digest,
        eval_row_count=eval_row_count,
    )
