"""The SFT dataset boundary: load, canonicalize, and validate OpenAI-format
fine-tuning rows (``{"messages": [...]}``, with optional ``tools`` and
per-assistant-message ``weight``).

:func:`load_sft_dataset` is the only JSONL reader for SFT data and
:func:`canonical_jsonl` the only serializer; validation and upload both
consume their output. :func:`canonical_jsonl` refuses a dataset that hasn't
loaded and validated cleanly, so the boundary holds even for a caller that
skips :func:`validate_sft_dataset`.

Platform-agnostic by construction: this package knows the dataset contract
and nothing about how a run is submitted.
"""

from __future__ import annotations

from benchmax.sft.dataset import (
    SftDataset,
    SftIssue,
    SftRow,
    SftSerializationError,
    canonical_jsonl,
    load_sft_dataset,
)
from benchmax.sft.validate import (
    SftConfigError,
    SftValidationReport,
    sft_config_bool,
    sft_validate_kwargs,
    validate_sft_dataset,
)

__all__ = [
    "canonical_jsonl",
    "load_sft_dataset",
    "validate_sft_dataset",
    "sft_validate_kwargs",
    "sft_config_bool",
    "SftConfigError",
    "SftDataset",
    "SftIssue",
    "SftRow",
    "SftSerializationError",
    "SftValidationReport",
]
