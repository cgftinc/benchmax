"""The SFT dataset boundary: load, canonicalize, and validate OpenAI-format
fine-tuning rows (``{"messages": [...]}``, with optional ``tools`` and
per-assistant-message ``weight``). :func:`load_sft_dataset` is the only
JSONL reader for SFT data; validation and upload both consume its output.
"""

from __future__ import annotations

from benchmax.sft.dataset import SftDataset, load_sft_dataset
from benchmax.sft.validate import (
    SftValidationReport,
    sft_validate_kwargs,
    validate_sft_dataset,
)

__all__ = [
    "load_sft_dataset",
    "validate_sft_dataset",
    "sft_validate_kwargs",
    "SftDataset",
    "SftValidationReport",
]
