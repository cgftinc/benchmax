"""Exceptions for trainer API operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmax.sft import SftValidationReport


class TrainerError(Exception):
    """Base exception for trainer API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(TrainerError):
    """Authentication failed (invalid or missing API key)."""

    pass


class JobLaunchError(TrainerError):
    """Failed to launch a training job."""

    pass


class SftDatasetInvalidError(RuntimeError):
    """:func:`castform.platform.upload_sft_run` refused to upload a dataset
    pair that does not pass :func:`benchmax.sft.validate_sft_dataset`.

    The upload path re-validates defensively rather than trusting that its
    caller already did, and raises this **before** touching storage — so a
    refused run leaves no partial upload behind. ``report`` is the full
    :class:`benchmax.sft.SftValidationReport`, carrying every issue with its
    source path and physical line, so a caller can report exactly what to fix
    without re-running validation.

    Not a :class:`TrainerError`: nothing was sent, so there is no HTTP status
    to carry. A ``RuntimeError`` subclass — matching the data-side
    :class:`benchmax.sft.SftSerializationError` — so a CLI's top-level error
    handling prints it as one clean stderr line instead of a traceback.
    """

    def __init__(self, report: SftValidationReport, message: str) -> None:
        self.report = report
        super().__init__(message)


class RolloutError(TrainerError):
    """Base for rollout-server errors. Carries HTTP status when available."""

    pass


class RolloutNotFound(RolloutError):  # noqa: N818 — public exported name
    """Rollout endpoint or referenced resource not found (HTTP 404)."""

    pass


class RolloutServerError(RolloutError):
    """Rollout server returned 5xx — treat as transient/retryable."""

    pass


class RolloutStreamError(RolloutError):
    """Rollout stream ended unexpectedly (no terminal event, timeout, or disconnect)."""

    pass
