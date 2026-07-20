"""Exceptions for trainer API operations."""


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


class SftLaunchNotSupportedError(JobLaunchError):
    """The platform rejected an SFT launch as an unrecognized launch arg.

    Raised only by ``TrainerClient.launch_sft_run`` when the response looks
    like the platform's generic unknown-launch-arg rejection — a narrow,
    best-effort heuristic (see that method's docstring for the exact match
    and its verification caveat). Subclasses ``JobLaunchError`` so existing
    ``except JobLaunchError`` call sites keep working; the original response
    error is preserved as ``__cause__``.
    """

    pass


class RolloutError(TrainerError):
    """Base for rollout-server errors. Carries HTTP status when available."""

    pass


class RolloutNotFound(RolloutError):
    """Rollout endpoint or referenced resource not found (HTTP 404)."""

    pass


class RolloutServerError(RolloutError):
    """Rollout server returned 5xx — treat as transient/retryable."""

    pass


class RolloutStreamError(RolloutError):
    """Rollout stream ended unexpectedly (no terminal event, timeout, or disconnect)."""

    pass
