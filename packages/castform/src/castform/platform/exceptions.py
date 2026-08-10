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


class ClientUnsupportedError(TrainerError):
    """Server will never satisfy this request from this client (HTTP 410/501).

    Deliberately NOT a :class:`RolloutServerError`: a 5xx is transient and worth
    a retry, this is not. It means the endpoint was retired server-side and the
    installed client predates that migration, so the only fix is an upgrade.
    Retrying burns the caller's time and hides the real cause.
    """

    pass


class JobLaunchError(TrainerError):
    """Failed to launch a training job."""

    pass


class RolloutError(TrainerError):
    """Base for rollout-server errors. Carries HTTP status when available."""

    pass


class RolloutNotFound(RolloutError):  # noqa: N818 — public exported name
    """Rollout endpoint or referenced resource not found (HTTP 404)."""

    pass


class RolloutServerError(RolloutError):
    """Rollout server returned 5xx — treat as transient/retryable."""

    pass
