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
