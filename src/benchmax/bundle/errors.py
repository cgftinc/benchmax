class BundlingError(Exception):
    """Base exception for bundling operations."""

    pass


class ValidationError(BundlingError):
    """Raised when pre-bundling validation fails."""

    pass


class DependencyError(BundlingError):
    """Raised when dependency installation fails on the remote side."""

    pass


class IncompatiblePythonError(BundlingError):
    """Raised when Python version constraints are not met."""

    pass


class IncompatibleBenchmaxError(BundlingError):
    """Raised when benchmax version constraints are not met."""

    pass
