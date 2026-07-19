"""Utility functions for synthetic data preparation."""

import sys


def ensure_safe_python_version() -> None:
    major, minor = sys.version_info[:2]

    if major == 3 and minor == 13:
        raise RuntimeError(
            "Python 3.13.x has a pathlib.Path pickle incompatibility "
            "that breaks cross-version unpickling. "
            "Please use Python <= 3.12 or >= 3.14."
        )
