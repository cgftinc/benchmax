"""benchmax.bundle - Remote class bundling for custom environments.

Usage::

    from benchmax.bundle import bundle_env, load_env, validate_env

    # On the local machine (e.g., Colab notebook):
    payload = bundle_env(
        MySearchEnv,
        pip_dependencies=["aiohttp"],
    )
    payload_bytes = payload.to_bytes()
    # Send payload_bytes to remote machine...

    # On the remote machine:
    env_class = load_env(payload_bytes)
    env = env_class(api_key="...", base_url="...")
"""

from benchmax.bundle.errors import (
    DependencyError,
    IncompatiblePythonError,
    BundlingError,
    ValidationError,
)
from benchmax.bundle.loader import load_env
from benchmax.bundle.bundler import bundle_env
from benchmax.bundle.payload import EnvPayload
from benchmax.bundle.validator import validate_payload

__all__ = [
    "bundle_env",
    "load_env",
    "validate_payload",
    "EnvPayload",
    "BundlingError",
    "ValidationError",
    "DependencyError",
    "IncompatiblePythonError",
]
