"""Bundle construction and secret-boundary inspection for hosted validation."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Any

from benchmax.bundle import Bundle, bundle_digest, dump_bundle, load_bundle

from order_resolution.order_env import OrderResolutionEnv

RUNTIME_DEPENDENCIES = (
    "psycopg[binary,pool]>=3.2.0,<4",
    "sqlalchemy[asyncio]>=2.0.0,<3",
)
CANONICAL_RUNTIME_DEPENDENCIES = (
    "psycopg[binary,pool]<4,>=3.2.0",
    "sqlalchemy[asyncio]<3,>=2.0.0",
)


def build_environment_bundle(runtime_database_url: str) -> Bundle:
    return dump_bundle(
        OrderResolutionEnv,
        constructor_args={
            "runtime_database_url": runtime_database_url,
            "retain_demo_worlds": False,
            "world_ttl_seconds": 3_600,
        },
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )


def inspect_environment_bundle(
    bundle: Bundle,
    *,
    runtime_database_url: str,
    forbidden_secrets: Mapping[str, str],
) -> dict[str, Any]:
    """Verify one locally built trusted bundle without emitting constructor values."""

    metadata_bytes = bundle.metadata.to_json_bytes()
    env_class, constructor_args = load_bundle(bundle, instantiate=False)
    if env_class.__name__ != "OrderResolutionEnv":
        raise RuntimeError("unexpected bundled environment class")
    expected_keys = {
        "runtime_database_url",
        "retain_demo_worlds",
        "world_ttl_seconds",
    }
    if set(constructor_args) != expected_keys:
        raise RuntimeError("unexpected bundled constructor contract")
    bundled_runtime_url = constructor_args["runtime_database_url"]
    if not isinstance(bundled_runtime_url, str) or not hmac.compare_digest(
        bundled_runtime_url, runtime_database_url
    ):
        raise RuntimeError("bundled runtime credential mismatch")
    if runtime_database_url.encode() not in bundle.pickled:
        raise RuntimeError("bundled runtime credential is missing")
    if bundle.metadata.pip_dependencies != CANONICAL_RUNTIME_DEPENDENCIES:
        raise RuntimeError("bundle runtime dependencies do not match the frozen contract")
    for label, secret in forbidden_secrets.items():
        if secret and any(
            secret.encode() in payload for payload in (bundle.pickled, metadata_bytes)
        ):
            raise RuntimeError(f"bundle secret-boundary failure: {label}")
    return {
        "digest": bundle_digest(bundle),
        "class": env_class.__name__,
        "constructor_keys": sorted(constructor_args),
        "pip_dependencies": list(bundle.metadata.pip_dependencies),
        "python_version": bundle.metadata.python_version,
        "benchmax_version": bundle.metadata.benchmax_version,
        "secret_boundary": "ok",
    }


__all__ = [
    "CANONICAL_RUNTIME_DEPENDENCIES",
    "RUNTIME_DEPENDENCIES",
    "build_environment_bundle",
    "inspect_environment_bundle",
]
