from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

__all__ = [
    "CustomSandboxCredentials",
    "DaytonaCredentials",
    "ModalCredentials",
    "SandboxCredentials",
]


@runtime_checkable
class SandboxCredentials(Protocol):
    """Host credentials needed by one Harbor sandbox provider."""

    @property
    def provider(self) -> str | None: ...

    def host_environment(self) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class ModalCredentials:
    """Explicit Modal credentials, translated only while Harbor is running."""

    provider: str = field(default="modal", init=False, repr=False)
    token_id: str = field(repr=False)
    token_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_secret(self.token_id, "Modal token_id")
        _require_secret(self.token_secret, "Modal token_secret")

    def host_environment(self) -> Mapping[str, str]:
        return {
            "MODAL_TOKEN_ID": self.token_id,
            "MODAL_TOKEN_SECRET": self.token_secret,
        }


@dataclass(frozen=True, slots=True)
class DaytonaCredentials:
    """Explicit Daytona credentials, translated only while Harbor is running.

    Daytona accepts either an API key or a JWT plus organization ID. ``target``
    selects a named Daytona target without relying on the host shell.
    """

    provider: str = field(default="daytona", init=False, repr=False)
    api_key: str | None = field(default=None, repr=False)
    jwt_token: str | None = field(default=None, repr=False)
    organization_id: str | None = field(default=None, repr=False)
    target: str | None = None

    def __post_init__(self) -> None:
        has_api_key = self.api_key is not None
        has_jwt = self.jwt_token is not None or self.organization_id is not None
        if has_api_key == has_jwt:
            raise ValueError(
                "Daytona credentials require either api_key or both jwt_token "
                "and organization_id"
            )
        if has_api_key:
            _require_secret(self.api_key, "Daytona api_key")
        else:
            _require_secret(self.jwt_token, "Daytona jwt_token")
            _require_secret(self.organization_id, "Daytona organization_id")
        if self.target is not None and not self.target:
            raise ValueError("Daytona target must be non-empty when provided")

    def host_environment(self) -> Mapping[str, str]:
        if self.api_key is not None:
            environment = {"DAYTONA_API_KEY": self.api_key}
        else:
            environment = {
                "DAYTONA_JWT_TOKEN": cast(str, self.jwt_token),
                "DAYTONA_ORGANIZATION_ID": cast(str, self.organization_id),
            }
        if self.target is not None:
            environment["DAYTONA_TARGET"] = self.target
        return environment


@dataclass(frozen=True, slots=True)
class CustomSandboxCredentials:
    """Credential bridge for a custom Harbor sandbox implementation."""

    values: Mapping[str, str] = field(repr=False)
    provider: str | None = None

    def __post_init__(self) -> None:
        if self.provider is not None and not self.provider.strip():
            raise ValueError("custom sandbox credential provider must be non-empty")
        object.__setattr__(self, "values", _validated_environment(self.values))

    def host_environment(self) -> Mapping[str, str]:
        return dict(self.values)


_PROCESS_ENVIRONMENT_CHANGED = asyncio.Condition()
_ACTIVE_CREDENTIALS: ContextVar[int | None] = ContextVar(
    "benchmax_active_sandbox_credentials",
    default=None,
)
_MISSING = object()
_active_environment: dict[str, str] | None = None
_active_environment_users = 0
_previous_environment: dict[str, str | object] = {}


@asynccontextmanager
async def sandbox_credentials_scope(
    credentials: SandboxCredentials | None,
) -> AsyncIterator[None]:
    """Expose credentials to provider SDKs without leaking across rollout groups."""

    if credentials is None:
        yield
        return

    active_credentials = _ACTIVE_CREDENTIALS.get()
    credential_id = id(credentials)
    if active_credentials == credential_id:
        yield
        return
    if active_credentials is not None:
        raise RuntimeError("cannot nest different sandbox credential scopes")

    environment = _validated_environment(credentials.host_environment())
    await _enter_process_environment(environment)
    token = _ACTIVE_CREDENTIALS.set(credential_id)
    try:
        yield
    finally:
        _ACTIVE_CREDENTIALS.reset(token)
        await _leave_process_environment()


async def _enter_process_environment(environment: dict[str, str]) -> None:
    """Share one credential environment; wait only when another one conflicts."""

    global _active_environment, _active_environment_users, _previous_environment

    async with _PROCESS_ENVIRONMENT_CHANGED:
        await _PROCESS_ENVIRONMENT_CHANGED.wait_for(
            lambda: _active_environment is None or _active_environment == environment
        )
        if _active_environment is None:
            _previous_environment = {
                key: os.environ.get(key, _MISSING) for key in environment
            }
            os.environ.update(environment)
            _active_environment = environment
        _active_environment_users += 1


async def _leave_process_environment() -> None:
    """Restore the host environment after its final concurrent user exits."""

    global _active_environment, _active_environment_users, _previous_environment

    async with _PROCESS_ENVIRONMENT_CHANGED:
        _active_environment_users -= 1
        if _active_environment_users:
            return

        for key, value in _previous_environment.items():
            if value is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = cast(str, value)
        _active_environment = None
        _previous_environment = {}
        _PROCESS_ENVIRONMENT_CHANGED.notify_all()


def _require_secret(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validated_environment(values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("sandbox credential environment must be a mapping")
    environment: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sandbox credential names must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise ValueError(f"sandbox credential {key!r} must be a non-empty string")
        environment[key] = value
    if not environment:
        raise ValueError("sandbox credentials must contain at least one value")
    return environment
