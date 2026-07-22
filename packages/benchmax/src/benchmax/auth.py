"""Explicit, call-time authentication for model requests.

BenchMax defines only the runtime contract. Platform packages and execution
runtimes provide concrete credential sources and bind injected credentials.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "InjectedAuth",
    "ModelAuth",
    "ModelRequestContext",
    "StaticBearerAuth",
    "bind_model_auth",
]


@dataclass(frozen=True, slots=True)
class ModelRequestContext:
    """Identity of the model request about to be authorized."""

    base_url: str
    model: str
    rollout_id: str


@runtime_checkable
class ModelAuth(Protocol):
    """Return headers immediately before each model HTTP request."""

    async def headers_for_request(
        self,
        context: ModelRequestContext,
    ) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class StaticBearerAuth:
    """Explicit bearer authentication for providers with a stable API key."""

    token: str

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("bearer token must be a non-empty string")

    async def headers_for_request(
        self,
        context: ModelRequestContext,
    ) -> Mapping[str, str]:
        del context
        return {"Authorization": f"Bearer {self.token}"}


_BOUND_MODEL_AUTH: ContextVar[Mapping[str, ModelAuth] | None] = ContextVar(
    "benchmax_bound_model_auth",
    default=None,
)


@dataclass(frozen=True, slots=True)
class InjectedAuth:
    """Serializable reference to authentication supplied by the runtime."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("injected auth name must be a non-empty string")

    async def headers_for_request(
        self,
        context: ModelRequestContext,
    ) -> Mapping[str, str]:
        providers = _BOUND_MODEL_AUTH.get()
        provider = providers.get(self.name) if providers is not None else None
        if provider is None:
            raise RuntimeError(
                f"No runtime model-auth provider was injected for {self.name!r}."
            )
        if isinstance(provider, InjectedAuth):
            raise RuntimeError(
                f"Injected model-auth provider {self.name!r} cannot reference "
                "another InjectedAuth."
            )
        return await provider.headers_for_request(context)


@contextmanager
def bind_model_auth(providers: Mapping[str, ModelAuth]) -> Iterator[None]:
    """Bind runtime providers for the current async execution context."""

    normalized = dict(providers)
    for name, provider in normalized.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model-auth provider names must be non-empty strings")
        if not isinstance(provider, ModelAuth):
            raise TypeError(f"model-auth provider {name!r} does not implement ModelAuth")
    token = _BOUND_MODEL_AUTH.set(normalized)
    try:
        yield
    finally:
        _BOUND_MODEL_AUTH.reset(token)
