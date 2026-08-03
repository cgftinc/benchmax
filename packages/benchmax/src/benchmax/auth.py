"""Explicit, call-time authentication for model requests.

benchmax defines only the runtime contract. Platform packages and execution
runtimes provide concrete credential sources and bind injected credentials.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
from openai import AsyncOpenAI, OpenAI

__all__ = [
    "InjectedAuth",
    "ModelAuth",
    "ModelRequestContext",
    "RequestModelAuth",
    "StaticBearerAuth",
    "bind_model_auth",
    "create_async_openai_client",
    "create_openai_client",
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

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("bearer token must be a non-empty string")

    async def headers_for_request(
        self,
        context: ModelRequestContext,
    ) -> Mapping[str, str]:
        del context
        return {"Authorization": f"Bearer {self.token}"}


class RequestModelAuth(httpx.Auth):
    """Apply a :class:`ModelAuth` immediately before each HTTP request.

    Both sync and async OpenAI-compatible clients use this adapter, so model
    credential selection never falls back to an SDK environment variable or a
    separate token resolver. When a sync client runs inside an active event
    loop, auth is resolved in a context-preserving helper thread; custom
    providers used there must not depend on primitives bound to another loop.
    """

    def __init__(self, auth: ModelAuth, context: ModelRequestContext) -> None:
        if not isinstance(auth, ModelAuth):
            raise TypeError("request auth must implement ModelAuth")
        self._auth = auth
        self._context = context

    def sync_auth_flow(
        self,
        request: httpx.Request,
    ) -> Iterator[httpx.Request]:
        headers = _resolve_headers_sync(self._auth, self._context)
        for name, value in headers.items():
            request.headers[name] = value
        yield request

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ) -> AsyncIterator[httpx.Request]:
        headers = await self._auth.headers_for_request(self._context)
        for name, value in headers.items():
            request.headers[name] = value
        yield request


def create_openai_client(
    *,
    model: str,
    base_url: str,
    auth: ModelAuth,
    request_id: str,
    max_retries: int = 2,
) -> OpenAI:
    """Create a synchronous OpenAI-compatible client with explicit auth."""

    context = ModelRequestContext(
        base_url=base_url,
        model=model,
        rollout_id=request_id,
    )
    return OpenAI(
        base_url=base_url,
        api_key="benchmax-explicit-auth",
        http_client=httpx.Client(auth=RequestModelAuth(auth, context)),
        max_retries=max_retries,
    )


def create_async_openai_client(
    *,
    model: str,
    base_url: str,
    auth: ModelAuth,
    request_id: str,
    max_retries: int = 2,
) -> AsyncOpenAI:
    """Create an asynchronous OpenAI-compatible client with explicit auth."""

    context = ModelRequestContext(
        base_url=base_url,
        model=model,
        rollout_id=request_id,
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key="benchmax-explicit-auth",
        http_client=httpx.AsyncClient(auth=RequestModelAuth(auth, context)),
        max_retries=max_retries,
    )


def _resolve_headers_sync(
    auth: ModelAuth,
    context: ModelRequestContext,
) -> Mapping[str, str]:
    """Resolve async ``ModelAuth`` from a synchronous HTTP client.

    RAG search backends expose synchronous embedding callables. When one is
    invoked from an async environment tool, its event loop is already running;
    resolve the auth coroutine in a context-preserving helper thread rather
    than attempting a nested event loop.
    """

    async def resolve() -> Mapping[str, str]:
        return await auth.headers_for_request(context)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve())

    copied_context = contextvars.copy_context()
    result: list[Mapping[str, str]] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(copied_context.run(lambda: asyncio.run(resolve())))
        except BaseException as error:  # propagate the original auth failure
            failure.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    if not result:
        raise RuntimeError("model authentication did not return headers")
    return result[0]


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
            raise RuntimeError(f"No runtime model-auth provider was injected for {self.name!r}.")
        if isinstance(provider, InjectedAuth):
            raise RuntimeError(
                f"Injected model-auth provider {self.name!r} cannot reference another InjectedAuth."
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
