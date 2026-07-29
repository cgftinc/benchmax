from __future__ import annotations

import httpx
import pytest
from benchmax.auth import (
    InjectedAuth,
    ModelRequestContext,
    RequestModelAuth,
    StaticBearerAuth,
    bind_model_auth,
)

_CONTEXT = ModelRequestContext(
    base_url="https://model.example/v1",
    model="test-model",
    rollout_id="rollout-1",
)


async def test_injected_auth_fails_without_a_runtime_provider() -> None:
    with pytest.raises(RuntimeError, match="No runtime model-auth provider"):
        await InjectedAuth("judge").headers_for_request(_CONTEXT)


async def test_injected_auth_resolves_provider_on_every_call() -> None:
    class RecordingAuth:
        def __init__(self) -> None:
            self.calls = 0

        async def headers_for_request(self, context: ModelRequestContext):
            assert context == _CONTEXT
            self.calls += 1
            return {"Authorization": f"Bearer token-{self.calls}"}

    provider = RecordingAuth()
    injected = InjectedAuth("judge")
    with bind_model_auth({"judge": provider}):
        first = await injected.headers_for_request(_CONTEXT)
        second = await injected.headers_for_request(_CONTEXT)

    assert first == {"Authorization": "Bearer token-1"}
    assert second == {"Authorization": "Bearer token-2"}
    assert provider.calls == 2


async def test_static_bearer_auth_is_explicit() -> None:
    headers = await StaticBearerAuth("secret").headers_for_request(_CONTEXT)
    assert headers == {"Authorization": "Bearer secret"}


def test_static_bearer_auth_repr_redacts_token() -> None:
    assert "secret" not in repr(StaticBearerAuth("secret"))


async def test_sync_request_auth_resolves_injected_provider_inside_running_loop() -> None:
    request = httpx.Request("POST", _CONTEXT.base_url)
    with bind_model_auth({"judge": StaticBearerAuth("runtime-secret")}):
        authenticated = list(
            RequestModelAuth(InjectedAuth("judge"), _CONTEXT).sync_auth_flow(request)
        )

    assert authenticated == [request]
    assert request.headers["authorization"] == "Bearer runtime-secret"


async def test_async_request_auth_resolves_injected_provider() -> None:
    request = httpx.Request("POST", _CONTEXT.base_url)
    auth = RequestModelAuth(InjectedAuth("judge"), _CONTEXT)
    with bind_model_auth({"judge": StaticBearerAuth("runtime-secret")}):
        authenticated = [item async for item in auth.async_auth_flow(request)]

    assert authenticated == [request]
    assert request.headers["authorization"] == "Bearer runtime-secret"
