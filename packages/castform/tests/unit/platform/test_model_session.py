from __future__ import annotations

import json

import httpx
from benchmax.auth import StaticBearerAuth
from castform.platform.model_session import ModelSession, ModelSessionClient


async def test_model_session_lifecycle_uses_platform_then_session_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "PUT":
            return httpx.Response(
                200,
                json={
                    "session_key": "session-secret",
                    "base_url": "/v1/sessions/validate-1",
                },
            )
        if request.url.path.endswith("/collect"):
            return httpx.Response(200, json={"num_calls": 1, "truncated": False})
        return httpx.Response(200, json={"deleted": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http:
        client = ModelSessionClient(
            base_url="https://llm.castform.dev/v1",
            model_auth=StaticBearerAuth("platform-token"),
            http_client=http,
        )
        session = await client.create(
            session_id="validate-1",
            model="gpt-test",
            max_context_tokens=2048,
            ttl_seconds=600,
        )
        capture = await client.collect(session)
        await client.discard(
            ModelSession(
                session_id="validate-2",
                base_url="https://llm.castform.dev/v1/sessions/validate-2",
                session_key="other-session-secret",
            )
        )
        await client.aclose()

    assert session.base_url == "https://llm.castform.dev/v1/sessions/validate-1"
    assert session.session_key == "session-secret"
    assert capture == {"num_calls": 1, "truncated": False}

    create = requests[0]
    assert create.method == "PUT"
    assert create.url.path == "/v1/sessions/validate-1"
    assert create.headers["authorization"] == "Bearer platform-token"
    assert json.loads(create.content) == {
        "model": "gpt-test",
        "max_context_tokens": 2048,
        "ttl_seconds": 600,
    }

    collect, discard = requests[1:]
    assert collect.method == "POST"
    assert collect.url.path == "/v1/sessions/validate-1/collect"
    assert collect.headers["authorization"] == "Bearer session-secret"
    assert discard.method == "DELETE"
    assert discard.url.path == "/v1/sessions/validate-2"
    assert discard.headers["authorization"] == "Bearer other-session-secret"
