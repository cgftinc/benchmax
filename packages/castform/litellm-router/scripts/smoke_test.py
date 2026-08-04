"""Exercise Qwen-backed routing through all three harness API protocols."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from castform_router.job_router import ROUTES
from castform_router.router_protocol import OpenAICompatibleRouteScorer
from castform_router.types import HarnessRouteRequest

BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
TRACE_BASE_URL = os.getenv("CASTFORM_BASE_URL", "http://localhost:3000")
API_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-local-dev")
ROUTE_ALIASES = {"claude-route", "glm-route", "codex-route"}


def request_json(
    *,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60,
) -> tuple[dict[str, Any], dict[str, str]]:
    request_headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode(),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read()), {
            key.lower(): value for key, value in response.headers.items()
        }


def trace(trace_id: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(
        f"{TRACE_BASE_URL}/api/traces/{trace_id}",
        timeout=10,
    ) as response:
        payload = json.load(response)
    return payload["events"]


def selected_alias(headers: dict[str, str]) -> str:
    alias = headers.get("x-litellm-model-group")
    assert alias in ROUTE_ALIASES, headers
    return alias


def main() -> None:
    scorer = OpenAICompatibleRouteScorer(
        base_url=BASE_URL,
        model="castform-router-0.8b",
        api_key=API_KEY,
    )
    predictions = scorer.score(
        HarnessRouteRequest(
            request_id="smoke-router-shape",
            session_id=None,
            task_text="Fix a parser regression.",
            task_domain="software_engineering",
            user_context={},
            workspace_context={"repository": "pallets/click"},
            candidate_routes=ROUTES,
        )
    )
    assert len(predictions) == len(ROUTES)

    chat_trace = "smoke-open-chat"
    chat, chat_headers = request_json(
        path="/v1/chat/completions",
        body={
            "model": "castform-auto-open",
            "messages": [{"role": "user", "content": "Rename this variable."}],
            "metadata": {
                "session_id": "smoke-open-session",
                "trace_id": chat_trace,
            },
        },
    )
    assert chat["choices"][0]["message"]["content"]
    first_alias = selected_alias(chat_headers)

    _, pinned_headers = request_json(
        path="/v1/chat/completions",
        body={
            "model": "castform-auto-open",
            "messages": [
                {"role": "user", "content": "Debug a distributed deadlock."}
            ],
            "metadata": {
                "session_id": "smoke-open-session",
                "trace_id": "smoke-open-pinned",
            },
        },
    )
    assert selected_alias(pinned_headers) == first_alias
    assert any(
        event["stage"] == "session.pin_reused"
        for event in trace("smoke-open-pinned")
    )

    responses, response_headers = request_json(
        path="/v1/responses",
        body={
            "model": "castform-auto-codex",
            "input": "Explain the repository layout.",
            "metadata": {
                "session_id": "smoke-codex-session",
                "trace_id": "smoke-codex-responses",
            },
        },
    )
    assert responses["status"] == "completed", responses
    selected_alias(response_headers)

    messages, message_headers = request_json(
        path="/v1/messages",
        headers={
            "anthropic-version": "2023-06-01",
            "x-castform-session-id": "smoke-claude-session",
            "x-castform-trace-id": "smoke-claude-messages",
        },
        body={
            "model": "castform-auto-claude",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Find the parser."}],
        },
    )
    assert messages["type"] == "message", messages
    assert messages["content"][0]["text"]
    selected_alias(message_headers)

    explicit, explicit_headers = request_json(
        path="/v1/chat/completions",
        body={
            "model": "codex-route",
            "messages": [{"role": "user", "content": "Explicit bypass."}],
        },
    )
    assert "Mock response" in explicit["choices"][0]["message"]["content"]
    assert explicit_headers.get("x-litellm-model-group") == "codex-route"

    print("PASS: Qwen JSON contract and in-harness model routing")
    print(f"  OpenAI Chat Completions → {first_alias}")
    print(f"  OpenAI Responses → {selected_alias(response_headers)}")
    print(f"  Anthropic Messages → {selected_alias(message_headers)}")
    print("  Session pinning and explicit bypass also passed")


if __name__ == "__main__":
    main()
