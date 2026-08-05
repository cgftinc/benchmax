from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from castform_router.litellm_callback import CastformAutoRouter
from castform_router.policy import select_backend
from castform_router.scorer import QwenScorer, request_payload
from castform_router.types import Backend, Prediction, RoutingRequest

BACKENDS = (
    Backend(name="cheap", model="small", provider="local", estimated_cost_usd=0.05),
    Backend(name="strong", model="large", provider="cloud", estimated_cost_usd=0.20),
)


def prediction(backend: str, probability: float) -> Prediction:
    return Prediction(
        backend=backend,
        success_probability=probability,
        input_token_band="under_64k",
        cache_read_token_band="under_64k",
        output_token_band="under_4k",
    )


def test_scorer_payload_excludes_cost() -> None:
    payload = request_payload(RoutingRequest(request_id="1", task="Fix it", backends=BACKENDS))
    assert payload["task"] == "Fix it"
    assert "cost" not in json.dumps(payload)


def test_policy_selects_cheapest_adequate_and_falls_back_to_best() -> None:
    predictions = (
        prediction("cheap", 0.85),
        prediction("strong", 0.95),
    )
    assert select_backend(BACKENDS, predictions, quality_threshold=0.84)[0].name == "cheap"
    assert select_backend(BACKENDS, predictions, quality_threshold=0.99) == (
        BACKENDS[1],
        "highest_quality_fallback",
    )


class FakeScorer:
    version = "fake-qwen"

    def score(self, request: RoutingRequest) -> tuple[Prediction, ...]:
        return (
            prediction("cheap", 0.86),
            prediction("strong", 0.96),
        )


def test_callback_rewrites_public_alias_for_litellm_dispatch(monkeypatch) -> None:
    monkeypatch.setattr("castform_router.litellm_callback.FALLBACK_MODEL", "strong")
    router = CastformAutoRouter(scorer=FakeScorer(), backends=BACKENDS)
    data = asyncio.run(
        router.async_pre_call_hook(
            None,
            None,
            {"model": "castform-auto", "messages": [{"role": "user", "content": "Fix it"}]},
            "completion",
        )
    )
    assert data["model"] == "cheap"
    assert data["metadata"]["castform_router"]["reason"] == "cheapest_adequate"


class Handler(BaseHTTPRequestHandler):
    received: dict = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        type(self).received = json.loads(self.rfile.read(size))
        task = json.loads(self.received["messages"][1]["content"])
        content = json.dumps(
            {
                "scorer_version": "qwen-test",
                "predictions": [
                    {
                        "backend": backend["name"],
                        "success_probability": 0.9,
                        "input_token_band": "under_64k",
                        "cache_read_token_band": "under_64k",
                        "output_token_band": "under_4k",
                    }
                    for backend in task["backends"]
                ],
            }
        )
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_qwen_scorer_uses_strict_schema() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scorer = QwenScorer(
            base_url=f"http://127.0.0.1:{server.server_port}", model="qwen", api_key="test"
        )
        predictions = scorer.score(RoutingRequest(request_id="1", task="Fix it", backends=BACKENDS))
    finally:
        server.shutdown()
        server.server_close()
    assert len(predictions) == 2
    assert Handler.received["response_format"]["json_schema"]["strict"] is True
