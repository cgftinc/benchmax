from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from castform_router.job_router import ROUTES
from castform_router.router_protocol import (
    SYSTEM_PROMPT,
    OpenAICompatibleRouteScorer,
    model_request_payload,
    model_response_json_schema,
    model_response_payload,
    parse_model_response,
)
from castform_router.types import HarnessRoutePrediction, HarnessRouteRequest


def test_learned_request_excludes_live_price_and_gateway_alias() -> None:
    request = HarnessRouteRequest(
        request_id="request-1",
        session_id=None,
        task_text="Fix the parser.",
        task_domain="software_engineering",
        user_context={"role": "developer"},
        workspace_context={"repository": "pallets/click"},
        candidate_routes=ROUTES,
    )

    payload = model_request_payload(request)

    assert payload["task"]["text"] == "Fix the parser."
    assert "estimated_cost_usd" not in str(payload)
    assert "gateway_model" not in str(payload)
    assert len(payload["candidate_routes"]) == len(ROUTES)


def test_system_prompt_preserves_policy_boundary() -> None:
    assert "Score every candidate exactly once" in SYSTEM_PROMPT
    assert "Do not solve the task, select a winner, use prices" in SYSTEM_PROMPT
    assert "Return only compact JSON" in SYSTEM_PROMPT


def test_response_payload_serializes_token_bands() -> None:
    payload = model_response_payload(
        router_model_version="test-v2",
        predictions=(
            HarnessRoutePrediction(
                route_id=ROUTES[0].route_id,
                success_probability=0.7,
                expected_input_tokens=300_000,
                expected_cache_read_tokens=0,
                expected_output_tokens=5_000,
            ),
        ),
    )

    prediction = payload["predictions"][0]
    assert payload["schema_version"] == "2"
    assert prediction["input_token_band"] == "256k_1m"
    assert prediction["cache_read_token_band"] == "zero"
    assert prediction["output_token_band"] == "4k_8k"


def test_router_response_requires_every_candidate() -> None:
    value = {
        "schema_version": "2",
        "router_model_version": "qwen35-08b-sft-v2",
        "predictions": [
            {
                "route_id": route.route_id,
                "success_probability": 0.8,
                "input_token_band": "under_64k",
                "cache_read_token_band": "zero",
                "output_token_band": "under_4k",
            }
            for route in ROUTES
        ],
    }

    version, predictions = parse_model_response(
        value,
        expected_route_ids=tuple(route.route_id for route in ROUTES),
    )

    assert version == "qwen35-08b-sft-v2"
    assert len(predictions) == len(ROUTES)
    assert predictions[0].expected_total_tokens == 34_816


def test_router_response_keeps_legacy_v1_compatibility() -> None:
    route_ids = tuple(route.route_id for route in ROUTES)
    value = {
        "schema_version": "1",
        "router_model_version": "legacy-v1",
        "predictions": [
            {
                "route_id": route_id,
                "success_probability": 0.8,
                "expected_input_tokens": 100,
                "expected_cache_read_tokens": 20,
                "expected_output_tokens": 30,
            }
            for route_id in route_ids
        ],
    }

    version, predictions = parse_model_response(
        value,
        expected_route_ids=route_ids,
    )

    assert version == "legacy-v1"
    assert predictions[0].expected_total_tokens == 150


def test_response_schema_is_strict_and_route_specific() -> None:
    route_ids = tuple(route.route_id for route in ROUTES)

    schema = model_response_json_schema(expected_route_ids=route_ids)

    assert schema["additionalProperties"] is False
    predictions = schema["properties"]["predictions"]
    assert predictions["minItems"] == len(ROUTES)
    assert predictions["maxItems"] == len(ROUTES)
    assert predictions["items"]["properties"]["route_id"]["enum"] == list(
        route_ids
    )
    assert "input_token_band" in predictions["items"]["required"]
    assert "expected_input_tokens" not in predictions["items"]["properties"]


def test_parser_rejects_fields_outside_frozen_contract() -> None:
    route_ids = tuple(route.route_id for route in ROUTES)
    value = {
        "schema_version": "2",
        "router_model_version": "test-router",
        "predictions": [
            {
                "route_id": route_id,
                "success_probability": 0.8,
                "input_token_band": "under_64k",
                "cache_read_token_band": "zero",
                "output_token_band": "under_4k",
            }
            for route_id in route_ids
        ],
        "selected_route": route_ids[0],
    }

    with pytest.raises(ValueError, match="unexpected fields: selected_route"):
        parse_model_response(value, expected_route_ids=route_ids)


class _SchemaCaptureHandler(BaseHTTPRequestHandler):
    received: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        type(self).received = json.loads(self.rfile.read(length))
        request = json.loads(self.received["messages"][1]["content"])
        content = json.dumps(
            {
                "schema_version": "2",
                "router_model_version": "test-router-v2",
                "predictions": [
                    {
                        "route_id": route["route_id"],
                        "success_probability": 0.8,
                        "input_token_band": "under_64k",
                        "cache_read_token_band": "zero",
                        "output_token_band": "under_4k",
                    }
                    for route in request["candidate_routes"]
                ],
            }
        )
        response = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def test_scorer_requests_strict_json_schema() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SchemaCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scorer = OpenAICompatibleRouteScorer(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="castform-router-0.8b",
        )
        scorer.score(
            HarnessRouteRequest(
                request_id="request-schema-test",
                session_id=None,
                task_text="Fix it.",
                task_domain="software_engineering",
                user_context={},
                workspace_context={},
                candidate_routes=ROUTES,
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    response_format = _SchemaCaptureHandler.received["response_format"]
    assert _SchemaCaptureHandler.received["metadata"] == {
        "trace_id": "request-schema-test",
        "request_id": "request-schema-test",
        "castform_stage": "route_scoring",
    }
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert _SchemaCaptureHandler.received["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
