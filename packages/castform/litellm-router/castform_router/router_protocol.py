"""Versioned learned-router wire contract and OpenAI-compatible scorer."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Protocol

from castform_router.types import (
    HarnessRoutePrediction,
    HarnessRouteRequest,
)

SCHEMA_VERSION = "1"
SYSTEM_PROMPT = (
    "You are Castform Router v1, a pre-solve route scorer for software "
    "engineering tasks. For every candidate route, estimate its probability "
    "of successfully completing the task and its expected input, cache-read, "
    "and output token counts.\n\n"
    "Rules:\n"
    "1. Use only the task, user_context, workspace_context, and candidate "
    "route metadata in the request.\n"
    "2. Score every candidate exactly once and preserve every route_id "
    "verbatim.\n"
    "3. Do not solve the task, select a winner, use prices, or assume provider "
    "availability.\n"
    "4. When evidence is weak, keep probabilities conservative instead of "
    "inventing strong differences.\n"
    "5. Return only compact JSON conforming to the supplied schema, with no "
    "markdown or explanation."
)

_RESPONSE_KEYS = frozenset(
    {"schema_version", "router_model_version", "predictions"}
)
_PREDICTION_KEYS = frozenset(
    {
        "route_id",
        "success_probability",
        "expected_input_tokens",
        "expected_cache_read_tokens",
        "expected_output_tokens",
        "uncertainty",
        "reason_codes",
    }
)


class RouteScorer(Protocol):
    """Replaceable model boundary used by the deterministic policy."""

    router_model_version: str

    def score(
        self,
        request: HarnessRouteRequest,
    ) -> tuple[HarnessRoutePrediction, ...]: ...


def model_request_payload(request: HarnessRouteRequest) -> dict[str, Any]:
    """Build the cost-independent payload visible to the learned model."""

    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request.request_id,
        "task": {
            "text": request.task_text,
            "domain": request.task_domain,
        },
        "user_context": request.user_context,
        "workspace_context": request.workspace_context,
        "candidate_routes": [
            {
                "route_id": route.route_id,
                "harness": route.harness,
                "model": route.model,
                "provider": route.provider,
            }
            for route in request.candidate_routes
        ],
    }


def model_response_payload(
    *,
    router_model_version: str,
    predictions: tuple[HarnessRoutePrediction, ...],
) -> dict[str, Any]:
    """Serialize predictions without policy-owned route selection."""

    return {
        "schema_version": SCHEMA_VERSION,
        "router_model_version": router_model_version,
        "predictions": [
            {
                key: value
                for key, value in asdict(prediction).items()
                if value not in (None, (), [])
            }
            for prediction in predictions
        ],
    }


def model_response_json_schema(
    *,
    expected_route_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Build the strict response schema sent to the model server by LiteLLM."""

    if not expected_route_ids or len(set(expected_route_ids)) != len(
        expected_route_ids
    ):
        raise ValueError("expected_route_ids must be non-empty and unique")
    prediction_count = len(expected_route_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "router_model_version": {"type": "string", "minLength": 1},
            "predictions": {
                "type": "array",
                "minItems": prediction_count,
                "maxItems": prediction_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "route_id": {
                            "type": "string",
                            "enum": list(expected_route_ids),
                        },
                        "success_probability": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "expected_input_tokens": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "expected_cache_read_tokens": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "expected_output_tokens": {
                            "type": "integer",
                            "minimum": 0,
                        },
                    },
                    "required": [
                        "route_id",
                        "success_probability",
                        "expected_input_tokens",
                        "expected_cache_read_tokens",
                        "expected_output_tokens",
                    ],
                },
            },
        },
        "required": [
            "schema_version",
            "router_model_version",
            "predictions",
        ],
    }


def parse_model_response(
    value: object,
    *,
    expected_route_ids: tuple[str, ...],
) -> tuple[str, tuple[HarnessRoutePrediction, ...]]:
    """Validate a model response before it reaches the pricing policy."""

    if not isinstance(value, dict):
        raise ValueError("router response must be a JSON object")
    unexpected_response_keys = set(value) - _RESPONSE_KEYS
    if unexpected_response_keys:
        raise ValueError(
            "router response contains unexpected fields: "
            + ", ".join(sorted(unexpected_response_keys))
        )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"router response schema_version must be {SCHEMA_VERSION!r}")
    version = value.get("router_model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("router_model_version must be a non-empty string")
    raw_predictions = value.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("router predictions must be an array")

    predictions: list[HarnessRoutePrediction] = []
    seen: set[str] = set()
    for raw in raw_predictions:
        if not isinstance(raw, dict):
            raise ValueError("every router prediction must be an object")
        unexpected_prediction_keys = set(raw) - _PREDICTION_KEYS
        if unexpected_prediction_keys:
            raise ValueError(
                "router prediction contains unexpected fields: "
                + ", ".join(sorted(unexpected_prediction_keys))
            )
        route_id = raw.get("route_id")
        if not isinstance(route_id, str) or route_id not in expected_route_ids:
            raise ValueError(f"router returned unknown route_id: {route_id!r}")
        if route_id in seen:
            raise ValueError(f"router returned duplicate route_id: {route_id}")
        seen.add(route_id)
        probability = _bounded_float(
            raw.get("success_probability"),
            "success_probability",
        )
        uncertainty = (
            _bounded_float(raw["uncertainty"], "uncertainty")
            if "uncertainty" in raw
            else None
        )
        reason_codes = raw.get("reason_codes", [])
        if not isinstance(reason_codes, list) or not all(
            isinstance(code, str) for code in reason_codes
        ):
            raise ValueError("reason_codes must be an array of strings")
        predictions.append(
            HarnessRoutePrediction(
                route_id=route_id,
                success_probability=probability,
                expected_input_tokens=_nonnegative_int(
                    raw.get("expected_input_tokens"),
                    "expected_input_tokens",
                ),
                expected_cache_read_tokens=_nonnegative_int(
                    raw.get("expected_cache_read_tokens"),
                    "expected_cache_read_tokens",
                ),
                expected_output_tokens=_nonnegative_int(
                    raw.get("expected_output_tokens"),
                    "expected_output_tokens",
                ),
                uncertainty=uncertainty,
                reason_codes=tuple(reason_codes),
            )
        )

    missing = set(expected_route_ids) - seen
    if missing:
        raise ValueError(
            "router omitted candidate routes: " + ", ".join(sorted(missing))
        )
    return version.strip(), tuple(predictions)


class OpenAICompatibleRouteScorer:
    """Call the LiteLLM 0.8B alias through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.router_model_version = model

    def score(
        self,
        request: HarnessRouteRequest,
    ) -> tuple[HarnessRoutePrediction, ...]:
        payload = model_request_payload(request)
        expected_route_ids = tuple(
            route.route_id for route in request.candidate_routes
        )
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 512,
            "metadata": {
                "trace_id": request.request_id,
                "request_id": request.request_id,
                "castform_stage": "route_scoring",
            },
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "castform_router_response_v1",
                    "strict": True,
                    "schema": model_response_json_schema(
                        expected_route_ids=expected_route_ids,
                    ),
                },
            },
        }
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise ValueError(
                f"router model server returned HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ValueError(f"router model server request failed: {error}") from error

        try:
            content = response_body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("router model server returned invalid JSON content") from error
        version, predictions = parse_model_response(
            parsed,
            expected_route_ids=expected_route_ids,
        )
        self.router_model_version = version
        return predictions


def _bounded_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    normalized = float(value)
    if not 0 <= normalized <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return normalized


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative integer")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return normalized
