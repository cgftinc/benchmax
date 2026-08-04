"""Qwen scorer served through an OpenAI-compatible LiteLLM endpoint."""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from castform_router.types import Prediction, RoutingRequest

SYSTEM_PROMPT = (
    "Score every candidate backend for its probability of successfully completing the task. "
    "Score each backend exactly once. Do not select a backend or use cost. "
    "Return only JSON matching the supplied schema."
)


class Scorer(Protocol):
    version: str

    def score(self, request: RoutingRequest) -> tuple[Prediction, ...]: ...


def request_payload(request: RoutingRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "task": request.task,
        "backends": [
            {"name": b.name, "model": b.model, "provider": b.provider} for b in request.backends
        ],
    }


def response_schema(names: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scorer_version": {"type": "string", "minLength": 1},
            "predictions": {
                "type": "array",
                "minItems": len(names),
                "maxItems": len(names),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "backend": {"type": "string", "enum": list(names)},
                        "success_probability": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["backend", "success_probability"],
                },
            },
        },
        "required": ["scorer_version", "predictions"],
    }


def parse_response(value: object, names: tuple[str, ...]) -> tuple[str, tuple[Prediction, ...]]:
    if not isinstance(value, dict) or set(value) != {"scorer_version", "predictions"}:
        raise ValueError("invalid scorer response")
    version, raw_predictions = value["scorer_version"], value["predictions"]
    if not isinstance(version, str) or not version.strip() or not isinstance(raw_predictions, list):
        raise ValueError("invalid scorer response")
    predictions = []
    for raw in raw_predictions:
        if not isinstance(raw, dict) or set(raw) != {"backend", "success_probability"}:
            raise ValueError("invalid prediction")
        backend, probability = raw["backend"], raw["success_probability"]
        if (
            backend not in names
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid prediction value")
        predictions.append(Prediction(backend=backend, success_probability=float(probability)))
    if len(predictions) != len(names) or {item.backend for item in predictions} != set(names):
        raise ValueError("scorer must return exactly one prediction per backend")
    return version.strip(), tuple(predictions)


class QwenScorer:
    def __init__(self, *, base_url: str, model: str, api_key: str, timeout: float = 60) -> None:
        self.base_url, self.model, self.api_key, self.timeout, self.version = (
            base_url.rstrip("/"),
            model,
            api_key,
            timeout,
            model,
        )

    def score(self, request: RoutingRequest) -> tuple[Prediction, ...]:
        names = tuple(backend.name for backend in request.backends)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request_payload(request))},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "castform_scores",
                    "strict": True,
                    "schema": response_schema(names),
                },
            },
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            result = json.load(response)
        version, predictions = parse_response(
            json.loads(result["choices"][0]["message"]["content"]), names
        )
        self.version = version
        return predictions
