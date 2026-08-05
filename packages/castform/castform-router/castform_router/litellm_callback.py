"""LiteLLM hook implementing task -> Qwen -> policy -> selected backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any

try:
    from litellm.integrations.custom_logger import CustomLogger
except ModuleNotFoundError:  # Unit tests do not need LiteLLM installed.

    class CustomLogger:
        pass


from castform_router.policy import select_backend
from castform_router.scorer import QwenScorer
from castform_router.types import Backend, RoutingRequest

logger = logging.getLogger("castform_router")
PUBLIC_MODEL = os.getenv("CASTFORM_AUTO_MODEL", "castform-auto")
QUALITY_THRESHOLD = float(os.getenv("CASTFORM_ROUTER_QUALITY_THRESHOLD", "0.84"))
FALLBACK_MODEL = os.getenv("CASTFORM_AUTO_FALLBACK_MODEL", "large-route")


def load_backends(raw: str | None) -> tuple[Backend, ...]:
    value = (
        json.loads(raw)
        if raw and raw.strip()
        else [
            {
                "name": "small-route",
                "model": "small",
                "provider": "local",
                "estimated_cost_usd": 0.05,
            },
            {
                "name": "medium-route",
                "model": "medium",
                "provider": "cloud",
                "estimated_cost_usd": 0.10,
            },
            {
                "name": "large-route",
                "model": "large",
                "provider": "cloud",
                "estimated_cost_usd": 0.30,
            },
        ]
    )
    if not isinstance(value, list) or not value:
        raise ValueError("CASTFORM_AUTO_BACKENDS_JSON must be a non-empty array")
    try:
        backends = tuple(
            Backend(
                name=str(item["name"]),
                model=str(item["model"]),
                provider=str(item["provider"]),
                estimated_cost_usd=float(item["estimated_cost_usd"]),
            )
            for item in value
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid CASTFORM_AUTO_BACKENDS_JSON") from error
    if any(not backend.name or backend.estimated_cost_usd < 0 for backend in backends) or len(
        {b.name for b in backends}
    ) != len(backends):
        raise ValueError("backend names must be unique and costs non-negative")
    return backends


def task_text(data: dict[str, Any]) -> str:
    messages = data.get("messages")
    if isinstance(messages, list):
        parts = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        text = [part for part in parts if isinstance(part, str)]
        if text:
            return "\n".join(text)
    value = data.get("input")
    return value if isinstance(value, str) else json.dumps(value) if value is not None else ""


class CastformAutoRouter(CustomLogger):
    def __init__(
        self, *, scorer: Any | None = None, backends: tuple[Backend, ...] | None = None
    ) -> None:
        super().__init__()
        self.backends = backends or load_backends(os.getenv("CASTFORM_AUTO_BACKENDS_JSON"))
        self.scorer = scorer or QwenScorer(
            base_url=os.getenv("CASTFORM_ROUTER_MODEL_BASE_URL", "http://localhost:4000"),
            model=os.getenv("CASTFORM_ROUTER_MODEL_NAME", "castform-router-qwen"),
            api_key=os.getenv(
                "CASTFORM_ROUTER_MODEL_API_KEY", os.getenv("LITELLM_MASTER_KEY", "sk-local-dev")
            ),
            timeout=float(os.getenv("CASTFORM_ROUTER_MODEL_TIMEOUT_SECONDS", "60")),
        )
        if FALLBACK_MODEL not in {backend.name for backend in self.backends}:
            raise ValueError("CASTFORM_AUTO_FALLBACK_MODEL must name a configured backend")

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict[str, Any], call_type: Any
    ) -> dict[str, Any]:
        del user_api_key_dict, cache, call_type
        if data.get("model") != PUBLIC_MODEL:
            return data
        request = RoutingRequest(
            request_id=str(data.get("request_id") or uuid.uuid4().hex),
            task=task_text(data),
            backends=self.backends,
        )
        try:
            predictions = await asyncio.to_thread(self.scorer.score, request)
            backend, reason = select_backend(
                self.backends, predictions, quality_threshold=QUALITY_THRESHOLD
            )
            metadata = {
                "selected_backend": backend.name,
                "reason": reason,
                "scorer_version": self.scorer.version,
                "quality_threshold": QUALITY_THRESHOLD,
                "predictions": [asdict(item) for item in predictions],
            }
        except Exception as error:
            logger.exception("Qwen scoring failed; using %s", FALLBACK_MODEL)
            backend = next(item for item in self.backends if item.name == FALLBACK_MODEL)
            metadata = {
                "selected_backend": backend.name,
                "reason": "scorer_error_fallback",
                "error_type": type(error).__name__,
            }
        data["model"] = backend.name
        original_metadata = data.get("metadata")
        data["metadata"] = {
            **(original_metadata if isinstance(original_metadata, dict) else {}),
            "castform_router": metadata,
        }
        return data


castform_auto_router = CastformAutoRouter()
