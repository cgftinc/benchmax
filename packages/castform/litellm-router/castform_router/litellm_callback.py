"""LiteLLM hook for Qwen-backed routing inside an existing coding harness."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from castform_router.gateway_routing import QwenGatewayRouter, load_gateway_candidates
from castform_router.router_protocol import OpenAICompatibleRouteScorer
from castform_router.trace import append_trace, valid_trace_id

logger = logging.getLogger("castform_router")

PUBLIC_MODEL_HARNESSES = {
    os.getenv("CASTFORM_AUTO_MODEL", "castform-auto"): "openai-compatible",
    os.getenv("CASTFORM_CODEX_MODEL", "castform-auto-codex"): "codex",
    os.getenv("CASTFORM_CLAUDE_MODEL", "castform-auto-claude"): "claude-code",
    os.getenv("CASTFORM_OPEN_MODEL", "castform-auto-open"): "openai-compatible",
}
ROUTER_MODEL_BASE_URL = os.getenv(
    "CASTFORM_ROUTER_MODEL_BASE_URL",
    "http://localhost:4000",
)
ROUTER_MODEL_NAME = os.getenv(
    "CASTFORM_ROUTER_MODEL_NAME",
    "castform-router-0.8b",
)
ROUTER_MODEL_API_KEY = os.getenv(
    "CASTFORM_ROUTER_MODEL_API_KEY",
    os.getenv("LITELLM_MASTER_KEY", "sk-local-dev"),
)
ROUTER_TIMEOUT_SECONDS = float(
    os.getenv("CASTFORM_ROUTER_MODEL_TIMEOUT_SECONDS", "60")
)
QUALITY_THRESHOLD = float(os.getenv("CASTFORM_ROUTER_QUALITY_THRESHOLD", "0.84"))
SESSION_TTL_SECONDS = int(os.getenv("CASTFORM_ROUTER_SESSION_TTL_SECONDS", "3600"))
FALLBACK_MODEL = os.getenv("CASTFORM_AUTO_FALLBACK_MODEL", "claude-route")


def _task_text(data: dict[str, Any]) -> str:
    messages = data.get("messages")
    if isinstance(messages, list):
        user_content: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                user_content.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        user_content.append(part["text"])
        if user_content:
            return "\n".join(user_content)

    input_value = data.get("input")
    if isinstance(input_value, str):
        return input_value
    if input_value is not None:
        try:
            return json.dumps(input_value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(input_value)
    return ""


def _metadata(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("metadata")
    return dict(value) if isinstance(value, dict) else {}


def _headers(data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str]:
    containers = (
        metadata.get("headers"),
        data.get("headers"),
        data.get("proxy_server_request", {}).get("headers")
        if isinstance(data.get("proxy_server_request"), dict)
        else None,
    )
    merged: dict[str, str] = {}
    for container in containers:
        if isinstance(container, dict):
            merged.update(
                {
                    str(key).lower(): str(value)
                    for key, value in container.items()
                    if value is not None
                }
            )
    return merged


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _trace_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "castform_route_override",
        "request_id",
        "session_id",
        "trace_id",
        "user_context",
        "workspace_context",
    }
    return {key: metadata[key] for key in allowed if key in metadata}


def _add_trace_header(data: dict[str, Any], trace_id: str) -> None:
    headers = data.get("extra_headers")
    if not isinstance(headers, dict):
        headers = {}
    headers["x-castform-trace-id"] = trace_id
    data["extra_headers"] = headers


class CastformAutoRouter(CustomLogger):
    """Replace a stable public alias with the deployment selected by Qwen."""

    def __init__(self) -> None:
        super().__init__()
        candidates = load_gateway_candidates(os.getenv("CASTFORM_AUTO_ROUTES_JSON"))
        if FALLBACK_MODEL not in {
            candidate.gateway_model for candidate in candidates
        }:
            raise ValueError(
                "CASTFORM_AUTO_FALLBACK_MODEL must name a configured gateway_model"
            )
        self._router = QwenGatewayRouter(
            scorer=OpenAICompatibleRouteScorer(
                base_url=ROUTER_MODEL_BASE_URL,
                model=ROUTER_MODEL_NAME,
                api_key=ROUTER_MODEL_API_KEY,
                timeout_seconds=ROUTER_TIMEOUT_SECONDS,
            ),
            candidates=candidates,
            quality_threshold=QUALITY_THRESHOLD,
            ttl_seconds=SESSION_TTL_SECONDS,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Any,
    ) -> dict[str, Any]:
        del user_api_key_dict, cache
        public_model = str(data.get("model") or "")
        metadata = _metadata(data)
        request_headers = _headers(data, metadata)
        supplied_trace_id = (
            _string(metadata.get("trace_id"))
            or _string(request_headers.get("x-castform-trace-id"))
        )
        trace_id = valid_trace_id(supplied_trace_id) or f"gateway-{uuid.uuid4().hex}"

        if public_model not in PUBLIC_MODEL_HARNESSES:
            _add_trace_header(data, trace_id)
            scorer_request = metadata.get("castform_stage") == "route_scoring"
            append_trace(
                trace_id,
                actor="LiteLLM",
                stage=(
                    "litellm.scorer_request_received"
                    if scorer_request
                    else "litellm.route_received"
                ),
                summary=(
                    "Received the route-scoring request and dispatched it to Qwen."
                    if scorer_request
                    else "Received an explicit gateway model."
                ),
                input={"gateway_model": public_model},
                output={
                    "next": "router_model_dispatch" if scorer_request else "provider_dispatch"
                },
            )
            return data

        normalized_call_type = str(getattr(call_type, "value", call_type)).lower()
        if not any(
            marker in normalized_call_type
            for marker in ("completion", "response", "message")
        ):
            raise ValueError(
                f"{public_model} does not support LiteLLM call type "
                f"{normalized_call_type!r}"
            )

        harness = PUBLIC_MODEL_HARNESSES[public_model]
        session_id = (
            _string(metadata.get("session_id"))
            or _string(request_headers.get("x-castform-session-id"))
            or _string(request_headers.get("x-claude-code-session-id"))
        )
        request_id = _string(metadata.get("request_id")) or trace_id
        route_override = (
            _string(metadata.get("castform_route_override"))
            or _string(request_headers.get("x-castform-route"))
        )
        user_context = metadata.get("user_context")
        workspace_context = metadata.get("workspace_context")
        task_text = _task_text(data)
        metadata["trace_id"] = trace_id

        append_trace(
            trace_id,
            actor=harness,
            stage="harness.model_request_received",
            summary="The existing harness requested the stable Castform model alias.",
            input={
                "public_model": public_model,
                "call_type": normalized_call_type,
                "session_id": session_id,
                "task_text": task_text,
                "metadata": _trace_metadata(metadata),
            },
        )

        try:
            decision = await asyncio.to_thread(
                self._router.route,
                task_text=task_text,
                harness=harness,
                request_id=request_id,
                session_id=session_id,
                user_context=(user_context if isinstance(user_context, dict) else None),
                workspace_context=(
                    workspace_context if isinstance(workspace_context, dict) else None
                ),
                route_override=route_override,
            )
        except Exception as error:
            logger.exception(
                "castform_auto.qwen_router_failed fallback=%s",
                FALLBACK_MODEL,
            )
            selected_model = FALLBACK_MODEL
            route_metadata: dict[str, Any] = {
                "selected_model": selected_model,
                "reason": "router_error_fallback",
                "policy_version": "fallback-v1",
                "cache_hit": False,
                "error_type": type(error).__name__,
            }
            append_trace(
                trace_id,
                actor="Castform router",
                stage="router.fallback",
                summary="Qwen routing failed, so the configured fallback was selected.",
                output=route_metadata,
            )
        else:
            selected_model = decision.selected_route.gateway_model
            predictions = [asdict(prediction) for prediction in decision.predictions]
            route_metadata = {
                "selected_model": selected_model,
                "selected_route": asdict(decision.selected_route),
                "reason": decision.reason,
                "policy_version": decision.policy_version,
                "router_model_version": decision.router_model_version,
                "quality_threshold": decision.quality_threshold,
                "cache_hit": decision.cache_hit,
            }
            if decision.cache_hit:
                append_trace(
                    trace_id,
                    actor="Session router",
                    stage="session.pin_reused",
                    summary="Reused the backend pinned for this harness session.",
                    input={"session_id": session_id, "harness": harness},
                    output=route_metadata,
                )
            else:
                append_trace(
                    trace_id,
                    actor="Qwen router",
                    stage="router.candidates_scored",
                    summary="Qwen scored every backend allowed for this harness.",
                    input={
                        "task_text": task_text,
                        "harness": harness,
                        "candidate_routes": [
                            prediction["route_id"] for prediction in predictions
                        ],
                    },
                    output={
                        "router_model_version": decision.router_model_version,
                        "predictions": predictions,
                    },
                )
                append_trace(
                    trace_id,
                    actor="Castform policy",
                    stage="policy.route_selected",
                    summary="Applied the deterministic cost-quality policy.",
                    input={"predictions": predictions},
                    output=route_metadata,
                )

        data["model"] = selected_model
        metadata["castform_router"] = route_metadata
        data["metadata"] = metadata
        _add_trace_header(data, trace_id)
        append_trace(
            trace_id,
            actor="LiteLLM adapter",
            stage="litellm.model_rewritten",
            summary="Replaced the stable Castform alias with the selected backend alias.",
            input={"model": public_model},
            output={"model": selected_model, "route_metadata": route_metadata},
        )
        logger.info(
            "castform_auto.route harness=%s session_id=%s selected_model=%s reason=%s "
            "cache_hit=%s trace_id=%s",
            harness,
            session_id,
            selected_model,
            route_metadata["reason"],
            route_metadata["cache_hit"],
            trace_id,
        )
        return data


castform_auto_router = CastformAutoRouter()
