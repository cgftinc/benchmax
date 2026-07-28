"""Async client for llm-proxy's ephemeral tracked model sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx
from benchmax.auth import ModelAuth, ModelRequestContext

__all__ = ["ModelSession", "ModelSessionClient"]


@dataclass(frozen=True, slots=True)
class ModelSession:
    """One llm-proxy session and its session-specific bearer."""

    session_id: str
    base_url: str
    session_key: str


class ModelSessionClient:
    """Create, collect, and discard tracked sessions against one llm-proxy."""

    def __init__(
        self,
        *,
        base_url: str,
        model_auth: ModelAuth,
        timeout_seconds: float = 30,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_auth = model_auth
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    async def create(
        self,
        *,
        session_id: str,
        model: str,
        max_context_tokens: int,
        ttl_seconds: int,
    ) -> ModelSession:
        session_url = f"{self._base_url}/sessions/{session_id}"
        headers = await self._model_auth.headers_for_request(
            ModelRequestContext(
                base_url=self._base_url,
                model=model,
                rollout_id=session_id,
            )
        )
        response = await self._http.put(
            session_url,
            json={
                "model": model,
                "max_context_tokens": max_context_tokens,
                "ttl_seconds": ttl_seconds,
            },
            headers=headers,
        )
        _raise_session_error(response, operation="creation")
        payload = _json_object(response, operation="creation")
        session_key = payload.get("session_key")
        returned_base_url = payload.get("base_url")
        if not isinstance(session_key, str) or not session_key:
            raise RuntimeError("llm-proxy session creation omitted session_key")
        if not isinstance(returned_base_url, str) or not returned_base_url:
            raise RuntimeError("llm-proxy session creation omitted base_url")
        return ModelSession(
            session_id=session_id,
            base_url=urljoin(f"{self._base_url}/", returned_base_url),
            session_key=session_key,
        )

    async def collect(self, session: ModelSession) -> dict[str, Any]:
        response = await self._http.post(
            f"{session.base_url}/collect",
            headers=_session_headers(session),
        )
        _raise_session_error(response, operation="collection")
        return _json_object(response, operation="collection")

    async def discard(self, session: ModelSession) -> None:
        response = await self._http.delete(
            session.base_url,
            headers=_session_headers(session),
        )
        _raise_session_error(response, operation="discard")


def _session_headers(session: ModelSession) -> dict[str, str]:
    return {"Authorization": f"Bearer {session.session_key}"}


def _raise_session_error(response: httpx.Response, *, operation: str) -> None:
    if response.is_success:
        return
    detail = response.text[:300]
    raise RuntimeError(
        f"llm-proxy session {operation} failed (HTTP {response.status_code}): {detail}"
    )


def _json_object(response: httpx.Response, *, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"llm-proxy session {operation} returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"llm-proxy session {operation} returned a non-object response"
        )
    return payload
