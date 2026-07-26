"""Castform-backed model authentication."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from benchmax.auth import (
    ModelAuth,
    ModelRequestContext,
    RequestModelAuth,
    StaticBearerAuth,
)
from openai import AsyncOpenAI, OpenAI

from castform import config
from castform.platform.credentials import castform_model_bearer

__all__ = [
    "CastformModelAuth",
    "create_async_openai_client",
    "create_openai_client",
    "model_auth_for_endpoint",
]


@dataclass(frozen=True, slots=True)
class CastformModelAuth:
    """Resolve the active Castform bearer immediately before every model call."""

    def _validate_audience(self, base_url: str) -> None:
        expected = urlparse(config.llm_url())
        actual = urlparse(base_url)
        if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
            raise RuntimeError(
                "Refusing to send Castform authentication to a non-Castform "
                f"model endpoint: {base_url}"
            )

    async def headers_for_request(
        self,
        context: ModelRequestContext,
    ) -> dict[str, str]:
        self._validate_audience(context.base_url)
        return {"Authorization": f"Bearer {castform_model_bearer()}"}


def model_auth_for_endpoint(
    *,
    api_key: str,
    base_url: str,
    purpose: str,
) -> ModelAuth:
    """Build model auth without consulting generic platform credentials.

    A non-empty explicit key is always static. With no key, only the configured
    Castform LLM endpoint may use the local login exchange; external endpoints
    must provide their own credential.
    """

    if api_key:
        return StaticBearerAuth(api_key)
    expected = urlparse(config.llm_url())
    actual = urlparse(base_url)
    if (actual.scheme, actual.netloc) == (expected.scheme, expected.netloc):
        return CastformModelAuth()
    raise ValueError(
        f"{purpose} requires an explicit API key because {base_url!r} is not "
        "the configured Castform LLM endpoint."
    )


def create_openai_client(
    *,
    model: str,
    base_url: str,
    auth: ModelAuth,
    request_id: str,
    max_retries: int = 2,
) -> OpenAI:
    """Create a sync OpenAI-compatible client using the ModelAuth seam."""

    context = ModelRequestContext(
        base_url=base_url,
        model=model,
        rollout_id=request_id,
    )
    return OpenAI(
        base_url=base_url,
        api_key="benchmax-explicit-auth",
        http_client=httpx.Client(auth=RequestModelAuth(auth, context)),
        max_retries=max_retries,
    )


def create_async_openai_client(
    *,
    model: str,
    base_url: str,
    auth: ModelAuth,
    request_id: str,
    max_retries: int = 2,
) -> AsyncOpenAI:
    """Create an async OpenAI-compatible client using the ModelAuth seam."""

    context = ModelRequestContext(
        base_url=base_url,
        model=model,
        rollout_id=request_id,
    )
    return AsyncOpenAI(
        base_url=base_url,
        api_key="benchmax-explicit-auth",
        http_client=httpx.AsyncClient(auth=RequestModelAuth(auth, context)),
        max_retries=max_retries,
    )
