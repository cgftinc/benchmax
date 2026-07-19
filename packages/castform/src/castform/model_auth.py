"""Castform-backed model authentication."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from benchmax.auth import ModelRequestContext

from castform import config
from castform.platform.credentials import platform_bearer

__all__ = ["CastformModelAuth"]


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
        return {"Authorization": f"Bearer {platform_bearer()}"}
