"""Serializable configuration and request handling for reward judges."""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
)

from benchmax.auth import ModelAuth, ModelRequestContext
from benchmax.envs.shared_types import RolloutFailure

_AUTH_ERRORS = (AuthenticationError, PermissionDeniedError)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class JudgeError(RolloutFailure):
    """A judge failed without producing a trustworthy reward signal."""

    def __init__(self, message: str) -> None:
        super().__init__("judge_error", message)


@dataclass(frozen=True, slots=True)
class Judge:
    """Configuration for an OpenAI-compatible reward judge.

    ``auth`` is resolved immediately before every request. An
    :class:`~benchmax.auth.InjectedAuth` therefore remains serializable in a
    bundle while the runtime supplies the real credential at execution time.
    """

    model: str
    base_url: str
    auth: ModelAuth
    timeout: float | None = 60.0
    max_retries: int = 3
    auth_attempts: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("judge model must be non-empty")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("judge base_url must be non-empty")
        if not isinstance(self.auth, ModelAuth):
            raise TypeError("judge auth must implement ModelAuth")
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or self.timeout <= 0
        ):
            raise ValueError("judge timeout must be positive or None")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("judge max_retries must be non-negative")
        if (
            isinstance(self.auth_attempts, bool)
            or not isinstance(self.auth_attempts, int)
            or self.auth_attempts < 1
        ):
            raise ValueError("judge auth_attempts must be positive")

    async def request_json(
        self,
        prompt: str,
        *,
        request_id: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Call the judge and return ``(parsed_object, raw_response)``."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("judge prompt must be non-empty")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("judge request_id must be non-empty")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("judge temperature must be numeric")
        if not math.isfinite(float(temperature)):
            raise ValueError("judge temperature must be finite")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("judge max_tokens must be positive or None")

        last_auth_error: Exception | None = None
        for attempt in range(self.auth_attempts):
            headers = await self.auth.headers_for_request(
                ModelRequestContext(
                    base_url=self.base_url,
                    model=self.model,
                    rollout_id=request_id,
                )
            )
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="benchmax-explicit-auth",
                default_headers=dict(headers),
                max_retries=self.max_retries,
            )
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "timeout": self.timeout,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            try:
                response = await client.chat.completions.create(**request)
                raw = (
                    (response.choices[0].message.content or "").strip()
                    if response.choices
                    else ""
                )
                if not raw:
                    raise ValueError("judge returned an empty response")
                return _parse_json_object(raw), raw
            except _AUTH_ERRORS as error:
                last_auth_error = error
                if attempt + 1 < self.auth_attempts:
                    await asyncio.sleep(0.5 * (attempt + 1))
            finally:
                await client.close()

        assert last_auth_error is not None
        raise last_auth_error


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object without repairing malformed judge output."""

    text = _THINK_BLOCK.sub("", raw).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge response did not contain a valid JSON object")
