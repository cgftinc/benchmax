"""Stdlib-HTTP tool-call Model for upstream mini-swe-agent.

Drop-in replacement for ``LitellmModel`` that speaks to one OpenAI-compatible
endpoint with ``urllib``, so the litellm dependency closure never enters the
sandbox. Everything else (agent loop, prompts, parsing, observation
formatting) is upstream minisweagent code.
"""

from __future__ import annotations

import json
import os
import time
import types
from typing import Any

from pydantic import BaseModel

from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.exceptions import FormatError

import urllib.error
import urllib.request


class CastformModelConfig(BaseModel):
    model_name: str
    base_url: str
    observation_template: str
    format_error_template: str
    request_timeout: int = 600
    multimodal_regex: str = ""


def _tool_call_shim(call: dict) -> types.SimpleNamespace:
    """Adapt a raw tool-call dict to the object shape upstream parsing expects."""

    return types.SimpleNamespace(
        id=call.get("id"),
        function=types.SimpleNamespace(
            name=call["function"]["name"],
            arguments=call["function"]["arguments"],
        ),
    )


class CastformToolcallModel:
    def __init__(self, **kwargs: Any) -> None:
        self.config = CastformModelConfig(**kwargs)

    @staticmethod
    def _wire_message(message: dict) -> dict:
        """Strip loop-internal and null fields the API would reject on echo."""

        allowed = ("role", "content", "tool_calls", "tool_call_id", "name")
        return {
            key: message[key]
            for key in allowed
            if message.get(key) is not None or key in ("role", "content")
        }

    def _request(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.config.model_name,
            "messages": [self._wire_message(m) for m in messages],
            "tools": [BASH_TOOL],
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
                # Cloudflare WAF on *.castform.dev rejects default urllib UAs.
                "User-Agent": "castform-mini-swe/2.4.5",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"model call failed: HTTP {error.code}: {body}") from error

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        response = self._request(messages)
        choice = response["choices"][0]
        raw_calls = choice["message"].get("tool_calls") or []
        try:
            actions = parse_toolcall_actions(
                [_tool_call_shim(call) for call in raw_calls],
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": choice.get("finish_reason")},
            )
        except FormatError as error:
            error.messages[0]["extra"]["response"] = response
            raise
        message = dict(choice["message"])
        message["extra"] = {
            "actions": actions,
            "response": response,
            "cost": 0.0,
            "timestamp": time.time(),
        }
        return message

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
