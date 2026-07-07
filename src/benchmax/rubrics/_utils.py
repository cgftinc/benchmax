import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import json_repair
from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError

from benchmax.platform.credentials import resolve_judge_key

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The OpenAI SDK treats 401/403 as terminal (max_retries does not cover them). A
# judge bearer can expire mid-run, so on an auth error we rebuild the client —
# which re-resolves the credential (a fresh platform_bearer, or a token_provider()
# call) — and retry a few times. This is what turns a silent judge outage into a
# recoverable blip instead of a run-wide score-0 collapse.
_AUTH_ERRORS = (AuthenticationError, PermissionDeniedError)
_MAX_AUTH_ATTEMPTS = 3  # 1 initial + 2 rebuild-and-retry
_AUTH_BACKOFF_SECONDS = 0.5


def _resolve_judge_credential(
    api_key: str,
    base_url: Optional[str],
    token_provider: Optional[Callable[[], str]],
) -> Optional[str]:
    """token_provider wins over api_key and is re-invoked per attempt (so a
    refreshed token is picked up); otherwise fall back to resolve_judge_key."""
    if token_provider is not None:
        return token_provider()
    return resolve_judge_key(api_key, base_url)


async def _judge_call_with_retry(
    base_url: Optional[str],
    api_key: str,
    token_provider: Optional[Callable[[], str]],
    call: Callable[[AsyncOpenAI], Awaitable[T]],
) -> T:
    """Run ``call(client)`` against a freshly built AsyncOpenAI, retrying auth
    failures with a rebuilt (credential-re-resolved) client. The client is closed
    after every attempt (the per-call construction previously leaked). Non-auth
    exceptions propagate unchanged to the caller's own error handler."""
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_AUTH_ATTEMPTS):
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=_resolve_judge_credential(api_key, base_url, token_provider),
            max_retries=3,
        )
        try:
            return await call(client)
        except _AUTH_ERRORS as e:
            last_exc = e
            if attempt + 1 < _MAX_AUTH_ATTEMPTS:
                await asyncio.sleep(_AUTH_BACKOFF_SECONDS * (attempt + 1))
        finally:
            await client.close()
    assert last_exc is not None
    raise last_exc


def _extract_json(s: str) -> dict:
    """Extract JSON from a response string, handling markdown code blocks and thinking tags."""
    # Strip <think>...</think> tags that some models emit before JSON.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    if s.startswith("```") and s.endswith("```"):
        s = "\n".join(s.splitlines()[1:-1]).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Salvage truncated/malformed JSON.
    start = s.rfind("{")
    if start != -1:
        repaired = json_repair.loads(s[start:])
        if isinstance(repaired, dict) and repaired:
            return repaired

    raise ValueError("Response did not contain valid JSON.")


def _extract_completion_text(completion: str | List[Dict]) -> str:
    if isinstance(completion, list):
        if not completion or completion[-1]["role"] != "assistant":
            return ""
        return completion[-1]["content"].strip()
    return str(completion).strip()


def _static_rubric_key(title: str) -> str:
    key = title.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return f"rubric_{key.strip('_')}"


async def _zero_rubric_result() -> Dict[str, Any]:
    return {"score": 0, "reasoning": "Empty response", "llm_output": ""}
