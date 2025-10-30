import asyncio
import html
import re
import threading
from typing import Any, Dict, List, Optional

import aiohttp


def clean_html(raw: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(text)


class APIKeyRotator:
    """Thread-safe round-robin iterator over API keys."""

    def __init__(self, keys: Optional[List[str]] = None):
        self._keys: List[str] = keys or []
        self._lock = threading.Lock()
        self._idx = 0

    def next(self) -> Optional[str]:
        """Return the next key, or None if no keys configured."""
        if not self._keys:
            return None
        with self._lock:
            key = self._keys[self._idx]
            self._idx = (self._idx + 1) % len(self._keys)
            return key


class RateLimitExceeded(Exception):
    """Raised when the API repeatedly returns HTTP 429."""


async def safe_request(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    timeout: float = 10.0,
    json: Any | None = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 20,
    rate_limit_seconds: float = 2.5,
) -> Optional[aiohttp.ClientResponse]:
    """
    Async HTTP request with exponential backoff on 429 rate limits.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL
        headers: Request headers
        params: Query parameters
        timeout: Request timeout in seconds
        json: JSON body for request
        max_retries: Maximum retry attempts on 429
        retry_delay_seconds: Base delay between retries
        rate_limit_seconds: Initial delay before first request

    Returns:
        aiohttp.ClientResponse object

    Raises:
        RateLimitExceeded: When max retries exhausted on 429 errors
    """
    await asyncio.sleep(rate_limit_seconds)

    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries + 1):
            async with session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 429:
                    # Read response content before returning
                    content = await resp.read()
                    # Create a new response object with the content
                    resp._body = content
                    return resp

                if attempt == max_retries:
                    raise RateLimitExceeded(
                        f"Rate limit hit and {max_retries} retries exhausted."
                    )

                print(f"Rate limit hit, retrying in {retry_delay_seconds:.1f}s...")
                await asyncio.sleep(retry_delay_seconds)
