import html
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests


def clean_html(raw: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(text)

class APIKeyRotator:
    """Round‑robin iterator over a list of Wikipedia API keys.

    Rotation is **thread‑safe** so that concurrent tool calls running in
    different threads (or async tasks) cannot race and pick the same key.
    """
    def __init__(self, keys: Optional[List[str]] = None):
        self._keys: List[str] = keys or []
        self._lock = threading.Lock()
        self._idx = 0

    def next(self) -> Optional[str]:
        """Return the *next* key, or *None* if no keys were configured."""
        if not self._keys:
            return None
        with self._lock:
            key = self._keys[self._idx]
            self._idx = (self._idx + 1) % len(self._keys)
            return key


class RateLimitExceeded(Exception):
    """Raised when the Wikipedia API repeatedly returns HTTP 429."""


def safe_request(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    timeout: float = 10.0,
    json: Any | None = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 20,
    rate_limit_seconds: float = 1,
) -> requests.Response:
    """HTTP request helper that retries on 429 using exponential back‑off."""
    time.sleep(rate_limit_seconds)  # Short delay to avoid hammering the server immediately
    for attempt in range(max_retries + 1):
        resp = requests.request(
            method, url, headers=headers, params=params, json=json, timeout=timeout
        )
        if resp.status_code != 429:
            return resp  # Success *or* non‑rate‑limit error → caller handles

        if attempt == max_retries:
            raise RateLimitExceeded(
                f"Rate‑limit hit and {max_retries} retries exhausted."
            )
        print(f"Rate limit hit, retrying in {retry_delay_seconds:.1f} seconds...")
        time.sleep(retry_delay_seconds)
    return resp