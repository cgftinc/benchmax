from html import unescape
from pathlib import Path
import re
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from envs.base_sandbox import BaseSandbox, ToolDefinition
from envs.types import RewardFunction

SYSTEM_PROMPT = """Please use the tools provided to get accurate, up-to-date information.
Formulate each search query as a concise 1–2 word entity.
Write your complete answer on the final line only, within the xml tags <answer></answer>.\n
"""

def reward_func(prompt: str, completion: str, ground_truth: str, workspace: Path, **kwargs) -> float:
    """
    Reward = 1 if `ground_truth` (case-insensitive) appears anywhere *inside*
    the first <answer> … </answer> block of `completion`; otherwise 0.

    Falls back to 0 if the tag is missing or empty.
    """
    # Grab only the text inside the first <answer> … </answer> pair (case-insensitive).
    m = re.search(r'<answer>(.*?)</answer>', completion, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0.0

    # Unescape any XML entities (&amp; → &, etc.) and normalise whitespace.
    answer_text = unescape(m.group(1)).strip().lower()
    return float(ground_truth.lower() == answer_text)

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


def _safe_request(
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


def _make_wikipedia_tools(key_rotator: APIKeyRotator):
    """Return concrete implementations of Wikipedia search tools."""

    def _headers() -> Dict[str, str]:
        api_key = key_rotator.next()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # — Search tool ————————————————————————————
    def wikipedia_search_tool(q: str, limit: int = 10, **kwargs) -> Any:
        query = q
        if not query:
            return "Error: Missing required parameter: 'q'"

        try:
            resp = _safe_request(
                "GET",
                "https://en.wikipedia.org/w/api.php",
                headers=_headers(),
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "utf8": 1,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return f"Error: API request failed with status {resp.status_code}"
            return resp.json()["query"]["search"]
        except Exception as e:
            return f"Error: {str(e)}"

    # — Article‑summary tool —————————————————————————
    def wikipedia_get_article_tool(title: str, **kwargs) -> Any:
        if not title:
            return "Error: Missing required parameter: 'title'"

        try:
            resp = _safe_request(
                "GET",
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                headers=_headers(),
            )
            if resp.status_code != 200:
                return f"Error: API request failed with status {resp.status_code}"
            return resp.json()
        except Exception as e:
            return f"Error: {str(e)}"

    return wikipedia_search_tool, wikipedia_get_article_tool


class WikipediaSandbox(BaseSandbox):
    """A `BaseSandbox` pre-loaded with Wikipedia tools and key rotation.

    Parameters
    ----------
    api_keys : list[str] | None
        A list of Wikipedia REST API keys to rotate through.  If *None*
        or empty, requests are made unsigned.
    """
    system_prompt: Optional[str] = SYSTEM_PROMPT
    _reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(
        self,
        api_keys: Optional[List[str]] | None = None,
        **kwargs,
    ):
        # rotate api keys to circumvent timeouts
        self._key_rotator = APIKeyRotator(api_keys)

        search_tool, article_tool = _make_wikipedia_tools(self._key_rotator)
        search_tool_definition = ToolDefinition(
            name="search_wikipedia",
            description="Search Wikipedia articles by keyword.",
            input_schema={
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "Query string to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default 10).",
                    },
                },
                "required": ["q"],
            }
        )
        article_tool_definition = ToolDefinition(
            name="get_wikipedia_article",
            description="Fetch the summary paragraph for a Wikipedia article.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact title of the Wikipedia article.",
                    }
                },
                "required": ["title"],
            },
        )
        self.tools: Dict[str, Tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (search_tool_definition, search_tool),
            article_tool_definition.name: (article_tool_definition, article_tool)
        }

    def list_tools(self) -> List[ToolDefinition]:
        """List available tools."""
        return [
            self.tools[k][0] for k in sorted(self.tools)
        ]
    
    def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute a tool.
        Rollout_id isn't necessary since there is no statefulness for this wikipedia environment
        """
        _, tool_function = self.tools[tool_name]
        return tool_function(**tool_args)

    def init_rollout(self, rollout_id: str) -> None:
        return super().init_rollout(rollout_id)
    
    def cleanup_rollout(self, rollout_id: str) -> None:
        return super().cleanup_rollout(rollout_id)
    
    def get_rollout_workspace(self, rollout_id: str) -> Path:
        return super().get_rollout_workspace(rollout_id)
    