from html import unescape
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition, StandardizedExample
from benchmax.envs.wikipedia.utils import APIKeyRotator, clean_html, safe_request

SYSTEM_PROMPT = """Please use the tools provided to get accurate, up-to-date information.
Formulate each search query as a concise 1-2 word entity.
Write your complete answer on the final line only as a concise entity, within the xml tags <answer></answer>.\n
"""


def text_match_reward_function(completion: str, ground_truth: str, **kwargs) -> float:
    """
    Score 1.0 if ground truth appears in <answer> tags, else 0.0.

    Args:
        completion: The model's generated text
        ground_truth: Expected answer (case-insensitive)
        **kwargs: Catch-all for BaseEnv compatibility

    Returns:
        1.0 if ground_truth matches the answer text, else 0.0
    """
    assert ground_truth is not None

    m = re.search(
        r"<answer>(.*?)</answer>", completion, flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        return 0.0

    answer_text = unescape(m.group(1)).strip().lower()
    return float(ground_truth.lower() == answer_text)


def _make_wikipedia_tools(key_rotator: APIKeyRotator):
    """Return Wikipedia search and article fetch tool implementations."""

    def _headers() -> Dict[str, str]:
        api_key = key_rotator.next()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def wikipedia_search_tool(q: str, limit: int = 10, **kwargs) -> Any:
        """
        Search Wikipedia articles by keyword.

        Args:
            q: Query string to search for
            limit: Maximum number of results (default 10)

        Returns:
            Formatted search results or error message
        """
        query = q
        if not query:
            return "Error: Missing required parameter: 'q'"

        try:
            resp = await safe_request(
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
            if resp is None:
                return (
                    "Error: Failed to obtain response from Wikipedia API after retries."
                )
            if resp.status != 200:
                return f"Error: API request failed with status {resp.status}"

            data = await resp.json()
            hits = data.get("query", {}).get("search", [])
            if not hits:
                return "No results found."

            lines = []
            for i, item in enumerate(hits, start=1):
                title = item.get("title", "—")
                snippet = clean_html(item.get("snippet", ""))
                lines.append(f"{i}. {title}\n   {snippet}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    async def wikipedia_get_article_tool(
        title: str, max_chars: int = 10000, **kwargs
    ) -> Any:
        """
        Fetch the full plaintext of a Wikipedia article.

        Args:
            title: Page title (e.g., "Python_(programming_language)")
            max_chars: Maximum characters to return (default 10,000)

        Returns:
            Article text or error message
        """
        if not title:
            return "Error: Missing required parameter: 'title'"

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }

        try:
            resp = await safe_request(
                "GET",
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=_headers(),
            )
            if resp is None:
                return (
                    "Error: Failed to obtain response from Wikipedia API after retries."
                )
            if resp.status != 200:
                return f"Error: API request failed with status {resp.status}"

            data = await resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return f"Error: No pages found for title '{title}'"

            page = next(iter(pages.values()))
            extract = page.get("extract")
            if extract is None:
                return f"Error: No extract found for '{title}'"
            return extract[:max_chars]

        except Exception as e:
            return f"Error: {str(e)}"

    return wikipedia_search_tool, wikipedia_get_article_tool


class WikipediaEnv(BaseEnv):
    """Wikipedia environment with search and article fetch tools."""

    system_prompt: str = SYSTEM_PROMPT

    def __init__(
        self,
        wikipedia_api_keys: Optional[List[str]] | None = None,
        **kwargs,
    ):
        self._key_rotator = APIKeyRotator(wikipedia_api_keys)

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
            },
        )

        article_tool_definition = ToolDefinition(
            name="get_wikipedia_article",
            description="Fetch the full plaintext of a Wikipedia article.",
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
        self._tools: Dict[str, Tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (search_tool_definition, search_tool),
            article_tool_definition.name: (article_tool_definition, article_tool),
        }

    async def shutdown(self):
        # no cleanup required
        pass

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> StandardizedExample:
        return StandardizedExample(
            prompt=example.get("Question", ""),
            ground_truth=example.get("Answer", None),
            init_rollout_args={},
        )

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools."""
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """
        Execute a tool.

        Args:
            rollout_id: Identifier for current rollout (unused for stateless env)
            tool_name: Name of the tool (e.g., "search_wikipedia")
            **tool_args: Arguments for the tool function

        Returns:
            Tool execution result or error message
        """
        _, tool_function = self._tools[tool_name]
        return await tool_function(**tool_args)

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """Initialize rollout (no-op for stateless environment)."""
        pass

    async def release_rollout(self, rollout_id: str) -> None:
        """Release rollout (no-op for stateless environment)."""
        pass

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        """Not implemented for this environment."""
        pass

    async def copy_content_to_workspace(
        self, rollout_id: str, src_content: str | bytes, dst_filename: str
    ) -> None:
        """Not implemented for this environment."""
        pass

    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        """Not implemented for this environment."""
        pass

    async def compute_reward(
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        """Compute rewards using the text match reward function."""
        return {
            "text_match": text_match_reward_function(completion, ground_truth, **kwargs)
        }
