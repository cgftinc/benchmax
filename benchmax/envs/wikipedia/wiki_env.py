from html import unescape
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmax.envs.base_env import BaseEnv, ToolDefinition
from benchmax.envs.types import RewardFunction, StandardizedExample
from benchmax.envs.wikipedia.utils import APIKeyRotator, clean_html, safe_request

SYSTEM_PROMPT = """Please use the tools provided to get accurate, up-to-date information.
Formulate each search query as a concise 1–2 word entity.
Write your complete answer on the final line only as a concise entity, within the xml tags <answer></answer>.\n
"""

def reward_func(prompt: str, completion: str, ground_truth: str, workspace: Path, **kwargs) -> float:
    """Score 1.0 if the ground-truth answer appears exactly inside
    the first `<answer>...</answer>` block; otherwise 0.0.

    Args:
        prompt:   The user’s original query.
        completion:  The model’s generated text.
        ground_truth:  Expected answer to match (case-insensitive).
        workspace:  Path to the current workspace (unused).
        **kwargs:  Catch-all for BaseEnv signature compatibility.
    Returns:
        1.0 if `ground_truth` (lowercased) exactly matches the unescaped
        text inside the first `<answer>` block, else 0.0.
    """
    assert ground_truth is not None
    # Grab only the text inside the first <answer> … </answer> pair (case-insensitive).
    m = re.search(r'<answer>(.*?)</answer>', completion, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0.0

    # Unescape any XML entities (&amp; → &, etc.) and normalise whitespace.
    answer_text = unescape(m.group(1)).strip().lower()
    return float(ground_truth.lower() == answer_text)

def _make_wikipedia_tools(key_rotator: APIKeyRotator):
    """Return concrete implementations of Wikipedia search tools."""

    def _headers() -> Dict[str, str]:
        api_key = key_rotator.next()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # — Search tool ————————————————————————————
    def wikipedia_search_tool(q: str, limit: int = 10, **kwargs) -> Any:
        """Search Wikipedia articles by keyword.
        
        Args:
            q: The query string to search for.
            limit: Maximum number of results to return (default 10).
        Returns:
            A string with the search results, or an error message.
        """
        query = q
        if not query:
            return "Error: Missing required parameter: 'q'"

        try:
            resp = safe_request(
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

            hits = resp.json().get("query", {}).get("search", [])
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

    # — Article‑tool —————————————————————————
    def wikipedia_get_article_tool(title: str, max_chars: int = 10000, **kwargs) -> Any:
        """
        Fetches the full plaintext of a Wikipedia article.

        Args:
            title: The page title (e.g. "Python_(programming_language)")
            max_chars: Maximum number of characters to return (default 10,000).
        Returns:
            A string with the full article text, or an error message.
        """
        if not title:
            return "Error: Missing required parameter: 'title'"
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": True,      # get plain text, not HTML
            "redirects": True,        # follow any redirects
            "titles": title
        }

        try:
            resp = safe_request("GET", api_url, params=params, headers=_headers())
            if resp.status_code != 200:
                return f"Error: API request failed with status {resp.status_code}"

            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return f"Error: No pages found for title '{title}'"

            # pages is a dict keyed by pageid; grab the first one
            page = next(iter(pages.values()))
            extract = page.get("extract")
            if extract is None:
                return f"Error: No extract found for '{title}'"
            return extract[:max_chars]

        except Exception as e:
            return f"Error: {str(e)}"

    return wikipedia_search_tool, wikipedia_get_article_tool


class WikipediaEnv(BaseEnv):
    """Wikipedia Benchmax environment with Wikipedia search & fetch tools."""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

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
        """Execute a tool. Rollout_id isn't necessary since there is no statefulness
        for this wikipedia environment.

        Args:
            rollout_id: Identifier for the current rollout (not used here).
            tool_name: Name of the tool to run (e.g. "search_wikipedia").
            **tool_args: Arguments for the tool function.
        Returns:
            The result of the tool function, or an error message.
        """
        _, tool_function = self.tools[tool_name]
        return tool_function(**tool_args)

    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        return {
            "prompt": example.get("Question", ""),
            "ground_truth": example.get("Answer", None),
            "init_rollout_args": {}
        }

    def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        return super().init_rollout(rollout_id, **rollout_args)
    
    def cleanup_rollout(self, rollout_id: str) -> None:
        return super().cleanup_rollout(rollout_id)
    
    def get_rollout_workspace(self, rollout_id: str) -> Path:
        return super().get_rollout_workspace(rollout_id)

if __name__ == "__main__":
    # Example usage
    wiki_env = WikipediaEnv()
    print("Wiki Env initialized with MCP configuration.")
    print("System prompt:", wiki_env.system_prompt)
    print("Available Tools: ", wiki_env.list_tools())
    print("Run Wikipedia Search Tool:")
    results = wiki_env.run_tool("demo", "search_wikipedia", q="Python programming language", limit=5)
    print(results)