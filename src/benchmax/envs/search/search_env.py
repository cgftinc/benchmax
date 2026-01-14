from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition, StandardizedExample

SYSTEM_PROMPT = """Please use the search tool provided to find relevant information from the corpus.
Formulate effective search queries to retrieve the most relevant chunks.
You can filter by metadata or filename to narrow your search.
Write your complete answer on the final line only as a concise entity, within the xml tags <answer></answer>.\n
"""


def percent_of_text_a_in_text_b(text_a, text_b):
    if not text_a:
        return 0.0

    matcher = SequenceMatcher(None, text_a, text_b)
    matched_chars = sum(
        size for _, _, size in matcher.get_matching_blocks()
    )
    return (matched_chars / len(text_a))


async def chunk_overlap_reward_function(
    completion: str,
    ground_truth: str,
    **kwargs: Any
) -> float:
    """
    Reward function that computes the percentage of overlapping text between
    the completion and the ground truth.

    Args:
        completion: The model's generated text
        ground_truth: The reference text to compare against
        **kwargs: Additional arguments (not used here)
    Returns:
        float: A score between 0.0 and 1.0 representing the overlap percentage.
    """
    reference_chunks = kwargs.get("reference_chunks", [])
    reference_string = " ".join(reference_chunks)
    completion_str = completion if isinstance(completion, str) else ""
    if isinstance(completion, list):
        completion_str = " ".join(
            [c.get("content", "") for c in completion if isinstance(c, dict) and c.get("role", "") != "assistant"]
        )
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            if msg.get("role", "") != "assistant":
                continue
            msg_content = msg.get("content", "")
            if msg_content.count("<tool_call>") >= 4:
                return 0.0

    if reference_string:
        overlap_score = percent_of_text_a_in_text_b(reference_string, completion_str)
        if overlap_score >= 0.25:
            return overlap_score
    return 0.0


class SearchEnv(BaseEnv):
    """Search environment with BM25 corpus search tool."""

    system_prompt: str = SYSTEM_PROMPT

    def __init__(
        self,
        api_key: str,
        corpus_id: str,
        base_url: str,
        **kwargs,
    ):
        """
        Initialize the search environment.

        Args:
            api_key: API key for authentication (required)
            corpus_id: ID of the corpus to search (required)
            base_url: Base URL of the search API (required)
        """
        if not api_key:
            raise ValueError("api_key is required")
        if not corpus_id:
            raise ValueError("corpus_id is required")

        self._api_key = api_key
        self._corpus_id = corpus_id
        self._base_url = base_url.rstrip("/")

        search_tool_definition = ToolDefinition(
            name="search_corpus",
            description="Search the corpus using BM25 with optional metadata and filename filtering.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata filters (e.g., {'ticker': 'DDOG', 'year': 2024}).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename filter. Simple string for substring match (e.g., 'config') or regex pattern (e.g., '.*\\.json$').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        )

        self._tools: Dict[str, Tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (search_tool_definition, self._search_corpus_tool)
        }

    async def _search_corpus_tool(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> str:
        """
        Search the corpus using BM25.

        Args:
            query: Search query string
            metadata: Optional metadata filters
            filename: Optional filename filter (substring or regex)
            limit: Maximum number of results

        Returns:
            Formatted search results or error message
        """
        if not query:
            return "Error: Missing required parameter: 'query'"

        # Build request body
        request_body = {"query": query, "limit": limit}
        if metadata:
            request_body["metadata"] = metadata
        if filename:
            request_body["filename"] = filename

        # Build URL
        url = f"{self._base_url}/api/corpora/{self._corpus_id}/search"
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return f"Error: API request failed with status {resp.status}: {error_text}"

                    data = await resp.json()

            results = data.get("results", [])
            total = data.get("total", 0)

            if not results:
                return "No results found."

            # Format results
            lines = []
            for i, item in enumerate(results, start=1):
                filename_val = item.get("filename", "—")
                score = item.get("score")
                score_str = f"(score: {score:.2f})" if score is not None else "(filtered)"
                content = item.get("content", "")
                metadata_val = item.get("metadata", {})

                lines.append(f"{i}. {filename_val} {score_str}")
                lines.append(f"   Content: {content}")
                if metadata_val:
                    lines.append(f"   Metadata: {metadata_val}")

            lines.append(f"\nTotal: {total} results")
            return "\n".join(lines)

        except aiohttp.ClientError as e:
            return f"Error: Network error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

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
            tool_name: Name of the tool (e.g., "search_corpus")
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
        """Compute rewards using the chunk overlap reward function."""
        return {
            "chunk_overlap": await chunk_overlap_reward_function(completion, ground_truth, **kwargs)
        }
