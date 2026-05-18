"""LinkerEnv — lightweight search environment for LLM-driven chunk linking.

Used by SearchAgentLinker to let an LLM search a corpus and find
related chunks for multi-hop QA generation. Much simpler than SearchEnv:
no reward components, no citation tracking, no answer correctness.
"""

from __future__ import annotations

import traceback
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Example, Messages, ToolDefinition

from benchmax.rag.corpus.search_client import SearchClient

MAX_TOOL_OUTPUT_CHARS = 8000
TOOL_OUTPUT_TRUNCATION_SUFFIX = "\n...[truncated]"

_SYSTEM_PROMPT = """\
You are building an evidence chain for multi-hop question generation.

Given a primary document chunk, find secondary chunks that each contribute \
UNIQUE, NECESSARY information. Together, these chunks must form a chain \
where answering a question requires combining information from ALL chunks — \
no single chunk suffices alone.

## Search Strategy

Search ITERATIVELY — one targeted search at a time:

1. Read the primary chunk. Identify a SPECIFIC piece of information that, \
combined with complementary information from another document, would require \
multi-step reasoning to answer.

2. Search for that SPECIFIC complementary information. Target the GAP — \
what is missing, not what is already known.

3. After reviewing results, reason about:
   - What UNIQUE information a result adds (that the primary does not have)
   - Why this information REQUIRES the primary chunk to be useful (logical dependency)
   - What information gap remains for the next search (if more hops needed)

4. Repeat for each additional hop needed.

## What Makes Good Multi-Hop Chunks

GOOD — logical dependency between chunks:
- Primary: "Team X uses framework Y for API development"
- Secondary: "Framework Y requires config Z for production deployment"
→ A question needs BOTH: "How should Team X configure their API for production?"

BAD — topically related but independent:
- Primary: "Team X uses framework Y"
- Secondary: "Team Z also uses framework Y"
→ No dependency. Either chunk answers independently.

## Output

After all searches, output your final selection:
<evidence_chain>
<chunk role="primary">What unique information this provides</chunk>
<chunk role="secondary" search_step="1">What unique info this adds + \
why it depends on the primary</chunk>
<connection>How the chunks connect to require multi-step reasoning</connection>
</evidence_chain>\
"""

_REASONING_MODE_HINTS: dict[str, str] = {
    "temporal": (
        "\nFocus on TEMPORAL connections: find chunks describing events, "
        "changes, or states at DIFFERENT times that must be combined to "
        "understand a chronological sequence."
    ),
    "inference": (
        "\nFocus on INFERENTIAL connections: find chunks where the answer "
        "is not stated directly but must be DEDUCED by combining evidence. "
        "Look for premises in different documents that together imply "
        "a conclusion."
    ),
    "sequential": (
        "\nFocus on SEQUENTIAL connections: find chunks describing "
        "different steps in a process that spans multiple documents."
    ),
}


class LinkerEnv(BaseEnv):
    """Search environment for LLM-driven chunk linking.

    Exposes a single ``search`` tool backed by a :class:`SearchClient`.
    The LLM generates queries to find related chunks.

    Args:
        search: A pickle-safe :class:`SearchClient` instance.
        max_search_calls: Maximum number of search calls allowed.
    """

    def __init__(
        self,
        search: SearchClient,
        *,
        max_search_calls: int = 3,
        **kwargs: Any,
    ) -> None:
        self._search = search
        self._max_search_calls = max_search_calls

        modes = sorted(search.available_modes)
        if "hybrid" in modes:
            self._default_mode = "hybrid"
        elif "lexical" in modes:
            self._default_mode = "lexical"
        elif modes:
            self._default_mode = modes[0]
        else:
            self._default_mode = "lexical"

        search_tool = ToolDefinition(
            name="search",
            description="Search the corpus for related content.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        )
        self._tools: dict[str, tuple[ToolDefinition, Any]] = {
            "search": (search_tool, self._search_tool),
        }

        self.system_prompt = _SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}'"
        _, tool_fn = self._tools[tool_name]
        return await tool_fn(**tool_args)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> Example:
        target_n = example.get("target_n", 1)
        reasoning_mode = example.get("reasoning_mode", "")
        mode_hint = _REASONING_MODE_HINTS.get(reasoning_mode, "")

        prompt = (
            f"Find {target_n} secondary chunk(s) that form a logical "
            f"evidence chain with the following primary chunk. Each "
            f"secondary must contribute UNIQUE, NECESSARY information "
            f"— not just topically related content.{mode_hint}\n\n"
            f"Primary chunk:\n{example.get('prompt', '')}"
        )
        return make_example(
            seed_messages=[{"role": "user", "content": prompt}],
            task={"target_n": target_n, "reasoning_mode": reasoning_mode},
        )

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {"linking": 1.0}

    # ------------------------------------------------------------------
    # Search tool
    # ------------------------------------------------------------------

    async def _search_tool(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        if not query:
            return "Error: Missing required parameter: 'query'"
        try:
            results = self._search.search(query=query, mode=self._default_mode, top_k=limit)
            return self._format_results(results)
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "")
            score = r.get("score", 0.0)
            content = r.get("content", "")
            header = f"{i}."
            if source:
                header += f" [source: {source}]"
            if score:
                header += f" (score: {score:.2f})"
            lines.append(f"{header}\n   {content}")
        return _truncate("\n".join(lines))


def _truncate(
    text: str,
    max_chars: int = MAX_TOOL_OUTPUT_CHARS,
    suffix: str = TOOL_OUTPUT_TRUNCATION_SUFFIX,
) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix
