from html import unescape
from pathlib import Path
import re
from typing import List, Any

from benchmax.envs.local_mcp_env import LocalMCPEnv
from benchmax.envs.types import RewardFunction, StandardizedExample

SYSTEM_PROMPT = """Please use the tools provided to do any computation.
Write your complete answer on the final line only, within the xml tags <answer></answer>.\n
"""

MCP_CONFIG = """
{
    "mcpServers": {
      "server-name": {
        "command": "uvx",
        "args": ["mcp-server-calculator"]
      }
    }
}
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

class MathEnv(LocalMCPEnv):
    """Environment for math problems, using local MCP tools."""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(self, **kwargs):
        super().__init__(MCP_CONFIG)
    
    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        return StandardizedExample(
            prompt=example.get("task", ""),
            ground_truth=example.get("answer", ""),
            init_rollout_args={}
        )
