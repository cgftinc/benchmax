from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Union, Awaitable
from fastmcp import Client
from html import unescape

RewardFunction = Callable[..., Union[float, Awaitable[float]]]


async def text_match_reward(
    completion: List[Dict[str, Any]],
    ground_truth: str,
    mcp_client: Client,
    workspace: Path,
    **kwargs: Any,
) -> float:
    """
    Reward = 1 if `ground_truth` (case-insensitive) appears anywhere *inside*
    the first <answer> … </answer> block of `completion`; otherwise 0.

    Falls back to 0 if the tag is missing or empty.
    """

    completion_text = completion[-1].get("content", "") if completion else ""

    # Grab only the text inside the first <answer> … </answer> pair (case-insensitive).
    m = re.search(
        r"<answer>(.*?)</answer>", completion_text, flags=re.IGNORECASE | re.DOTALL
    )
    if m is None:
        return 0.0

    # Unescape any XML entities (&amp; → &, etc.) and normalise whitespace.
    answer_text = unescape(m.group(1)).strip().lower()

    try:
        # Try to interpret both as floats for numerical comparison.
        return float(float(ground_truth.lower()) == float(answer_text))
    except ValueError:
        return 0.0


# -------------------------------
# Export reward functions
# -------------------------------
reward_functions: Dict[str, RewardFunction] = {"match": text_match_reward}
