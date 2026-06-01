import json
import re
from typing import Any, Dict, List

import json_repair


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
