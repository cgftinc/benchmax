"""Response parsing utilities for LLM outputs in Q&A generation."""

from __future__ import annotations

import json


def parse_json_from_llm_response(response_text: str) -> dict | None:
    """
    Extract and parse JSON from LLM response that may contain markdown code fences.

    Handles:
    - ```json ... ``` blocks
    - ``` ... ``` blocks
    - Raw JSON

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        clean_text = response_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        return json.loads(clean_text)
    except (json.JSONDecodeError, IndexError):
        return None


def parse_corpus_summary_response(response: str) -> tuple[str, list[str]]:
    """
    Parse corpus summary and example queries from LLM JSON response.

    Expected JSON structure:
    {
        "thoughts": "...",
        "summary": "...",
        "example_queries": ["q1", "q2", ...]
    }

    Args:
        response: Raw LLM response containing JSON

    Returns:
        Tuple of (corpus_summary, example_queries_list)
        Returns ("", []) if parsing fails
    """
    result = parse_json_from_llm_response(response)
    if result is None:
        return "", []

    summary = result.get("summary", "")
    example_queries = result.get("example_queries", [])

    if not isinstance(example_queries, list):
        example_queries = []

    return summary, example_queries
