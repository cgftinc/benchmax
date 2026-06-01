import asyncio
from typing import Dict, List, Literal, Optional, cast

from openai import AsyncOpenAI

from ._utils import _extract_json
from .cache import (
    _empty_cache_entry,
    _format_cached_rubrics_for_prompt,
    filter_and_cache_rubrics,
    load_rubric_cache,
)
from .prompts import INSTANCE_WISE_RUBRIC_GENERATION_PROMPT
from benchmax.platform.credentials import resolve_judge_key

from .rubric import _cache_dict_to_rubric, evaluate_single_rubric


async def generate_instance_wise_adaptive_rubrics(
    question: str,
    ground_truth: str,
    response_list: List[str],
    model_name: Optional[str] = None,
    existing_rubrics: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: str = "",
    timeout: Optional[float] = None,
) -> Optional[dict]:
    """
    Generate instance-wise adaptive rubrics using OpenAI async client.

    Args:
        question: The original question
        ground_truth: The reference answer
        response_list: List of model responses to analyze
        existing_rubrics: Optional existing rubrics to consider
        model_name: Model name for rubric generation (defaults to RUBRIC_GENERATION_MODEL env var)
        base_url: Base URL for the OpenAI-compatible endpoint
        api_key: API key for authentication (defaults to empty string)
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing positive_rubrics and negative_rubrics, or None if generation fails
    """
    prompt_suffix = f"Question: {question}\nGround Truth: {ground_truth}\nResponses:\n"
    for i, response in enumerate(response_list):
        prompt_suffix += f"Response {i + 1}:\n{response}\n\n"

    if existing_rubrics:
        prompt_suffix += f"\n\nExisting Rubrics:\n{existing_rubrics}"

    prompt = INSTANCE_WISE_RUBRIC_GENERATION_PROMPT + prompt_suffix

    # Explicit api_key wins; otherwise resolve the Castform platform credential
    # seam (ACT_AS_TOKEN_PATH in training, PLATFORM_API_KEY in playground /
    # self-serve) — the same surface the search clients use. Falls back to None
    # (→ OPENAI_API_KEY) when no platform credential is present, for direct use.
    client = AsyncOpenAI(
        base_url=base_url, api_key=resolve_judge_key(api_key, base_url), max_retries=3
    )

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=timeout,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""
        if not content:
            print("Empty response from model")
            return None

        obj = _extract_json(content)
        print(f"Generated instance-wise adaptive rubrics: {obj}")
        return obj

    except Exception as e:
        print(f"Prompt: {prompt}")
        print(f"Error generating instance-wise adaptive rubrics: {e}")
        return None


async def _generate_and_cache_rubrics(
    completion_texts: List[str],
    user_prompt: str,
    ground_truth: str,
    model_name: str,
    llm_judge_url: str,
    timeout: Optional[float],
    question_hash: str,
    existing_rubrics: Optional[str],
    cache: Dict,
) -> Dict:
    """Generate adaptive rubrics, evaluate variance across responses, and update cache."""
    print(f"Generating rubrics for {len(completion_texts)} responses...")

    existing_rubrics_str = _format_cached_rubrics_for_prompt(cache) or existing_rubrics

    rubric_result = await generate_instance_wise_adaptive_rubrics(
        question=user_prompt,
        ground_truth=ground_truth,
        response_list=[c for c in completion_texts if c],
        model_name=model_name,
        existing_rubrics=existing_rubrics_str,
        base_url=llm_judge_url,
        timeout=timeout,
    )

    if rubric_result:
        for rubric_type, rtype in [
            ("positive_rubrics", cast(Literal["positive", "negative"], "positive")),
            ("negative_rubrics", cast(Literal["positive", "negative"], "negative")),
        ]:
            for rubric_dict in rubric_result.get(rubric_type, []):
                rubric = _cache_dict_to_rubric(rubric_dict, rtype)
                eval_tasks = [
                    evaluate_single_rubric(
                        rubric=rubric,
                        question=user_prompt,
                        response=resp,
                        model_name=model_name,
                        base_url=llm_judge_url,
                        timeout=timeout,
                    )
                    for resp in completion_texts
                ]
                scores = [r["score"] for r in await asyncio.gather(*eval_tasks)]
                filter_and_cache_rubrics(
                    question_hash=question_hash,
                    new_rubrics={rubric_type: [rubric_dict]},
                    rubric_type=rubric_type,
                    scores=scores,
                )

    return load_rubric_cache().get(question_hash, _empty_cache_entry())
