"""Instance-specific rubric generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from ._prompts import INSTANCE_WISE_RUBRIC_GENERATION_PROMPT
from .cache import AdaptiveRubrics, RubricCache
from .judge import Judge, JudgeError
from .rubric import Rubric, evaluate_single_rubric

logger = logging.getLogger(__name__)


async def generate_adaptive_rubrics(
    *,
    question: str,
    ground_truth: str,
    responses: Sequence[str],
    judge: Judge,
    existing_rubrics: str | None = None,
) -> AdaptiveRubrics:
    """Generate discriminative rubrics for a question and its responses."""

    response_block = "\n\n".join(
        f"Response {index}:\n{response}"
        for index, response in enumerate(responses, start=1)
    )
    prompt = (
        INSTANCE_WISE_RUBRIC_GENERATION_PROMPT
        + f"\nQuestion: {question}\n"
        + f"Ground Truth: {ground_truth}\n"
        + f"Responses:\n{response_block}\n"
    )
    if existing_rubrics and existing_rubrics.strip():
        prompt += f"\nExisting Rubrics:\n{existing_rubrics.strip()}\n"

    try:
        payload, _ = await judge.request_json(
            prompt,
            request_id="adaptive-rubric-generation",
        )
        generated = AdaptiveRubrics(
            positive=_parse_generated(payload, "positive_rubrics", "positive"),
            negative=_parse_generated(payload, "negative_rubrics", "negative"),
        )
    except JudgeError:
        raise
    except Exception as error:
        logger.exception("Adaptive rubric generation failed")
        raise JudgeError(f"adaptive rubric generation failed: {error}") from error

    logger.info(
        "adaptive_rubrics.generated positive=%d negative=%d",
        len(generated.positive),
        len(generated.negative),
    )
    return generated


async def generate_and_cache_adaptive_rubrics(
    *,
    question: str,
    ground_truth: str,
    responses: Sequence[str],
    judge: Judge,
    cache: RubricCache,
    existing_rubrics: str | None = None,
) -> AdaptiveRubrics:
    """Generate rubrics and retain those that discriminate the responses."""

    nonempty = tuple(response.strip() for response in responses if response.strip())
    if len(nonempty) < 2:
        return cache.get(question)

    cached_prompt = cache.get(question).format_for_prompt()
    generated = await generate_adaptive_rubrics(
        question=question,
        ground_truth=ground_truth,
        responses=nonempty,
        judge=judge,
        existing_rubrics=cached_prompt or existing_rubrics,
    )
    for rubric in generated.all:
        evaluations = await asyncio.gather(
            *(
                evaluate_single_rubric(
                    rubric,
                    question=question,
                    response=response,
                    ground_truth=ground_truth,
                    judge=judge,
                    log_result=False,
                )
                for response in responses
                if response.strip()
            )
        )
        cache.consider(question, rubric, [result.score for result in evaluations])
    return cache.get(question)


def _parse_generated(
    payload: dict[str, object],
    field: str,
    polarity: str,
) -> tuple[Rubric, ...]:
    raw_rubrics = payload.get(field, [])
    if not isinstance(raw_rubrics, list):
        raise ValueError(f"judge response field {field!r} must be a list")
    rubrics: list[Rubric] = []
    for index, value in enumerate(raw_rubrics):
        if not isinstance(value, dict):
            raise ValueError(f"judge response {field}[{index}] must be an object")
        title = value.get("title")
        description = value.get("description")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"judge response {field}[{index}].title must be non-empty")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"judge response {field}[{index}].description must be non-empty"
            )
        rubrics.append(
            Rubric(
                title=title.strip(),
                description=description.strip(),
                polarity=polarity,  # type: ignore[arg-type]
            )
        )
    return tuple(rubrics)
