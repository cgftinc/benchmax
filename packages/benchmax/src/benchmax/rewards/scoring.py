"""Composable rubric scoring functions that return reward maps."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Sequence
from typing import Any

from .adaptive import generate_and_cache_adaptive_rubrics
from .cache import RubricCache
from .helpers import Completion, extract_completion_text
from .judge import Judge
from .rubric import (
    Rubric,
    RubricEvaluation,
    evaluate_rubric_ranking,
    evaluate_single_rubric,
)

logger = logging.getLogger(__name__)


def rubric_reward_key(rubric: Rubric) -> str:
    """Return the stable reward key for a rubric."""

    slug = re.sub(r"[^a-z0-9]+", "_", rubric.title.lower()).strip("_")
    if not slug:
        raise ValueError("rubric title must contain at least one letter or digit")
    return f"rubric_{slug}"


async def score_rubrics(
    rollout_id: str,
    completion: Completion,
    *,
    ground_truth: Any,
    rubrics: Sequence[Rubric],
    question: str,
    judge: Judge,
) -> dict[str, float]:
    """Score one completion independently against every rubric."""

    _validate_rubric_keys(rubrics, allow_empty=False)
    if not isinstance(rollout_id, str) or not rollout_id.strip():
        raise ValueError("rollout_id must be non-empty")
    text = extract_completion_text(completion)
    if not text:
        rewards = {rubric_reward_key(rubric): 0.0 for rubric in rubrics}
        logger.info("rubric.scored rollout_id=%s empty_completion=true", rollout_id)
        return rewards

    evaluations = await asyncio.gather(
        *(
            evaluate_single_rubric(
                rubric,
                question=question,
                response=text,
                ground_truth=str(ground_truth) if ground_truth is not None else None,
                judge=judge,
                log_result=False,
            )
            for rubric in rubrics
        )
    )
    rewards = {
        rubric_reward_key(rubric): rubric.reward_for(evaluation.score)
        for rubric, evaluation in zip(rubrics, evaluations, strict=True)
    }
    logger.info("rubric.scored rollout_id=%s rewards=%s", rollout_id, rewards)
    return rewards


async def score_group_rubrics(
    rollout_ids: Sequence[str],
    completions: Sequence[Completion],
    *,
    ground_truth: Any,
    question: str,
    judge: Judge,
    rubrics: Sequence[Rubric] = (),
    use_adaptive: bool = False,
    existing_rubrics: Sequence[Rubric] = (),
    cache: RubricCache | None = None,
) -> list[dict[str, float]]:
    """Score a sibling group independently against static and adaptive rubrics."""

    _validate_group(rollout_ids, completions)
    _validate_rubric_keys(rubrics, allow_empty=use_adaptive)
    texts = [extract_completion_text(completion) for completion in completions]

    static_tasks: list[tuple[int, Rubric, Awaitable[RubricEvaluation]]] = []
    for index, text in enumerate(texts):
        if not text:
            continue
        for rubric in rubrics:
            static_tasks.append(
                (
                    index,
                    rubric,
                    evaluate_single_rubric(
                        rubric,
                        question=question,
                        response=text,
                        ground_truth=(
                            str(ground_truth) if ground_truth is not None else None
                        ),
                        judge=judge,
                        log_result=False,
                    ),
                )
            )

    rewards = [
        {rubric_reward_key(rubric): 0.0 for rubric in rubrics}
        for _ in completions
    ]
    if static_tasks:
        results = await asyncio.gather(*(task for _, _, task in static_tasks))
        for (index, rubric, _), result in zip(static_tasks, results, strict=True):
            rewards[index][rubric_reward_key(rubric)] = rubric.reward_for(result.score)

    if use_adaptive:
        active_cache = cache or RubricCache()
        adaptive = await generate_and_cache_adaptive_rubrics(
            question=question,
            ground_truth=str(ground_truth) if ground_truth is not None else "",
            responses=texts,
            judge=judge,
            cache=active_cache,
            existing_rubrics=existing_rubrics,
        )
        adaptive_evaluations: list[
            tuple[int, Rubric, Awaitable[RubricEvaluation]]
        ] = []
        for index, text in enumerate(texts):
            if not text:
                continue
            for rubric in adaptive.all:
                adaptive_evaluations.append(
                    (
                        index,
                        rubric,
                        evaluate_single_rubric(
                            rubric,
                            question=question,
                            response=text,
                            ground_truth=(
                                str(ground_truth) if ground_truth is not None else ""
                            ),
                            judge=judge,
                            log_result=False,
                        ),
                    )
                )
        totals = [0.0] * len(completions)
        counts = [0] * len(completions)
        if adaptive_evaluations:
            results = await asyncio.gather(
                *(task for _, _, task in adaptive_evaluations)
            )
            for (index, rubric, _), result in zip(
                adaptive_evaluations, results, strict=True
            ):
                totals[index] += rubric.reward_for(result.score)
                counts[index] += 1
        for index, reward in enumerate(rewards):
            reward["rubric_adaptive"] = (
                totals[index] / counts[index] if counts[index] else 0.0
            )

    for rollout_id, reward in zip(rollout_ids, rewards, strict=True):
        logger.info("rubric.scored rollout_id=%s rewards=%s", rollout_id, reward)
    return rewards


async def rank_group_rubrics(
    rollout_ids: Sequence[str],
    completions: Sequence[Completion],
    *,
    ground_truth: Any,
    question: str,
    judge: Judge,
    rubrics: Sequence[Rubric],
    include_ground_truth: bool = True,
) -> list[dict[str, float]]:
    """Rank a sibling group once per rubric and return per-rollout rewards."""

    _validate_group(rollout_ids, completions)
    _validate_rubric_keys(rubrics, allow_empty=False)
    texts = [extract_completion_text(completion) for completion in completions]
    reference = (
        str(ground_truth).strip()
        if include_ground_truth and ground_truth is not None
        else ""
    )
    results = await asyncio.gather(
        *(
            evaluate_rubric_ranking(
                rubric,
                question=question,
                responses=texts,
                ground_truth=reference or None,
                judge=judge,
                log_result=False,
            )
            for rubric in rubrics
        )
    )

    rewards: list[dict[str, float]] = [{} for _ in completions]
    for rubric, result in zip(rubrics, results, strict=True):
        key = rubric_reward_key(rubric)
        for index, score in enumerate(result.scores):
            rewards[index][key] = score
    for rollout_id, reward in zip(rollout_ids, rewards, strict=True):
        logger.info("rubric.ranked rollout_id=%s rewards=%s", rollout_id, reward)
    return rewards


def _validate_group(
    rollout_ids: Sequence[str], completions: Sequence[Completion]
) -> None:
    if len(rollout_ids) != len(completions):
        raise ValueError("rollout_ids and completions must have the same length")
    if not rollout_ids:
        raise ValueError("rollout group must be non-empty")
    if any(
        not isinstance(rollout_id, str) or not rollout_id.strip()
        for rollout_id in rollout_ids
    ):
        raise ValueError("rollout_ids must be non-empty strings")
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("rollout_ids must be unique")


def _validate_rubric_keys(
    rubrics: Sequence[Rubric], *, allow_empty: bool
) -> None:
    if not rubrics and not allow_empty:
        raise ValueError("at least one rubric is required")
    keys = [rubric_reward_key(rubric) for rubric in rubrics]
    if len(set(keys)) != len(keys):
        raise ValueError("rubric titles must produce unique reward keys")
