"""Instance-specific rubric generation and its caller-owned cache."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .judge import Judge, JudgeError
from .prompts import build_adaptive_rubric_prompt
from .rubric import Rubric, RubricPolarity, evaluate_single_rubric

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdaptiveRubrics:
    """The positive and negative rubrics selected for one prompt."""

    positive: tuple[Rubric, ...] = ()
    negative: tuple[Rubric, ...] = ()

    @property
    def all(self) -> tuple[Rubric, ...]:
        return self.positive + self.negative


@dataclass(frozen=True, slots=True)
class _Candidate:
    rubric: Rubric
    deviation: float


class RubricCache:
    """An isolated in-memory cache of discriminative adaptive rubrics.

    The caller controls the cache lifetime by retaining this object. Nothing is
    written to disk and no state is shared between environments implicitly.
    """

    def __init__(self, *, max_per_polarity: int = 3) -> None:
        if (
            isinstance(max_per_polarity, bool)
            or not isinstance(max_per_polarity, int)
            or max_per_polarity < 1
        ):
            raise ValueError("max_per_polarity must be positive")
        self.max_per_polarity = max_per_polarity
        self._entries: dict[str, dict[RubricPolarity, dict[str, _Candidate]]] = {}

    def get(self, prompt: str) -> AdaptiveRubrics:
        """Return the selected rubrics for ``prompt``."""

        entry = self._entries.get(prompt_key(prompt))
        if entry is None:
            return AdaptiveRubrics()
        return AdaptiveRubrics(
            positive=self._selected(entry["positive"]),
            negative=self._selected(entry["negative"]),
        )

    def consider(
        self,
        prompt: str,
        rubric: Rubric,
        scores: tuple[float, ...] | list[float],
    ) -> bool:
        """Consider a rubric, retaining it only when scores vary.

        Returns whether the rubric was retained after applying the cache limit.
        Rubric titles identify candidates within each polarity.
        """

        numeric_scores = tuple(float(score) for score in scores)
        if any(not math.isfinite(score) for score in numeric_scores):
            raise ValueError("adaptive rubric scores must be finite")
        if len(numeric_scores) < 2 or len(set(numeric_scores)) < 2:
            return False
        deviation = float(statistics.pstdev(numeric_scores))
        key = prompt_key(prompt)
        entry = self._entries.setdefault(key, {"positive": {}, "negative": {}})
        candidates = entry[rubric.polarity]
        candidates[rubric.title] = _Candidate(rubric, deviation)
        retained_titles = {
            candidate.rubric.title
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: (-candidate.deviation, candidate.rubric.title),
            )[: self.max_per_polarity]
        }
        for title in tuple(candidates):
            if title not in retained_titles:
                del candidates[title]
        return rubric.title in retained_titles

    @staticmethod
    def _selected(candidates: dict[str, _Candidate]) -> tuple[Rubric, ...]:
        return tuple(
            candidate.rubric
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: (-candidate.deviation, candidate.rubric.title),
            )
        )


def prompt_key(prompt: str) -> str:
    """Hash prompts so cache internals do not retain potentially sensitive text."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


async def generate_adaptive_rubrics(
    *,
    question: str,
    ground_truth: str,
    responses: Sequence[str],
    judge: Judge,
    existing_rubrics: Sequence[Rubric] = (),
) -> AdaptiveRubrics:
    """Generate discriminative rubrics for a question and its responses."""

    prompt = build_adaptive_rubric_prompt(
        question=question,
        ground_truth=ground_truth,
        responses=responses,
        existing_rubrics=existing_rubrics,
    )

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
    existing_rubrics: Sequence[Rubric] = (),
) -> AdaptiveRubrics:
    """Generate rubrics and retain those that discriminate the responses."""

    nonempty = tuple(response.strip() for response in responses if response.strip())
    if len(nonempty) < 2:
        return cache.get(question)

    prompt_rubrics = _merge_rubrics(existing_rubrics, cache.get(question).all)
    generated = await generate_adaptive_rubrics(
        question=question,
        ground_truth=ground_truth,
        responses=nonempty,
        judge=judge,
        existing_rubrics=prompt_rubrics,
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


def _merge_rubrics(*groups: Sequence[Rubric]) -> tuple[Rubric, ...]:
    merged: dict[tuple[RubricPolarity, str], Rubric] = {}
    for group in groups:
        for rubric in group:
            merged[(rubric.polarity, rubric.title)] = rubric
    return tuple(merged.values())


def _parse_generated(
    payload: dict[str, object],
    field: str,
    polarity: RubricPolarity,
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
            raise ValueError(f"judge response {field}[{index}].description must be non-empty")
        rubrics.append(
            Rubric(
                title=title.strip(),
                description=description.strip(),
                polarity=polarity,
            )
        )
    return tuple(rubrics)
