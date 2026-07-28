"""Rubric definitions and judge-backed evaluation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .judge import Judge, JudgeError
from .prompts import (
    build_rubric_evaluation_prompt,
    build_rubric_ranking_prompt,
)

logger = logging.getLogger(__name__)

RubricPolarity = Literal["positive", "negative"]


@dataclass(frozen=True, slots=True)
class Rubric:
    """A criterion evaluated by a judge.

    A positive rubric rewards demonstrating the criterion. A negative rubric
    rewards avoiding it. ``score_map`` defines the judge's allowed raw scores;
    omitted maps default to binary ``0``/``1`` scoring.
    """

    title: str
    description: str
    polarity: RubricPolarity = "positive"
    score_map: Mapping[float, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("rubric title must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("rubric description must be non-empty")
        if self.polarity not in {"positive", "negative"}:
            raise ValueError("rubric polarity must be 'positive' or 'negative'")
        if self.score_map is not None:
            if len(self.score_map) < 2:
                raise ValueError("rubric score_map must contain at least two scores")
            for score, description in self.score_map.items():
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise TypeError("rubric score_map keys must be numeric")
                if not math.isfinite(float(score)):
                    raise ValueError("rubric score_map keys must be finite")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError("rubric score_map descriptions must be non-empty")

    @property
    def allowed_scores(self) -> tuple[float, ...]:
        scores = self.score_map.keys() if self.score_map is not None else (0.0, 1.0)
        return tuple(sorted(float(score) for score in scores))

    def reward_for(self, raw_score: float) -> float:
        """Normalize an allowed raw score into a reward in ``[0, 1]``."""

        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError("raw_score must be numeric")
        numeric = float(raw_score)
        if not math.isfinite(numeric) or numeric not in self.allowed_scores:
            raise ValueError(f"raw_score must be one of {list(self.allowed_scores)}")
        low, high = self.allowed_scores[0], self.allowed_scores[-1]
        normalized = (numeric - low) / (high - low)
        return normalized if self.polarity == "positive" else 1.0 - normalized


@dataclass(frozen=True, slots=True)
class RankingAnchor:
    """A reference response with a known score on the ranking scale."""

    response: str
    score: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, str) or not self.response.strip():
            raise ValueError("ranking anchor response must be non-empty")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
            or not 0 <= self.score <= 1
        ):
            raise ValueError("ranking anchor score must be within [0, 1]")
        if self.label is not None and (
            not isinstance(self.label, str) or not self.label.strip()
        ):
            raise ValueError("ranking anchor label must be non-empty or None")


@dataclass(frozen=True, slots=True)
class RubricEvaluation:
    score: float
    reasoning: str
    raw_response: str


@dataclass(frozen=True, slots=True)
class RubricRanking:
    scores: tuple[float, ...]
    ranking: tuple[tuple[int, ...], ...]
    reasoning: str
    raw_response: str


async def evaluate_single_rubric(
    rubric: Rubric,
    *,
    question: str,
    response: str,
    judge: Judge,
    ground_truth: str | None = None,
    log_result: bool = True,
) -> RubricEvaluation:
    """Evaluate one response against one rubric."""

    prompt = build_rubric_evaluation_prompt(
        rubric,
        question=question,
        response=response,
        ground_truth=ground_truth,
    )
    try:
        payload, raw = await judge.request_json(
            prompt,
            request_id=f"rubric:{_slug(rubric.title)}",
        )
        score = _required_allowed_score(payload, rubric)
        reasoning = _optional_reasoning(payload)
    except Exception as error:
        logger.exception("Rubric evaluation failed for %r", rubric.title)
        raise JudgeError(f"rubric {rubric.title!r} evaluation failed: {error}") from error

    result = RubricEvaluation(score=score, reasoning=reasoning, raw_response=raw)
    if log_result:
        logger.info(
            "rubric.evaluated title=%r polarity=%s score=%s reasoning=%s",
            rubric.title,
            rubric.polarity,
            result.score,
            result.reasoning,
        )
    return result


async def evaluate_rubric_ranking(
    rubric: Rubric,
    *,
    question: str,
    responses: Sequence[str],
    judge: Judge,
    ground_truth: str | None = None,
    anchors: Sequence[RankingAnchor] = (),
    below_ground_truth_ceiling: float = 0.3,
    log_result: bool = True,
) -> RubricRanking:
    """Rank responses and convert their positions into scores in ``[0, 1]``."""

    if not 0 <= below_ground_truth_ceiling <= 1:
        raise ValueError("below_ground_truth_ceiling must be within [0, 1]")
    ranking_anchors = tuple(anchors)
    if any(not isinstance(anchor, RankingAnchor) for anchor in ranking_anchors):
        raise TypeError("anchors must contain RankingAnchor values")
    if any(
        left.score > right.score
        for left, right in zip(ranking_anchors, ranking_anchors[1:])
    ):
        raise ValueError("ranking anchors must be ordered from worst to best score")
    if ranking_anchors and ground_truth and ground_truth.strip():
        raise ValueError("pass ground_truth or anchors, not both")

    output_scores = [0.0] * len(responses)
    nonempty = [
        (index, response.strip())
        for index, response in enumerate(responses)
        if response and response.strip()
    ]
    if not nonempty:
        return RubricRanking(
            scores=tuple(output_scores),
            ranking=(),
            reasoning="All responses are empty",
            raw_response="",
        )

    use_ground_truth = bool(ground_truth and ground_truth.strip())
    if len(nonempty) == 1 and not use_ground_truth and not ranking_anchors:
        output_scores[nonempty[0][0]] = 1.0
        return RubricRanking(
            scores=tuple(output_scores),
            ranking=((0,),),
            reasoning="Only one non-empty response",
            raw_response="",
        )

    judged_items = [response for _, response in nonempty]
    ground_truth_index: int | None = None
    if use_ground_truth:
        ground_truth_index = len(judged_items)
        judged_items.append(str(ground_truth).strip())

    anchor_indices: list[int] = []
    for anchor in ranking_anchors:
        anchor_indices.append(len(judged_items))
        judged_items.append(anchor.response.strip())

    prompt = build_rubric_ranking_prompt(
        rubric,
        question=question,
        responses=judged_items,
    )
    try:
        payload, raw = await judge.request_json(
            prompt,
            request_id=f"rubric-ranking:{_slug(rubric.title)}",
        )
        ranking = _parse_ranking(payload, len(judged_items))
        positions = _ranking_positions(ranking)
        max_position = float(len(judged_items) - 1)

        if ranking_anchors:
            seams = _monotonic_seams(
                sorted(
                    zip(
                        (positions[index] for index in anchor_indices),
                        (anchor.score for anchor in ranking_anchors),
                        strict=True,
                    )
                )
            )
            local_scores = [
                _band_score(positions[index], seams, max_position)
                for index in range(len(nonempty))
            ]
        elif ground_truth_index is not None:
            ground_truth_position = positions[ground_truth_index]
            local_scores = [
                _ground_truth_score(
                    positions[index],
                    ground_truth_position,
                    max_position,
                    below_ground_truth_ceiling,
                )
                for index in range(len(nonempty))
            ]
        else:
            local_scores = [
                1.0 - positions[index] / max_position
                if max_position > 0
                else 1.0
                for index in range(len(nonempty))
            ]
        reasoning = _optional_reasoning(payload)
    except Exception as error:
        logger.exception("Rubric ranking failed for %r", rubric.title)
        raise JudgeError(f"rubric {rubric.title!r} ranking failed: {error}") from error

    for (original_index, _), score in zip(nonempty, local_scores, strict=True):
        output_scores[original_index] = _clip01(score)

    result = RubricRanking(
        scores=tuple(output_scores),
        ranking=ranking,
        reasoning=reasoning,
        raw_response=raw,
    )
    if log_result:
        logger.info(
            "rubric.ranked title=%r polarity=%s ranking=%s scores=%s "
            "anchors=%s reasoning=%s",
            rubric.title,
            rubric.polarity,
            result.ranking,
            result.scores,
            tuple(
                (anchor.label or f"anchor@{anchor.score:g}", anchor.score)
                for anchor in ranking_anchors
            ),
            result.reasoning,
        )
    return result


def _required_allowed_score(payload: Mapping[str, object], rubric: Rubric) -> float:
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("judge response must contain a numeric score")
    numeric = float(score)
    if not math.isfinite(numeric):
        raise ValueError("judge score must be finite")
    if numeric not in rubric.allowed_scores:
        raise ValueError(
            f"judge score {score!r} is not one of {list(rubric.allowed_scores)}"
        )
    return numeric


def _optional_reasoning(payload: Mapping[str, object]) -> str:
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("judge reasoning must be a string when provided")
    return reasoning


def _parse_ranking(
    payload: Mapping[str, object], item_count: int
) -> tuple[tuple[int, ...], ...]:
    raw_ranking = payload.get("ranking")
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise ValueError("judge response must contain a non-empty ranking")

    ranking: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for raw_tier in raw_ranking:
        if not isinstance(raw_tier, list) or not raw_tier:
            raise ValueError("judge ranking tiers must be non-empty lists")
        tier: list[int] = []
        for index in raw_tier:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("judge ranking indices must be integers")
            if not 0 <= index < item_count:
                raise ValueError(
                    f"judge ranking index {index} is outside 0..{item_count - 1}"
                )
            if index in seen:
                raise ValueError(f"judge ranking repeated response index {index}")
            seen.add(index)
            tier.append(index)
        ranking.append(tuple(tier))

    missing = sorted(set(range(item_count)) - seen)
    if missing:
        raise ValueError(f"judge ranking omitted response indices: {missing}")
    return tuple(ranking)


def _ranking_positions(ranking: Sequence[Sequence[int]]) -> dict[int, float]:
    positions: dict[int, float] = {}
    offset = 0
    for tier in ranking:
        midpoint = offset + (len(tier) - 1) / 2
        positions.update({index: midpoint for index in tier})
        offset += len(tier)
    return positions


def _monotonic_seams(
    seams: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for position, edge in seams:
        while output and edge > output[-1][1]:
            previous_position, previous_edge = output.pop()
            position = previous_position
            edge = max(previous_edge, edge)
        output.append((position, edge))
    return output


def _band_score(
    position: float,
    seams: Sequence[tuple[float, float]],
    max_position: float,
) -> float:
    best_position, best_edge = seams[0]
    worst_position, worst_edge = seams[-1]
    if position <= best_position:
        if best_position == 0:
            return best_edge
        return best_edge + (1 - best_edge) * (
            (best_position - position) / best_position
        )
    if position >= worst_position:
        span = max_position - worst_position
        return worst_edge * ((max_position - position) / span if span > 0 else 0)
    for (left_position, left_edge), (right_position, right_edge) in zip(
        seams, seams[1:]
    ):
        if left_position <= position <= right_position:
            span = right_position - left_position
            if span == 0:
                return max(left_edge, right_edge)
            return right_edge + (left_edge - right_edge) * (
                (right_position - position) / span
            )
    raise RuntimeError("position did not fall within anchor seams")


def _ground_truth_score(
    position: float,
    ground_truth_position: float,
    max_position: float,
    below_ceiling: float,
) -> float:
    if position < ground_truth_position:
        if ground_truth_position == 0:
            return 0.5
        return 0.5 + 0.5 * (
            (ground_truth_position - position) / ground_truth_position
        )
    if position == ground_truth_position:
        return 0.5
    span = max_position - ground_truth_position
    if span <= 0:
        return below_ceiling
    return below_ceiling * (1 - (position - ground_truth_position) / span)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _slug(value: str) -> str:
    return "-".join(value.lower().split())
