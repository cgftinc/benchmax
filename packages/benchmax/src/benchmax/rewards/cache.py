"""Caller-owned storage for adaptive rubrics."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass

from .rubric import Rubric, RubricPolarity


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
        self._entries: dict[
            str, dict[RubricPolarity, dict[str, _Candidate]]
        ] = {}

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
        entry = self._entries.setdefault(
            key, {"positive": {}, "negative": {}}
        )
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
