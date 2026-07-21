"""Caller-owned storage for adaptive rubrics."""

from __future__ import annotations

import hashlib
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

    def format_for_prompt(self) -> str | None:
        """Describe these rubrics for a subsequent generation request."""

        if not self.all:
            return None
        sections: list[str] = []
        for heading, rubrics in (
            ("Positive rubrics", self.positive),
            ("Negative rubrics", self.negative),
        ):
            if rubrics:
                sections.append(heading + ":")
                sections.extend(
                    f"- {rubric.title}: {rubric.description}" for rubric in rubrics
                )
        return "\n".join(sections)


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
        if max_per_polarity < 1:
            raise ValueError("max_per_polarity must be positive")
        self.max_per_polarity = max_per_polarity
        self._entries: dict[
            str, dict[RubricPolarity, dict[str, _Candidate]]
        ] = {}

    @staticmethod
    def key_for_prompt(prompt: str) -> str:
        """Return a stable key for a prompt."""

        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> AdaptiveRubrics:
        """Return the selected rubrics for ``prompt``."""

        entry = self._entries.get(self.key_for_prompt(prompt))
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

        if len(scores) < 2 or len(set(scores)) < 2:
            return False
        deviation = float(statistics.pstdev(scores))
        key = self.key_for_prompt(prompt)
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

    def snapshot(self) -> dict[str, AdaptiveRubrics]:
        """Return an immutable view of all cache entries, keyed by prompt hash."""

        return {
            key: AdaptiveRubrics(
                positive=self._selected(entry["positive"]),
                negative=self._selected(entry["negative"]),
            )
            for key, entry in self._entries.items()
        }

    @staticmethod
    def _selected(candidates: dict[str, _Candidate]) -> tuple[Rubric, ...]:
        return tuple(
            candidate.rubric
            for candidate in sorted(
                candidates.values(),
                key=lambda candidate: (-candidate.deviation, candidate.rubric.title),
            )
        )
