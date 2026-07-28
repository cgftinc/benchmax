"""Diversity-aware scaling for sibling rewards."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .judge import Judge, JudgeError
from .prompts import DEFAULT_DIVERSITY_INSTRUCTIONS, build_diversity_prompt

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class NgramDiversityConfig:
    """Offline character n-gram clustering configuration."""

    n: int = 3
    similarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class LLMDiversityConfig:
    """Judge-backed semantic clustering configuration."""

    judge: Judge
    instructions: str = DEFAULT_DIVERSITY_INSTRUCTIONS
    max_tokens: int = 512
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be non-empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


type DiversityConfig = NgramDiversityConfig | LLMDiversityConfig


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Cluster assignments and their derived reward divisors."""

    cluster_ids: tuple[str, ...]
    divisors: tuple[float, ...]
    labels: tuple[str, ...] = ()
    raw_response: str | None = None

    @property
    def n_clusters(self) -> int:
        return len(set(self.cluster_ids))


async def cluster_texts(
    texts: Sequence[str],
    config: DiversityConfig,
    *,
    context: str = "",
) -> ClusterResult:
    """Cluster texts using the selected explicit backend."""

    clean_texts = tuple(str(text) for text in texts)
    if not clean_texts:
        return ClusterResult(cluster_ids=(), divisors=())
    if len(clean_texts) == 1:
        return ClusterResult(cluster_ids=("0",), divisors=(1.0,))
    if isinstance(config, NgramDiversityConfig):
        return _cluster_by_ngram(clean_texts, config)
    return await _cluster_by_llm(clean_texts, config, context)


async def scale_by_diversity(
    rewards: Sequence[Mapping[str, float]],
    texts: Sequence[str],
    config: DiversityConfig,
    *,
    context: str = "",
) -> tuple[list[dict[str, float]], ClusterResult]:
    """Divide every reward component by its text's cluster size."""

    if len(rewards) != len(texts):
        raise ValueError("rewards and texts must have the same length")
    result = await cluster_texts(texts, config, context=context)
    scaled = [
        {key: value / divisor for key, value in reward.items()}
        for reward, divisor in zip(rewards, result.divisors, strict=True)
    ]
    return scaled, result


def _cluster_by_ngram(
    texts: tuple[str, ...], config: NgramDiversityConfig
) -> ClusterResult:
    ngrams = tuple(_ngram_set(text, config.n) for text in texts)
    parents = list(range(len(texts)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if _jaccard(ngrams[left], ngrams[right]) >= config.similarity_threshold:
                union(left, right)

    roots = [find(index) for index in range(len(texts))]
    root_ids = {root: str(index) for index, root in enumerate(dict.fromkeys(roots))}
    cluster_ids = tuple(root_ids[root] for root in roots)
    counts = Counter(cluster_ids)
    return ClusterResult(
        cluster_ids=cluster_ids,
        divisors=tuple(float(counts[cluster_id]) for cluster_id in cluster_ids),
    )


async def _cluster_by_llm(
    texts: tuple[str, ...], config: LLMDiversityConfig, context: str
) -> ClusterResult:
    prompt = build_diversity_prompt(
        texts,
        context=context,
        instructions=config.instructions,
    )

    try:
        payload, raw = await config.judge.request_json(
            prompt,
            request_id="diversity-clustering",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        cluster_ids, labels = _parse_assignments(payload, len(texts))
    except JudgeError:
        raise
    except Exception as error:
        logger.exception("LLM diversity clustering failed")
        raise JudgeError(f"diversity clustering failed: {error}") from error

    counts = Counter(cluster_ids)
    return ClusterResult(
        cluster_ids=cluster_ids,
        divisors=tuple(float(counts[cluster_id]) for cluster_id in cluster_ids),
        labels=labels,
        raw_response=raw,
    )


def _parse_assignments(
    payload: Mapping[str, object], item_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("judge response field 'assignments' must be a list")
    assignments: dict[int, tuple[str, str]] = {}
    for assignment in raw_assignments:
        if not isinstance(assignment, dict):
            raise ValueError("judge diversity assignments must be objects")
        index = assignment.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("judge diversity assignment indices must be integers")
        if not 0 <= index < item_count:
            raise ValueError(f"judge diversity index {index} is out of range")
        if index in assignments:
            raise ValueError(f"judge diversity assignment repeated index {index}")
        cluster_id = assignment.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            raise ValueError(f"judge diversity assignment {index} needs a cluster_id")
        label = assignment.get("label", "")
        if not isinstance(label, str):
            raise ValueError(f"judge diversity assignment {index} label must be a string")
        assignments[index] = (cluster_id.strip(), label.strip())
    missing = sorted(set(range(item_count)) - assignments.keys())
    if missing:
        raise ValueError(f"judge diversity assignments omitted indices: {missing}")
    return (
        tuple(assignments[index][0] for index in range(item_count)),
        tuple(assignments[index][1] for index in range(item_count)),
    )


def _ngram_set(text: str, n: int) -> set[str]:
    clean = text.lower().strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[index : index + n] for index in range(len(clean) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    score = len(left & right) / len(left | right)
    if not math.isfinite(score):
        raise RuntimeError("computed a non-finite Jaccard similarity")
    return score
