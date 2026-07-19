"""Diversity-based reward scaling for group reward functions.

Clusters texts (e.g. attack strategies, response approaches) by similarity,
then returns divisors so that each rollout's reward can be scaled by its
cluster size. Rollouts with unique strategies keep full reward; duplicates
share it.

Supports three clustering methods:
- ``llm``: Send texts to an LLM that groups them by semantic tactic.
- ``ngram``: Fast, offline n-gram Jaccard similarity clustering.
- ``embedding``: (future) Cosine-similarity clustering via embeddings.

Usage::

    from benchmax.rewards.diversity import DiversityConfig, scale_by_diversity

    async def compute_group_rewards(self, rollouts):
        raw_rewards = [await self.compute_reward(rollout) for rollout in rollouts]
        texts = [rollout.messages[-1]["content"] for rollout in rollouts]
        scaled, cluster_info = await scale_by_diversity(
            rewards=raw_rewards,
            texts=texts,
            config=DiversityConfig(method="llm", model="...", base_url="..."),
            context=rollouts[0].example_args.get("behavior", ""),
        )
        return {
            rollout.rollout_id: reward
            for rollout, reward in zip(rollouts, scaled, strict=True)
        }
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from benchmax.auth import ModelAuth, ModelRequestContext
from benchmax.rubrics._utils import _extract_json, _resolve_judge_auth

logger = logging.getLogger(__name__)

# AsyncOpenAI imported lazily inside _cluster_by_llm to avoid pulling in
# unpicklable context vars at module level (breaks cloudpickle bundling).


DEFAULT_CLUSTER_PROMPT = """\
You are clustering text entries for a diversity reward.

Context:
{context}

Below are {n} text entries from independent rollouts. Each entry is a strategy
or approach trace.

Cluster by the **underlying tactic or approach**, not exact wording.
- Paraphrases and synonyms of the same tactic = same cluster.
- "academic framing" and "research framing" = same cluster.
- Empty entries, refusals, and "NO_TOOL_CALL" should share one "null" cluster.
- Every index from 0 to {max_idx} must appear exactly once.

{items}

Return only valid JSON in this exact shape:
{{"assignments": [{{"index": 0, "cluster_id": "tactic_name", "label": "short description"}}]}}"""


@dataclass
class DiversityConfig:
    """Configuration for diversity clustering.

    Set ``method`` to choose the clustering backend:
    - ``"llm"`` — requires ``model``, ``base_url``, and optionally ``api_key``.
    - ``"ngram"`` — fast offline clustering, no API calls.
    """

    method: Literal["llm", "ngram"] = "llm"

    # LLM clustering options
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    auth: ModelAuth | None = None
    prompt_template: str = DEFAULT_CLUSTER_PROMPT
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: float = 60.0

    # N-gram clustering options
    ngram_n: int = 3
    similarity_threshold: float = 0.5

    # LLM retry count (matches rubric.py default of 3)
    max_retries: int = 3

    # On clustering error, "unique" means every rollout is its own cluster
    # (no penalty). "uniform" means all rollouts share one cluster.
    fallback_on_error: Literal["unique", "uniform"] = "unique"


@dataclass
class ClusterResult:
    """Result of clustering a list of texts."""

    cluster_ids: List[str]
    divisors: List[float]
    labels: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    @property
    def n_clusters(self) -> int:
        return len(set(self.cluster_ids))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ngram_set(text: str, n: int) -> set[str]:
    """Return the set of character n-grams for a text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_by_ngram(texts: List[str], n: int, threshold: float) -> ClusterResult:
    """Greedy single-linkage clustering using n-gram Jaccard similarity.

    Note: single-linkage can chain clusters — if A~B and B~C but not A~C,
    all three end up in the same cluster. This is fast but aggressive.
    """
    ngrams = [_ngram_set(t, n) for t in texts]
    cluster_ids: list[int] = [-1] * len(texts)
    next_cluster = 0

    for i in range(len(texts)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_cluster
        for j in range(i + 1, len(texts)):
            if cluster_ids[j] != -1:
                continue
            if _jaccard(ngrams[i], ngrams[j]) >= threshold:
                cluster_ids[j] = next_cluster
        next_cluster += 1

    str_ids = [str(c) for c in cluster_ids]
    counts = Counter(str_ids)
    divisors = [float(counts[cid]) for cid in str_ids]
    return ClusterResult(cluster_ids=str_ids, divisors=divisors)


async def _cluster_by_llm(
    texts: List[str],
    config: DiversityConfig,
    context: str,
) -> ClusterResult:
    """Cluster texts by sending them to an LLM."""
    from openai import AsyncOpenAI

    items = "\n\n".join(f"[{i}]\n{t}" for i, t in enumerate(texts))
    prompt = config.prompt_template.format(
        context=context or "(none)",
        n=len(texts),
        max_idx=len(texts) - 1,
        items=items,
    )

    resolved_auth = _resolve_judge_auth(config.auth, config.api_key, None)
    headers = await resolved_auth.headers_for_request(
        ModelRequestContext(
            base_url=config.base_url,
            model=config.model,
            rollout_id="diversity-judge",
        )
    )
    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key="benchmax-runtime-auth",
        default_headers=dict(headers),
        max_retries=config.max_retries,
    )
    try:
        resp = await client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
        )
    finally:
        await client.close()

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _extract_json(raw)
    assignments = parsed.get("assignments", [])

    # Build cluster_ids, ensuring every index is covered
    cluster_map: dict[int, str] = {}
    label_map: dict[int, str] = {}
    for a in assignments:
        idx = int(a["index"])
        cluster_map[idx] = str(a.get("cluster_id", f"unknown_{idx}"))
        label_map[idx] = str(a.get("label", ""))

    cluster_ids = []
    labels = []
    for i in range(len(texts)):
        cluster_ids.append(cluster_map.get(i, f"unmapped_{i}"))
        labels.append(label_map.get(i, ""))

    counts = Counter(cluster_ids)
    divisors = [float(counts[cid]) for cid in cluster_ids]
    return ClusterResult(cluster_ids=cluster_ids, divisors=divisors, labels=labels, raw_response=raw)


def _fallback_result(n: int, mode: Literal["unique", "uniform"]) -> ClusterResult:
    """Generate a fallback ClusterResult on error."""
    if mode == "unique":
        ids = [f"fallback_{i}" for i in range(n)]
        return ClusterResult(cluster_ids=ids, divisors=[1.0] * n)
    else:
        return ClusterResult(cluster_ids=["all"] * n, divisors=[float(n)] * n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def cluster_texts(
    texts: List[str],
    config: DiversityConfig,
    *,
    context: str = "",
) -> ClusterResult:
    """Cluster a list of texts by similarity.

    Args:
        texts: The strings to cluster (e.g. strategy traces, response summaries).
        config: Clustering configuration (method, model, thresholds, etc.).
        context: Optional context for the clustering prompt (e.g. the goal/behavior).

    Returns:
        A ``ClusterResult`` with cluster assignments and per-item divisors.
    """
    if not texts:
        return ClusterResult(cluster_ids=[], divisors=[])
    if len(texts) == 1:
        return ClusterResult(cluster_ids=["0"], divisors=[1.0])

    # Validate config up front — these are caller bugs, not transient failures.
    if config.method == "llm":
        if not config.model or not config.base_url:
            raise ValueError("LLM clustering requires 'model' and 'base_url' in DiversityConfig")
    elif config.method != "ngram":
        raise ValueError(f"Unknown clustering method: {config.method!r}")

    try:
        if config.method == "ngram":
            return _cluster_by_ngram(texts, config.ngram_n, config.similarity_threshold)
        return await _cluster_by_llm(texts, config, context)
    except RuntimeError:
        # Missing explicit auth is not transient —
        # propagate so the caller (and training) fails loudly rather than
        # silently producing un-scaled rewards for an entire run.
        raise
    except Exception as e:
        logger.warning(
            "Clustering failed (%s: %s), using fallback=%s",
            type(e).__name__, e, config.fallback_on_error,
        )
        return _fallback_result(len(texts), config.fallback_on_error)


async def scale_by_diversity(
    rewards: List[Dict[str, float]],
    texts: List[str],
    config: DiversityConfig,
    *,
    context: str = "",
) -> tuple[List[Dict[str, float]], ClusterResult]:
    """Cluster texts and divide each rollout's rewards by its cluster size.

    This is the primary entry point for diversity-scaled group rewards.

    Args:
        rewards: Per-rollout reward dicts (e.g. from ``compute_reward`` or ``_score_one``).
        texts: Per-rollout texts to cluster (e.g. strategy traces).
        config: Clustering configuration.
        context: Optional context for clustering (e.g. the behavior/goal).

    Returns:
        A tuple of ``(scaled_rewards, cluster_result)`` where ``scaled_rewards``
        is a list of reward dicts with every value divided by the cluster divisor,
        and ``cluster_result`` contains the cluster assignments for observability.
    """
    if len(rewards) != len(texts):
        raise ValueError(f"rewards ({len(rewards)}) and texts ({len(texts)}) must have same length")

    result = await cluster_texts(texts, config, context=context)

    scaled_rewards: List[Dict[str, float]] = []
    for reward, divisor in zip(rewards, result.divisors):
        d = max(divisor, 1.0)
        scaled_rewards.append({k: v / d for k, v in reward.items()})

    return scaled_rewards, result
