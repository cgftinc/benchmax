"""Reward primitives for benchmax environments."""

from .adaptive import (
    AdaptiveRubrics,
    RubricCache,
    generate_adaptive_rubrics,
    generate_and_cache_adaptive_rubrics,
)
from .deterministic import (
    Completion,
    citation_score,
    clip01,
    count_search_calls,
    extract_answer_block,
    extract_completion_text,
    overlap_reward,
    percent_of_text_a_in_text_b,
    search_within_budget,
    tool_call_efficiency,
)
from .diversity import (
    ClusterResult,
    DiversityConfig,
    LLMDiversityConfig,
    NgramDiversityConfig,
    cluster_texts,
    scale_by_diversity,
)
from .judge import Judge, JudgeError
from .rubric import (
    RankingAnchor,
    Rubric,
    RubricEvaluation,
    RubricPolarity,
    RubricRanking,
    evaluate_rubric_ranking,
    evaluate_single_rubric,
)
from .scoring import (
    rank_group_rubrics,
    rubric_reward_key,
    score_group_rubrics,
    score_rubrics,
)

__all__ = [
    "AdaptiveRubrics",
    "ClusterResult",
    "Completion",
    "DiversityConfig",
    "Judge",
    "JudgeError",
    "LLMDiversityConfig",
    "NgramDiversityConfig",
    "RankingAnchor",
    "Rubric",
    "RubricCache",
    "RubricEvaluation",
    "RubricPolarity",
    "RubricRanking",
    "citation_score",
    "clip01",
    "cluster_texts",
    "count_search_calls",
    "evaluate_rubric_ranking",
    "evaluate_single_rubric",
    "extract_answer_block",
    "extract_completion_text",
    "generate_adaptive_rubrics",
    "generate_and_cache_adaptive_rubrics",
    "overlap_reward",
    "percent_of_text_a_in_text_b",
    "rank_group_rubrics",
    "rubric_reward_key",
    "scale_by_diversity",
    "score_group_rubrics",
    "score_rubrics",
    "search_within_budget",
    "tool_call_efficiency",
]
