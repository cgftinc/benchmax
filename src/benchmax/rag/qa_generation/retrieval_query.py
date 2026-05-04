"""Shared retrieval-query derivation for BM25-focused evaluation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

from benchmax.envs.postgres_search.query_rewriter import heuristic_query_rewrite
from benchmax.rag.qa_generation.style_controls import (
    QUERY_STYLE_EXPERT,
    QUERY_STYLE_KEYS,
    QUERY_STYLE_NATURAL,
    classify_query_style,
)


@dataclass
class QueryRewriteConfig:
    """Configuration for deriving retrieval-friendly query text."""

    enabled: bool = True
    max_terms: int = 16
    max_chars: int = 140
    apply_to_all_rows: bool = True


def resolve_retrieval_query(
    point: MutableMapping[str, Any],
    *,
    rewrite_cfg: QueryRewriteConfig,
    persist: bool = True,
) -> str:
    """Resolve retrieval query with explicit-field precedence and lazy derivation.

    Resolution order:
    1) Existing `retrieval_query` when non-empty.
    2) Derived rewrite from `user_question` / `question`.
    3) Raw `user_question` / `question` fallback.
    """
    explicit_query = str(point.get("retrieval_query", "")).strip()
    if explicit_query:
        if persist:
            point["retrieval_query"] = explicit_query
        return explicit_query

    user_question = (
        str(point.get("user_question", "")).strip() or str(point.get("question", "")).strip()
    )
    if persist:
        point["user_question"] = user_question

    if not user_question:
        if persist:
            point["retrieval_query"] = ""
        return ""

    should_rewrite = bool(rewrite_cfg.enabled)
    if should_rewrite and not rewrite_cfg.apply_to_all_rows:
        observed = str((point.get("eval_scores") or {}).get("query_style_observed", "")).strip()
        if observed not in QUERY_STYLE_KEYS:
            observed = classify_query_style(user_question)
        should_rewrite = observed in {QUERY_STYLE_NATURAL, QUERY_STYLE_EXPERT}

    if should_rewrite:
        rewritten = heuristic_query_rewrite(
            user_question,
            max_terms=rewrite_cfg.max_terms,
            max_chars=rewrite_cfg.max_chars,
        )
        resolved = rewritten or user_question
    else:
        resolved = user_question

    if persist:
        point["retrieval_query"] = resolved
    return resolved
