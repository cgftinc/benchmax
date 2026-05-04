"""Shared style-control utilities for QA generation pipelines.

This module centralizes style constants, distribution math, style
classification, and metadata helpers so multiple generators/pipelines can
reuse one implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from benchmax.rag.qa_generation.generated_qa import GeneratedQA

QUERY_STYLE_KEYWORD = "keyword"
QUERY_STYLE_NATURAL = "natural"
QUERY_STYLE_EXPERT = "expert"
QUERY_STYLE_KEYS = [QUERY_STYLE_KEYWORD, QUERY_STYLE_NATURAL, QUERY_STYLE_EXPERT]

DEFAULT_QUERY_STYLE_DISTRIBUTION = {
    QUERY_STYLE_KEYWORD: 0.6,
    QUERY_STYLE_NATURAL: 0.3,
    QUERY_STYLE_EXPERT: 0.1,
}

QA_TYPE_STYLE_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "lookup": {
        QUERY_STYLE_KEYWORD: 0.35,
        QUERY_STYLE_NATURAL: 0.45,
        QUERY_STYLE_EXPERT: 0.20,
    },
    "multi_hop": {
        QUERY_STYLE_KEYWORD: 0.10,
        QUERY_STYLE_NATURAL: 0.60,
        QUERY_STYLE_EXPERT: 0.30,
    },
}


def get_style_distribution(qa_type: str) -> dict[str, float]:
    """Return style distribution for a given QA type, or DEFAULT_QUERY_STYLE_DISTRIBUTION."""
    return QA_TYPE_STYLE_DISTRIBUTIONS.get(qa_type, DEFAULT_QUERY_STYLE_DISTRIBUTION)


@dataclass
class StyleControlConfig:
    """Shared style-control configuration."""

    enabled: bool = True
    distribution: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_QUERY_STYLE_DISTRIBUTION)
    )


def zero_style_counts() -> dict[str, int]:
    """Create an empty style-count map."""
    return {key: 0 for key in QUERY_STYLE_KEYS}


def normalize_style_distribution(raw_distribution: Any) -> dict[str, float]:
    """Normalize an arbitrary style distribution into a probability map."""
    dist = dict(DEFAULT_QUERY_STYLE_DISTRIBUTION)
    if isinstance(raw_distribution, dict):
        for key in QUERY_STYLE_KEYS:
            if key in raw_distribution:
                try:
                    value = float(raw_distribution[key])
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    dist[key] = value

    total = sum(dist.values())
    if total <= 0:
        dist = dict(DEFAULT_QUERY_STYLE_DISTRIBUTION)
        total = sum(dist.values())
    return {key: dist[key] / total for key in QUERY_STYLE_KEYS}


def allocate_largest_remainder(
    total: int,
    distribution: Mapping[str, float],
) -> dict[str, int]:
    """Allocate integer style counts with largest-remainder rounding."""
    if total <= 0:
        return zero_style_counts()

    normalized = normalize_style_distribution(dict(distribution))
    raw = {key: normalized[key] * total for key in QUERY_STYLE_KEYS}
    base = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        fractional = sorted(
            QUERY_STYLE_KEYS,
            key=lambda key: (raw[key] - base[key], normalized[key]),
            reverse=True,
        )
        for key in fractional[:remainder]:
            base[key] += 1
    return base


def style_sequence_from_counts(style_counts: Mapping[str, int]) -> list[str]:
    """Expand counts into a deterministic ordered style sequence."""
    sequence: list[str] = []
    for style in QUERY_STYLE_KEYS:
        sequence.extend([style] * max(0, int(style_counts.get(style, 0))))
    return sequence


def classify_query_style(question: str, language: str = "") -> str:
    """Classify a question into keyword, natural, or expert style.

    When *language* is provided, uses language-appropriate tokenization and
    interrogative detection so that non-Latin scripts are not misclassified
    as keyword style.
    """
    text = question.strip()
    lower = text.lower()

    # Unicode-aware tokenization: split on whitespace, then count.
    # For CJK/Korean where spaces are optional, also count by characters.
    tokens = re.findall(r"[\w$.:/-]+", lower, re.UNICODE)
    token_count = len(tokens)

    if token_count == 0:
        return QUERY_STYLE_KEYWORD

    starts_interrogative = lower.startswith(
        (
            "how ",
            "what ",
            "why ",
            "when ",
            "where ",
            "who ",
            "which ",
            "can ",
            "should ",
            "does ",
            "is ",
            "are ",
            "do ",
            "did ",
        )
    )
    has_question_mark = "?" in text

    # Korean interrogative patterns (어떻게/무엇/왜/언제/어디/누가/~인가/~나요 etc.)
    _lang = language.lower()
    if _lang in ("korean", "ko") or (not _lang and re.search(r"[\uac00-\ud7a3]", text)):
        kr_interrogative = bool(
            re.search(
                r"(어떻게|무엇|무슨|왜|언제|어디|누가|어느|어떤"
                r"|인가요|ᆫ가요|나요|습니까|ᆯ까요|할까|는지|할지)",
                text,
            )
        )
        starts_interrogative = starts_interrogative or kr_interrogative

    # Japanese interrogative patterns (何/どう/なぜ/いつ/どこ/誰/か？ etc.)
    if _lang in ("japanese", "ja") or (
        not _lang and re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text)
    ):
        jp_interrogative = bool(
            re.search(r"(何|どう|なぜ|いつ|どこ|誰|どの|どんな|ですか|ますか)", text)
        )
        starts_interrogative = starts_interrogative or jp_interrogative
    expert_terms = {
        "troubleshoot",
        "debug",
        "error",
        "failed",
        "failure",
        "timeout",
        "mismatch",
        "incident",
        "fix",
        "compare",
        "comparison",
        "tradeoff",
        "versus",
        "vs",
        "difference",
        "why",
        "root",
        "cause",
        "optimize",
        "prerequisite",
    }
    expert_hits = sum(1 for token in tokens if token in expert_terms)
    has_expert_pattern = bool(
        re.search(
            r"\b(vs|versus|root cause|best practice|why .* fail|how to fix|troubleshoot)\b",
            lower,
        )
    )

    if expert_hits >= 2 or (has_expert_pattern and token_count >= 5):
        return QUERY_STYLE_EXPERT
    if has_question_mark or starts_interrogative or token_count >= 9:
        return QUERY_STYLE_NATURAL
    if token_count <= 7 and lower == text:
        return QUERY_STYLE_KEYWORD
    return QUERY_STYLE_NATURAL


def get_item_question(item: GeneratedQA) -> str:
    """Return the normalized question string from a GeneratedQA item."""
    return str(item.qa.get("question", "")).strip()


def set_item_style_observed(item: GeneratedQA, observed: str) -> None:
    """Persist observed style in both generation metadata and eval_scores."""
    if observed not in QUERY_STYLE_KEYS:
        return
    item.generation_metadata["query_style_observed"] = observed
    eval_scores = cast(dict[str, Any], dict(item.qa.get("eval_scores", {}) or {}))
    eval_scores["query_style_observed"] = observed
    item.qa["eval_scores"] = eval_scores


def annotate_item_style_observed(item: GeneratedQA) -> str:
    """Classify and store observed question style for the item."""
    observed = classify_query_style(get_item_question(item))
    set_item_style_observed(item, observed)
    return observed
