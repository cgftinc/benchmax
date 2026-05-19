"""Deterministic quality guards for Pipeline."""

from __future__ import annotations

import re
from typing import Any

from benchmax.rag.qa_generation.pipeline_config import PipelineContext, DeterministicGuardsConfig
from benchmax.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA

_STYLE_LABELS = frozenset({"natural", "keyword", "expert"})

# Languages where characters encode more meaning — answer length thresholds
# are divided by 3 for these to avoid over-rejecting concise but complete answers.
_HIGH_DENSITY_LANGUAGES = frozenset({
    "korean", "ko", "chinese", "zh", "japanese", "ja",
})

# Patterns indicating a meta-question about generation failure rather than a real domain question
_META_QUESTION_PATTERNS = [
    re.compile(
        r"\b(can'?t|cannot|unable to|not possible to)\s+(create|generate|form|construct)", re.I
    ),
    re.compile(r"\bno valid\s+(multi[- ]?hop\s+)?question", re.I),
    re.compile(r"\b(why\s+)?(can'?t|cannot)\s+I\s+(create|generate|form)", re.I),
    re.compile(r"\bchunks?\s+(do not|don'?t|lack|cannot)\s+(support|provide|contain)", re.I),
    re.compile(r"\binsufficient\s+(interconnected\s+)?(evidence|information|data|chunks)", re.I),
    re.compile(r"\black(s|ing)?\s+(realistic\s+)?(connection|link|evidence)", re.I),
    re.compile(r"\bnot possible (to generate|with (the\s+)?provided)", re.I),
]

_META_ANSWER_PATTERNS = [
    re.compile(r"^no valid (answer|question|response)", re.I),
    re.compile(r"^(chunks?|evidence)\s+(do not|don'?t)\s+support", re.I),
    re.compile(r"^insufficient\b", re.I),
    re.compile(r"^cannot (be )?(determined|answered|generated)", re.I),
    re.compile(r"^not possible\b", re.I),
    re.compile(r"\bmulti[- ]?hop chaining\b", re.I),
    re.compile(r"^(no|lacking)\s+(valid\s+)?(evidence|connection|link)", re.I),
]


def _is_meta_question(question: str, answer: str) -> bool:
    """Check if the QA is a meta-statement about generation failure."""
    for pattern in _META_QUESTION_PATTERNS:
        if pattern.search(question):
            return True
    for pattern in _META_ANSWER_PATTERNS:
        if pattern.search(answer):
            return True
    return False


# Unicode ranges for language detection.
_LANGUAGE_CHAR_RANGES: dict[str, list[tuple[int, int]]] = {
    "korean": [(0xAC00, 0xD7A3), (0x3131, 0x318E)],
    "japanese": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
    "chinese": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "arabic": [(0x0600, 0x06FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "hindi": [(0x0900, 0x097F)],
}


def _check_language(text: str, language: str) -> bool:
    """Check if text contains a minimum proportion of target language characters.

    Returns True if the language has no configured char ranges (unknown language)
    or if >=50% of non-space characters match the target range.
    """
    lang_key = language.lower()
    ranges = _LANGUAGE_CHAR_RANGES.get(lang_key)
    if not ranges:
        return True  # unknown language — can't check, pass through

    non_space = text.replace(" ", "")
    if not non_space:
        return True

    matching = sum(1 for c in non_space if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return matching / len(non_space) >= 0.5


def _word_set(text: str) -> set[str]:
    """Lowercase word tokens for Jaccard similarity."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _max_chunk_pair_overlap(
    reference_chunks: list[Any],
    threshold: float,
) -> tuple[bool, float]:
    """Check if any pair of reference chunks exceeds Jaccard similarity threshold.

    Returns (exceeded, max_similarity).
    """
    if len(reference_chunks) < 2:
        return False, 0.0
    word_sets = []
    for chunk in reference_chunks:
        content = str(chunk.get("content", "") if isinstance(chunk, dict) else "")
        word_sets.append(_word_set(content))

    max_sim = 0.0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            sim = _jaccard(word_sets[i], word_sets[j])
            max_sim = max(max_sim, sim)
            if max_sim >= threshold:
                return True, max_sim
    return False, max_sim


class DeterministicGuardsFilter:
    """Applies deterministic validation checks before expensive evaluation."""

    def __init__(self, cfg: DeterministicGuardsConfig) -> None:
        self.cfg = cfg

    def evaluate(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        corpus_language = str(context.get("corpus_language", "") or "").strip().lower()
        stats = context.setdefault("deterministic_guard_stats", {"passed": 0, "rejected": 0})
        for item in items:
            if item.filter_verdict is not None and not item.is_passed:
                continue

            question = str(item.qa.get("question", "")).strip()
            answer = str(item.qa.get("answer", "")).strip()
            reference_chunks = list(item.qa.get("reference_chunks", []) or [])
            qa_type = str(item.qa.get("qa_type", "")).strip()
            if not qa_type:
                qa_type = str(item.generation_metadata.get("qa_type_target", "")).strip()

            reason = self._validate(
                question=question,
                answer=answer,
                reference_chunks=reference_chunks,
                qa_type=qa_type,
                corpus_language=corpus_language,
            )
            if reason is None:
                item.filter_verdict = FilterVerdict(
                    status="passed",
                    reason="deterministic_passed",
                    reasoning="Passed deterministic guards.",
                    metadata={
                        "filter_mode": "deterministic_guards",
                        "reason_code": "deterministic_passed",
                        "confidence": 1.0,
                        "retrieval_query": str(item.qa.get("retrieval_query", "")).strip()
                        or question,
                        "ref_overlap_ratio": None,
                        "feedback_type": None,
                        "refinement_hint": None,
                    },
                )
                stats["passed"] = int(stats.get("passed", 0)) + 1
                continue

            item.filter_verdict = FilterVerdict(
                status="rejected",
                reason="deterministic_rejected",
                reasoning=reason,
                metadata={
                    "filter_mode": "deterministic_guards",
                    "reason_code": "deterministic_rejected",
                    "confidence": 1.0,
                    "retrieval_query": str(item.qa.get("retrieval_query", "")).strip() or question,
                    "ref_overlap_ratio": None,
                    "feedback_type": None,
                    "refinement_hint": reason,
                },
            )
            stats["rejected"] = int(stats.get("rejected", 0)) + 1
        return items

    def _validate(
        self,
        *,
        question: str,
        answer: str,
        reference_chunks: list[Any],
        qa_type: str = "",
        corpus_language: str = "",
    ) -> str | None:
        if len(question) < self.cfg.min_question_chars:
            return "question_too_short"
        if question.strip().lower() in _STYLE_LABELS:
            return "question_is_style_label"
        min_chars = (
            self.cfg.min_answer_chars_multi_hop
            if qa_type == "multi_hop"
            else self.cfg.min_answer_chars
        )
        # CJK languages pack ~2-3x more meaning per character.
        if corpus_language and corpus_language.lower() in _HIGH_DENSITY_LANGUAGES:
            min_chars = max(1, min_chars // 3)
        if len(answer) < min_chars:
            return "answer_too_short"
        if corpus_language and not _check_language(question, corpus_language):
            return "wrong_language"
        if qa_type == "multi_hop":
            if len(reference_chunks) < 2:
                return "multi_hop_insufficient_chunks"
        else:
            if len(reference_chunks) < self.cfg.min_reference_chunks:
                return "insufficient_reference_chunks"
        if _is_meta_question(question, answer):
            return "meta_question_about_generation"
        if qa_type == "multi_hop" and len(reference_chunks) >= 2:
            exceeded, _ = _max_chunk_pair_overlap(
                reference_chunks, self.cfg.chunk_overlap_threshold
            )
            if exceeded:
                return "multi_hop_high_chunk_overlap"
        return None
