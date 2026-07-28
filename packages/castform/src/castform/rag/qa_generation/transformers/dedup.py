"""Post-filter question deduplication transformer.

Removes near-duplicate questions from the passed set using n-gram
Jaccard similarity.  Zero LLM cost — pure string matching.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from castform.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from castform.rag.qa_generation.pipeline_config import PipelineContext

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


@dataclass
class DedupConfig:
    """Configuration for question deduplication."""

    enabled: bool = True
    similarity_threshold: float = 0.70
    ngram_size: int = 2
    stats_key: str = "dedup_stats"


class QuestionDeduplicator:
    """Removes near-duplicate questions via n-gram Jaccard similarity.

    For each pair of passed items, computes bigram Jaccard similarity
    on the question text.  When similarity exceeds the threshold, the
    later item is marked as a duplicate and moved to rejected.
    """

    def __init__(self, cfg: DedupConfig | None = None) -> None:
        self.cfg = cfg or DedupConfig()

    def deduplicate(
        self,
        passed: list[GeneratedQA],
        rejected: list[GeneratedQA],
        context: PipelineContext,
    ) -> tuple[list[GeneratedQA], list[GeneratedQA]]:
        """Remove near-duplicates from *passed*, appending them to *rejected*.

        Returns:
            Tuple of (deduplicated_passed, updated_rejected).
        """
        if not self.cfg.enabled or len(passed) < 2:
            return passed, rejected

        stats = context.setdefault(
            self.cfg.stats_key,
            {"duplicates_removed": 0, "total_evaluated": len(passed)},
        )

        threshold = self.cfg.similarity_threshold
        n = self.cfg.ngram_size

        # Pre-compute ngram sets for all questions.
        ngram_sets: list[set[tuple[str, ...]]] = []
        for item in passed:
            question = str(item.qa.get("question", ""))
            tokens = _WORD_RE.findall(question.lower())
            ngrams = {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
            ngram_sets.append(ngrams)

        # Greedily mark duplicates — keep the first occurrence.
        is_duplicate = [False] * len(passed)
        for i in range(len(passed)):
            if is_duplicate[i]:
                continue
            for j in range(i + 1, len(passed)):
                if is_duplicate[j]:
                    continue
                sim = _jaccard(ngram_sets[i], ngram_sets[j])
                if sim >= threshold:
                    is_duplicate[j] = True

        deduplicated: list[GeneratedQA] = []
        new_rejected: list[GeneratedQA] = list(rejected)
        dup_count = 0

        for i, item in enumerate(passed):
            if is_duplicate[i]:
                item.filter_verdict = FilterVerdict(
                    status="rejected",
                    reason="near_duplicate",
                    reasoning=("Question is a near-duplicate of an earlier passed question."),
                    metadata={
                        "filter_mode": "dedup",
                        "reason_code": "near_duplicate",
                        "similarity_threshold": threshold,
                    },
                )
                new_rejected.append(item)
                dup_count += 1
            else:
                deduplicated.append(item)

        stats["duplicates_removed"] = stats.get("duplicates_removed", 0) + dup_count
        stats["total_evaluated"] = stats.get("total_evaluated", 0) + len(passed)

        if dup_count > 0:
            logger.info(
                "Dedup: removed %d near-duplicates (threshold=%.2f)",
                dup_count,
                threshold,
            )

        return deduplicated, new_rejected


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class IncrementalDeduplicator:
    """Maintains a growing set of accepted question n-grams for per-batch dedup.

    Used inside the work queue to catch duplicates before they enter
    the expensive filter chain.  Thread-safe for parallel batch mode.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        ngram_size: int = 2,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size
        self._accepted_ngrams: list[set[tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def check_batch(
        self,
        items: list[GeneratedQA],
    ) -> tuple[list[GeneratedQA], list[GeneratedQA]]:
        """Check a batch against accepted set.

        Returns:
            (unique, duplicates) — duplicates have filter_verdict set.
        """
        with self._lock:
            unique: list[GeneratedQA] = []
            duplicates: list[GeneratedQA] = []
            for item in items:
                question = str(item.qa.get("question", ""))
                ngrams = self._compute_ngrams(question)
                if self._is_duplicate(ngrams):
                    item.filter_verdict = FilterVerdict(
                        status="rejected",
                        reason="early_near_duplicate",
                        reasoning=("Near-duplicate of already-accepted question."),
                        metadata={
                            "reason_code": "early_near_duplicate",
                        },
                    )
                    duplicates.append(item)
                else:
                    unique.append(item)
            return unique, duplicates

    def register_accepted(self, items: list[GeneratedQA]) -> None:
        """Add newly accepted items to the dedup index."""
        with self._lock:
            for item in items:
                question = str(item.qa.get("question", ""))
                ngrams = self._compute_ngrams(question)
                self._accepted_ngrams.append(ngrams)

    def _is_duplicate(self, ngrams: set[tuple[str, ...]]) -> bool:
        if not ngrams:
            return False
        for accepted in self._accepted_ngrams:
            if not accepted:
                continue
            if _jaccard(ngrams, accepted) >= self.similarity_threshold:
                return True
        return False

    def _compute_ngrams(self, text: str) -> set[tuple[str, ...]]:
        tokens = _WORD_RE.findall(text.lower())
        n = self.ngram_size
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
