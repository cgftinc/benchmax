"""Eval JSONL schema for the Neon corpus retrieval eval.

Contract-freeze artifact (Slice A). The models here are the *real* frozen wire
schema (they validate today); the eval *data* is generated in a later slice.

Gold-chunk identity carries the hash explicitly
-----------------------------------------------
``Chunk.to_dict`` in ``rag/chunkers/models.py`` OMITS the ``hash`` field, so the
gold/decoy references cannot be recovered from a serialized chunk. This schema
therefore carries the chunk hashes as first-class fields (``gold_chunk_hashes``,
``decoy_chunk_hashes``) — an eval record is self-contained and does not depend on
re-deriving identity from content.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from castform.rag.corpus.search_schema.search_types import SearchMode


class NeonEvalThresholds(BaseModel, frozen=True):
    """Per-mode pass thresholds plus the lexical-ablation delta.

    Args:
        hit_at_k: Minimum hit@k (fraction of queries whose gold set is retrieved
            within the top ``k``).
        mrr_at_k: Minimum mean reciprocal rank over the top ``k``.
        k: The retrieval depth ``k`` these thresholds are measured at.
    """

    hit_at_k: float
    mrr_at_k: float
    k: int = 5


# Frozen default thresholds per mode. Hybrid must clear the highest bar; vector
# beats lexical; the ablation delta is how much hybrid must beat lexical-only.
DEFAULT_THRESHOLDS: dict[SearchMode, NeonEvalThresholds] = {
    "lexical": NeonEvalThresholds(hit_at_k=0.80, mrr_at_k=0.65, k=5),
    "vector": NeonEvalThresholds(hit_at_k=0.85, mrr_at_k=0.70, k=5),
    "hybrid": NeonEvalThresholds(hit_at_k=0.90, mrr_at_k=0.75, k=5),
}

LEXICAL_ABLATION_MIN_DELTA = 0.05
"""Minimum hit@5 improvement hybrid must show over lexical-only retrieval."""


class NeonEvalRecord(BaseModel):
    """One retrieval-eval example.

    Args:
        capability: The retrieval capability under test (e.g. a filter or mode).
        search_mode: Mode the example exercises.
        query: The eval query text.
        filter_dsl: Optional filter predicate in the JSON DSL form
            (``search_schema`` serialization), or ``None`` for unfiltered.
        gold_chunk_hashes: Exact chunk-hash ids that count as correct hits.
        decoy_chunk_hashes: Chunk-hash ids that must NOT be surfaced as hits.
    """

    capability: str
    search_mode: SearchMode
    query: str
    filter_dsl: dict | None = None
    gold_chunk_hashes: list[str] = Field(min_length=1)
    decoy_chunk_hashes: list[str] = Field(default_factory=list)
