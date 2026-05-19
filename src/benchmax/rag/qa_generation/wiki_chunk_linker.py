"""Entity-graph-based chunk linker.

Unlike MetadataChunkLinker (search-based), WikiChunkLinker finds
complementary chunks by traversing the entity-chunk bipartite graph
built during corpus profiling.  It is complementary to MetadataChunkLinker,
not a replacement — it excels at cross-file, cross-topic connections
but lacks the semantic relevance signal from BM25/vector search.

No wiki page generation is required — the linker operates entirely on the
entity-chunk graph (entity_chunk_index, chunk_entity_index, entity_cooccurrence)
stored on CorpusProfile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from benchmax.rag.qa_generation.anchor_selector import AnchorBundle

logger = logging.getLogger(__name__)


@dataclass
class WikiChunkLinkerConfig:
    """Configuration for WikiChunkLinker."""

    max_secondaries: int = 3
    min_chunk_chars: int = 400
    max_primary_similarity: float = 0.55
    shared_entity_weight: float = 0.3
    new_entity_weight: float = 0.3
    entity_proximity_weight: float = 0.2
    cross_file_weight: float = 0.2


class WikiChunkLinker:
    """Entity-graph-based linker using the entity-chunk bipartite graph.

    Operates entirely on pre-computed graph data stored on CorpusProfile.
    No wiki page generation required.

    Algorithm:
    1. Look up primary chunk's entities via ``chunk_entity_index``
    2. Find candidate chunks sharing entities via ``entity_chunk_index``
    3. Score candidates by: shared entities (coherence), new entities
       (complementary info), entity proximity (co-occurrence), cross-file bonus
    4. Coverage-greedy selection maximizing entity diversity across secondaries
    """

    def __init__(
        self,
        source: Any,
        profile: Any,
        *,
        config: WikiChunkLinkerConfig | None = None,
    ) -> None:
        self.source = source
        self.profile = profile
        self.config = config or WikiChunkLinkerConfig()
        self._used_hashes: set[str] = set()

    def link(
        self,
        primary_chunk: Any,
        *,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
        reasoning_mode: str = "",
    ) -> AnchorBundle:
        """Find secondary chunks via entity graph traversal."""
        hop_count = target_hop_count or 2
        n_secondaries = min(hop_count - 1, self.config.max_secondaries)
        if n_secondaries <= 0:
            return self._empty_bundle(primary_chunk, hop_count)

        primary_hash = getattr(primary_chunk, "hash", None)
        if not primary_hash:
            return self._empty_bundle(primary_chunk, hop_count)

        # Step 1: Get primary chunk's entities from graph
        primary_entity_names = self.profile.chunk_entity_index.get(primary_hash, [])
        if not primary_entity_names:
            return self._empty_bundle(primary_chunk, hop_count)

        primary_entity_set = {e.lower() for e in primary_entity_names}
        primary_doc = _get_doc_id(primary_chunk)

        # Step 2: Find candidate chunks sharing entities
        candidate_hashes: set[str] = set()
        for entity_name in primary_entity_set:
            chunk_hashes = self.profile.entity_chunk_index.get(entity_name, set())
            candidate_hashes.update(chunk_hashes)
        candidate_hashes.discard(primary_hash)
        candidate_hashes -= self._used_hashes

        if not candidate_hashes:
            return self._empty_bundle(primary_chunk, hop_count)

        # Build pool lookup once for fast hash→chunk resolution
        pool_lookup: dict[str, Any] = {}
        if corpus_pool:
            pool_lookup = {
                getattr(c, "hash", None): c for c in corpus_pool
                if getattr(c, "hash", None)
            }

        # Step 3: Score candidates
        eq_map = self.profile.entity_quality_map
        p_text = primary_chunk.content if hasattr(primary_chunk, "content") else ""
        primary_tokens = set(p_text.lower().split()) if p_text else set()

        # scored: (hash, score, chunk_obj, candidate_entities)
        scored: list[tuple[str, float, Any, set[str]]] = []

        for ch in candidate_hashes:
            candidate_entities = set(
                e.lower() for e in self.profile.chunk_entity_index.get(ch, [])
            )
            shared = primary_entity_set & candidate_entities
            new = candidate_entities - primary_entity_set

            if not shared:
                continue  # no coherence signal

            # Shared entity score (coherence): quality-weighted
            shared_score = sum(eq_map.get(e, 0.5) for e in shared) / max(
                len(primary_entity_set), 1
            )

            # New entity score (complementary): quality-weighted
            new_score = (
                sum(eq_map.get(e, 0.5) for e in new) / max(len(candidate_entities), 1)
                if new
                else 0.0
            )

            # Entity proximity via co-occurrence graph
            proximity_score = self._entity_proximity_score(
                primary_entity_set, candidate_entities,
            )

            # Cross-file bonus
            chunk_obj = self._resolve_chunk(ch, pool_lookup)
            if chunk_obj is None:
                continue
            c_text = chunk_obj.content if hasattr(chunk_obj, "content") else ""
            if len(c_text) < self.config.min_chunk_chars:
                continue
            cross_file = 1.0 if _get_doc_id(chunk_obj) != primary_doc else 0.0

            # Jaccard similarity check (reject near-duplicates)
            if self.config.max_primary_similarity > 0 and primary_tokens:
                cand_tokens = set(c_text.lower().split())
                if cand_tokens:
                    jaccard = len(primary_tokens & cand_tokens) / len(
                        primary_tokens | cand_tokens
                    )
                    if jaccard > self.config.max_primary_similarity:
                        continue

            cfg = self.config
            score = (
                shared_score * cfg.shared_entity_weight
                + new_score * cfg.new_entity_weight
                + proximity_score * cfg.entity_proximity_weight
                + cross_file * cfg.cross_file_weight
            )
            scored.append((ch, score, chunk_obj, candidate_entities))

        if not scored:
            return self._empty_bundle(primary_chunk, hop_count)

        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Coverage-greedy selection from top candidates
        selected = self._select_diverse(
            scored, primary_entity_set, n_secondaries,
        )

        # Register used hashes
        self._used_hashes.add(primary_hash)
        for chunk in selected:
            h = getattr(chunk, "hash", None)
            if h:
                self._used_hashes.add(h)

        actual_hop_count = len(selected) + 1
        confidence = len(selected) / max(n_secondaries, 1)

        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=selected,
            target_hop_count=min(hop_count, actual_hop_count),
            structural_hints={
                "linker": "wiki",
                "confidence": confidence,
                "candidates_found": len(scored),
                "primary_entities": list(primary_entity_set),
                "hop_demoted": actual_hop_count < hop_count,
                "requested_hop_count": hop_count,
                "reasoning_mode": reasoning_mode,
            },
        )

    def reset_used_hashes(self) -> None:
        """Clear used-hash tracking between generation rounds."""
        self._used_hashes.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _entity_proximity_score(
        self, primary_entities: set[str], candidate_entities: set[str],
    ) -> float:
        """Score entity-level proximity via co-occurrence graph.

        Returns [0, 1] based on how strongly the primary and candidate
        chunks' entities co-occur across the corpus. Normalized relative
        to the max co-occurrence in the graph (corpus-size-agnostic).
        """
        cooc = self.profile.entity_cooccurrence
        if not cooc:
            return 0.0
        # Check all pairs between primary and candidate entities
        total_cooc = 0
        n_pairs = 0
        for pe in primary_entities:
            for ce in candidate_entities:
                if pe == ce:
                    continue
                key = (min(pe, ce), max(pe, ce))
                total_cooc += cooc.get(key, 0)
                n_pairs += 1
        if n_pairs == 0:
            return 0.0
        avg_cooc = total_cooc / n_pairs
        # Normalize relative to corpus: top 10% of max co-occurrence = 1.0
        max_cooc = max(cooc.values()) if cooc else 1
        return min(1.0, avg_cooc / max(max_cooc * 0.1, 1.0))

    def _resolve_chunk(
        self, chunk_hash: str, pool_lookup: dict[str, Any],
    ) -> Any | None:
        """Look up a chunk object by hash."""
        collection = getattr(self.source, "collection", None)
        if collection:
            return collection.get_chunk_by_hash(chunk_hash)
        return pool_lookup.get(chunk_hash)

    def _select_diverse(
        self,
        scored: list[tuple[str, float, Any, set[str]]],
        primary_entities: set[str],
        n: int,
    ) -> list[Any]:
        """Coverage-greedy selection: pick candidates maximizing new entities."""
        selected: list[Any] = []
        covered: set[str] = set(primary_entities)
        # Work from top-K candidates (capped for performance)
        pool = scored[:n * 5]

        for _ in range(n):
            if not pool:
                break
            # Pick candidate that adds the most new entity coverage
            best_idx = max(
                range(len(pool)),
                key=lambda i: len(pool[i][3] - covered),
            )
            _ch, _score, chunk_obj, cand_entities = pool.pop(best_idx)
            selected.append(chunk_obj)
            covered.update(cand_entities)

        return selected

    def _empty_bundle(self, primary_chunk: Any, hop_count: int) -> AnchorBundle:
        """Return a bundle with no secondaries."""
        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=[],
            target_hop_count=hop_count,
            structural_hints={"linker": "wiki", "confidence": 0.0},
        )


def _get_doc_id(chunk: Any) -> str:
    """Extract document identifier from chunk metadata."""
    md = getattr(chunk, "metadata_dict", {})
    if not md and hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, (list, tuple)):
            md = dict(raw)
        elif isinstance(raw, dict):
            md = raw
        else:
            md = {}
    for key in ("file", "doc_id", "document_id", "thread_id", "source"):
        val = md.get(key)
        if val:
            return str(val)
    return ""
