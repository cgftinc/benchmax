"""MetadataChunkLinker — zero-cost linking via search + entity patterns.

**Search results ARE the secondaries**, rather than random selection with
BM25 hints on the side.

Query strategy (in priority order):
1. Header-context queries from chunk metadata (h2, h3, section_header, title)
2. Code-pattern regex matches from discriminative entity patterns
3. Compound entity queries (pairwise entity combinations found in chunk)
4. TF-IDF keyphrases (corpus-informed, content-aware)

Post-search pipeline:
- QA-type-aware same-file filtering
- Cross-question deduplication (prevents chunk reuse across questions)
- Coherence floor (minimum Jaccard with primary chunk)
- Entity-based coherence re-ranking
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from benchmax.rag.qa_generation.anchor_selector import AnchorBundle
from benchmax.rag.qa_generation.corpus_profile import CorpusProfile
from benchmax.rag.qa_generation.wiki_builder import WikiIndex

logger = logging.getLogger(__name__)


@dataclass
class MetadataLinkerConfig:
    """Configuration for ``MetadataChunkLinker``."""

    max_candidates: int = 10
    max_secondaries: int = 3
    min_chunk_chars: int = 400
    filter_same_file: bool = True
    min_coherence: float = 0.15
    max_secondary_similarity: float = 0.55
    max_primary_similarity: float = 0.55
    retry_confidence: float = 0.5
    header_keys: tuple[str, ...] = ("h2", "h3", "section_header", "title", "h1")
    wiki_cooccurrence_weight: float = 0.3


@dataclass(frozen=True)
class _ModeOverrides:
    """Per-call overrides derived from reasoning_mode."""

    filter_same_file: bool
    min_coherence: float
    prefer_search_mode: str | None = None
    prefer_date_diversity: bool = False
    prefer_index_proximity: bool = False
    index_proximity_range: tuple[int, int] = (2, 5)


class MetadataChunkLinker:
    """Zero-cost chunk linker using header-context queries, entity patterns,
    and search-based secondary selection.

    No LLM calls are made during linking.
    """

    def __init__(
        self,
        source: Any,
        profile: CorpusProfile,
        *,
        config: MetadataLinkerConfig | None = None,
        wiki_index: WikiIndex | None = None,
    ) -> None:
        self.source = source
        self.profile = profile
        self.config = config or MetadataLinkerConfig()
        self.wiki_index = wiki_index
        self._used_hashes: set[str] = set()
        self._last_primary_entities: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link(
        self,
        primary_chunk: Any,
        *,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
        reasoning_mode: str = "",
    ) -> AnchorBundle:
        """Find secondary chunks for multi-hop QA generation."""
        hop_count = target_hop_count or 2
        n_secondaries = min(hop_count - 1, self.config.max_secondaries)
        if n_secondaries <= 0:
            return self._empty_bundle(primary_chunk, hop_count)

        overrides = self._resolve_mode_overrides(reasoning_mode, primary_chunk)

        # Primary attempt: header + entity queries.
        queries = self._generate_queries(primary_chunk)
        if not queries:
            return self._empty_bundle(primary_chunk, hop_count)

        candidates = self._search_and_filter(primary_chunk, queries, overrides)
        secondary_chunks = self._select_diverse(candidates, n_secondaries)

        # Retry with content-derived queries if confidence is low.
        confidence = self._compute_confidence(secondary_chunks, n_secondaries)
        retried = False
        if confidence < self.config.retry_confidence:
            retry_queries = self._generate_content_queries(primary_chunk)
            retry_queries = [q for q in retry_queries if q not in queries]
            if retry_queries:
                retry_candidates = self._search_and_filter(
                    primary_chunk, retry_queries, overrides
                )
                # Merge: existing candidates first, then new ones.
                seen = {getattr(c, "hash", id(c)) for c in candidates}
                for rc in retry_candidates:
                    h = getattr(rc, "hash", id(rc))
                    if h not in seen:
                        candidates.append(rc)
                        seen.add(h)
                secondary_chunks = self._select_diverse(candidates, n_secondaries)
                queries = queries + retry_queries
                confidence = self._compute_confidence(secondary_chunks, n_secondaries)
                retried = True

        # Register used hashes for cross-question dedup.
        primary_h = getattr(primary_chunk, "hash", None)
        if primary_h:
            self._used_hashes.add(primary_h)
        for chunk in secondary_chunks:
            h = getattr(chunk, "hash", None)
            if h:
                self._used_hashes.add(h)

        self._enrich_profile(queries, candidates)

        # Demote hop count to match actual secondaries found.
        actual_hop_count = len(secondary_chunks) + 1
        demoted_hops = actual_hop_count < hop_count

        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=secondary_chunks,
            target_hop_count=min(hop_count, actual_hop_count),
            structural_hints={
                "linker": "metadata",
                "confidence": confidence,
                "search_mode": self.profile.best_search_mode,
                "queries_used": queries,
                "candidates_found": len(candidates),
                "retried": retried,
                "hop_demoted": demoted_hops,
                "requested_hop_count": hop_count,
                "reasoning_mode": reasoning_mode,
            },
        )

    def reset_used_hashes(self) -> None:
        """Clear cross-question dedup state (e.g. between batches)."""
        self._used_hashes.clear()

    # ------------------------------------------------------------------
    # Search-and-filter pipeline (shared by primary + retry)
    # ------------------------------------------------------------------

    def _search_and_filter(
        self,
        primary_chunk: Any,
        queries: list[str],
        overrides: _ModeOverrides,
    ) -> list[Any]:
        """Run the full search → filter → coherence floor → rank → diverse select."""
        search_results = self._search(
            primary_chunk, queries, search_mode_override=overrides.prefer_search_mode
        )

        # Inject wiki-co-occurring chunks into the search results so they
        # enter the candidate pool even when BM25 doesn't return them.
        if self.wiki_index is not None:
            primary_hash = getattr(primary_chunk, "hash", None)
            if primary_hash and primary_hash in self.wiki_index.chunk_to_pages:
                existing_hashes = {
                    r.get("hash") or getattr(r.get("chunk"), "hash", None)
                    for r in search_results
                }
                wiki_hashes: set[str] = set()
                for page_title in self.wiki_index.chunk_to_pages[primary_hash]:
                    page = self.wiki_index.pages.get(page_title)
                    if page:
                        wiki_hashes.update(page.source_chunk_ids)
                        for linked_title in page.cross_links:
                            linked_page = self.wiki_index.pages.get(linked_title)
                            if linked_page:
                                wiki_hashes.update(linked_page.source_chunk_ids)
                wiki_hashes.discard(primary_hash)
                wiki_hashes -= existing_hashes
                # Look up actual chunk objects and inject with score 0.
                if hasattr(self.source, "collection") and self.source.collection:
                    for wh in wiki_hashes:
                        chunk_obj = self.source.collection.get_chunk_by_hash(wh)
                        if chunk_obj is not None:
                            search_results.append({"chunk": chunk_obj, "max_score": 0.0})

        # Filter: same-file, min chars, dedup, used hashes.
        # Returns (chunk, search_score) pairs to preserve relevance signal.
        filtered = self._filter_candidates(
            primary_chunk, search_results, filter_same_file=overrides.filter_same_file
        )

        # Coherence floor (binary) + compute Jaccard for ranking.
        primary_content = _get_content(primary_chunk)
        primary_tokens = set(primary_content.lower().split())
        scored: list[tuple[Any, float, float]] = []  # (chunk, search_score, jaccard)
        for chunk, search_score in filtered:
            chunk_tokens = set(_get_content(chunk).lower().split())
            jaccard = (
                len(primary_tokens & chunk_tokens) / len(primary_tokens | chunk_tokens)
                if primary_tokens and chunk_tokens
                else 0.0
            )
            if overrides.min_coherence > 0 and jaccard < overrides.min_coherence:
                continue
            if self.config.max_primary_similarity > 0 and jaccard > self.config.max_primary_similarity:  # noqa: E501
                continue
            scored.append((chunk, search_score, jaccard))

        if not scored:
            return []

        # Composite ranking: search relevance + jaccard + entity overlap + wiki co-occurrence.
        primary_entities = self._last_primary_entities
        max_search = max(s for _, s, _ in scored) or 1.0

        # Compute wiki-boosted hashes from the primary chunk.
        wiki_boosted: set[str] = set()
        if self.wiki_index is not None:
            primary_hash = getattr(primary_chunk, "hash", None)
            if primary_hash and primary_hash in self.wiki_index.chunk_to_pages:
                # Co-occurrence: chunks that share wiki pages with the primary.
                for page_title in self.wiki_index.chunk_to_pages[primary_hash]:
                    page = self.wiki_index.pages.get(page_title)
                    if page:
                        wiki_boosted.update(page.source_chunk_ids)
                        # Cross-link signal: chunks from related entity pages.
                        for linked_title in page.cross_links:
                            linked_page = self.wiki_index.pages.get(linked_title)
                            if linked_page:
                                wiki_boosted.update(linked_page.source_chunk_ids)
            # Don't boost the primary chunk itself.
            if primary_hash:
                wiki_boosted.discard(primary_hash)

        ranked: list[tuple[float, int, Any]] = []
        for idx, (chunk, search_score, jaccard) in enumerate(scored):
            entity_ratio = 0.0
            if primary_entities:
                content_lower = _get_content(chunk).lower()
                hits = sum(1 for e in primary_entities if e in content_lower)
                entity_ratio = hits / len(primary_entities)

            wiki_bonus = 0.0
            if wiki_boosted:
                chunk_hash = getattr(chunk, "hash", None)
                if chunk_hash and chunk_hash in wiki_boosted:
                    wiki_bonus = self.config.wiki_cooccurrence_weight

            # Normalise weights to sum to 1 when wiki bonus applies.
            base_weight = 1.0 - wiki_bonus
            composite = (
                base_weight * (
                    0.4 * (search_score / max_search)
                    + 0.3 * jaccard
                    + 0.3 * entity_ratio
                )
                + wiki_bonus
            )
            ranked.append((composite, idx, chunk))

        ranked.sort(key=lambda x: (-x[0], x[1]))
        candidates = [chunk for _, _, chunk in ranked]

        # Mode-specific reranking.
        if overrides.prefer_date_diversity:
            candidates = self._rerank_by_date_diversity(primary_chunk, candidates)
        if overrides.prefer_index_proximity:
            candidates = self._rerank_by_index_proximity(
                primary_chunk, candidates, overrides.index_proximity_range
            )

        return candidates

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def _generate_queries(self, primary_chunk: Any) -> list[str]:
        """Build search queries: headers, code patterns, compound entities,
        then TF-IDF keyphrases."""
        content = _get_content(primary_chunk)
        content_lower = content.lower()
        max_queries = self.config.max_candidates

        queries: list[str] = []

        # 1. Header-context queries (most precise signal).
        queries.extend(self._extract_header_queries(primary_chunk))

        # 2. Code-pattern regex matches + collect entities for compounds.
        discriminative = self.profile.get_discriminative_entities()
        found_entities: list[str] = []

        # Text entities: fast path from graph, fallback to scan
        primary_hash = getattr(primary_chunk, "hash", None)
        if primary_hash and primary_hash in self.profile.chunk_entity_index:
            graph_entities = self.profile.chunk_entity_index[primary_hash]
            disc_lower = {e.name.lower() for e in discriminative if e.type != "code_pattern"}
            found_entities = [e for e in graph_entities if e.lower() in disc_lower]
        else:
            for entity in discriminative:
                if entity.type != "code_pattern" and entity.name.lower() in content_lower:
                    found_entities.append(entity.name)

        # Code patterns: still need regex to extract matched tokens
        for entity in discriminative:
            if entity.type != "code_pattern":
                continue
            try:
                matches = re.findall(entity.name, content)
                for match in matches[:2]:
                    token = match[0] if isinstance(match, tuple) else match
                    token = str(token).strip()
                    if token and token not in queries:
                        queries.append(token)
            except re.error:
                continue
            if len(queries) >= max_queries:
                break

        # 3. Compound entity queries (pairwise combinations).
        for e1, e2 in combinations(found_entities[:4], 2):
            compound = f"{e1} {e2}"
            if compound not in queries:
                queries.append(compound)
            if len(queries) >= max_queries:
                break

        # 4. TF-IDF keyphrase queries (content-aware, corpus-informed).
        if len(queries) < max_queries and self.profile.token_df_sample_size > 0:
            keyphrases = self.profile.get_tfidf_keyphrases(
                content,
                top_n=max_queries - len(queries),
            )
            existing_lower = {q.lower() for q in queries}
            for kp in keyphrases:
                if kp.lower() not in existing_lower:
                    queries.append(kp)
                    existing_lower.add(kp.lower())
                if len(queries) >= max_queries:
                    break

        # Store entities for coherence scoring.
        self._last_primary_entities = {e.lower() for e in found_entities}

        return queries[:max_queries]

    def _extract_header_queries(self, chunk: Any) -> list[str]:
        """Extract search queries from chunk metadata using configured keys.

        List-type values are expanded so each item becomes a separate query.
        """
        md = _get_metadata(chunk)
        queries: list[str] = []
        for key in self.config.header_keys:
            val = md.get(key)
            if not val:
                continue
            # Expand list/tuple values into individual queries.
            items = val if isinstance(val, (list, tuple)) else [val]
            for item in items:
                q = str(item).strip()
                if q and q not in queries:
                    queries.append(q)
        return queries

    def _generate_content_queries(self, primary_chunk: Any) -> list[str]:
        """Generate queries from chunk body text — used as retry when
        header/entity queries don't find enough candidates.

        Extracts:
        1. First sentence (typically a summary of the chunk topic).
        2. Longest capitalized phrase (likely a product/feature name).
        """
        content = _get_content(primary_chunk).strip()
        if not content:
            return []

        queries: list[str] = []

        # 1. First sentence (up to 120 chars, truncated at sentence boundary).
        first_line = content.split("\n", 1)[0].strip()
        for sep in (".", "!", "?"):
            idx = first_line.find(sep)
            if 0 < idx < 120:
                first_line = first_line[: idx + 1]
                break
        first_line = first_line[:120].strip()
        # Strip leading markdown (e.g., "### Heading" or "1. Step")
        clean = re.sub(r"^(?:#+ |[\d]+\.\s*)", "", first_line).strip()
        if clean and len(clean) > 10:
            queries.append(clean)

        # 2. Longest capitalized multi-word phrase (likely a feature name).
        caps = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", content)
        if caps:
            best = max(caps, key=len)
            if best not in queries:
                queries.append(best)

        return queries[: self.config.max_candidates]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        primary_chunk: Any,
        queries: list[str],
        search_mode_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the full corpus using the best available mode."""
        if search_mode_override and search_mode_override in self.profile.search_modes:
            mode = search_mode_override
        else:
            mode = self.profile.best_search_mode
        search_mode = mode if mode != "lexical" else None
        # Oversample to give the diversity filter room to work.
        oversample_k = self.config.max_candidates * 3
        try:
            return self.source.search_related(
                primary_chunk,
                queries,
                top_k=oversample_k,
                mode=search_mode,
            )
        except Exception:
            logger.debug(
                "search_related failed with mode=%s, retrying without mode",
                search_mode,
            )
            return self.source.search_related(
                primary_chunk,
                queries,
                top_k=oversample_k,
            )

    # ------------------------------------------------------------------
    # Candidate filtering & ranking
    # ------------------------------------------------------------------

    def _filter_candidates(
        self,
        primary_chunk: Any,
        search_results: list[dict[str, Any]],
        filter_same_file: bool | None = None,
    ) -> list[tuple[Any, float]]:
        """Filter search results to valid secondary chunk candidates.

        Returns list of (chunk, search_score) tuples.
        """
        primary_hash = getattr(primary_chunk, "hash", None)
        primary_file = _get_file(primary_chunk)
        min_chars = self.config.min_chunk_chars

        filter_same = (
            filter_same_file if filter_same_file is not None else self.config.filter_same_file
        )

        candidates: list[tuple[Any, float]] = []
        seen_hashes: set[str] = set()

        for result in search_results:
            chunk = result.get("chunk")
            if chunk is None:
                continue

            chunk_hash = getattr(chunk, "hash", None)
            search_score = float(result.get("max_score", 0.0) or 0.0)

            # Skip the primary chunk itself.
            if chunk_hash and chunk_hash == primary_hash:
                continue

            # Skip duplicates within this search.
            if chunk_hash and chunk_hash in seen_hashes:
                continue

            # Skip chunks already used in previous questions.
            if chunk_hash and chunk_hash in self._used_hashes:
                continue

            # Skip tiny chunks.
            chunk_content = _get_content(chunk)
            if len(chunk_content) < min_chars:
                continue

            # Same-file filtering.
            if filter_same and primary_file:
                chunk_file = _get_file(chunk)
                if chunk_file and chunk_file == primary_file:
                    continue

            if chunk_hash:
                seen_hashes.add(chunk_hash)
            candidates.append((chunk, search_score))

        return candidates

    def _select_diverse(self, candidates: list[Any], n: int) -> list[Any]:
        """Greedily select up to *n* candidates, skipping any that are too
        similar to an already-selected candidate (Jaccard > max_secondary_similarity).
        """
        if not candidates or n <= 0:
            return []
        threshold = self.config.max_secondary_similarity
        selected: list[Any] = []
        selected_tokens: list[set[str]] = []
        for chunk in candidates:
            if len(selected) >= n:
                break
            tokens = set(_get_content(chunk).lower().split())
            too_similar = False
            for prev_tokens in selected_tokens:
                jaccard = (
                    len(tokens & prev_tokens) / len(tokens | prev_tokens)
                    if tokens and prev_tokens
                    else 0.0
                )
                if jaccard > threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(chunk)
                selected_tokens.append(tokens)
        return selected

    # ------------------------------------------------------------------
    # Reasoning-mode resolution & reranking
    # ------------------------------------------------------------------

    def _resolve_mode_overrides(self, reasoning_mode: str, primary_chunk: Any) -> _ModeOverrides:
        """Map reasoning_mode to concrete pipeline overrides."""
        mode = reasoning_mode.strip().lower() if reasoning_mode else ""
        defaults = _ModeOverrides(
            filter_same_file=self.config.filter_same_file,
            min_coherence=self.config.min_coherence,
        )

        if mode == "temporal":
            if _has_date_metadata(primary_chunk):
                return replace(defaults, prefer_date_diversity=True)
            logger.debug("temporal mode: primary chunk lacks date metadata; using factual")
            return defaults

        if mode == "inference":
            preferred = None
            if "vector" in self.profile.search_modes:
                preferred = "vector"
            elif "hybrid" in self.profile.search_modes:
                preferred = "hybrid"
            return replace(
                defaults,
                min_coherence=max(self.config.min_coherence * 0.5, 0.05),
                prefer_search_mode=preferred,
            )

        if mode == "sequential":
            md = _get_metadata(primary_chunk)
            if md.get("file") and md.get("index") is not None:
                return replace(
                    defaults,
                    filter_same_file=False,
                    prefer_index_proximity=True,
                )
            logger.debug("sequential mode: primary chunk lacks file+index; using factual")
            return defaults

        return defaults  # factual or unrecognized

    def _rerank_by_date_diversity(self, primary_chunk: Any, candidates: list[Any]) -> list[Any]:
        """Boost candidates whose date differs from the primary chunk.

        Uses a weighted combination: original rank (from coherence) is
        the primary signal, date diversity is a tiebreaker. Candidates
        at similar coherence ranks get re-ordered by date spread, but
        a highly coherent candidate won't be displaced by an unrelated
        one that just happens to have a distant date.
        """
        primary_date = _parse_date(primary_chunk)
        if primary_date is None or not candidates:
            return candidates

        # Normalize date distances to [0, 1] for weighting.
        day_diffs: list[int] = []
        for chunk in candidates:
            chunk_date = _parse_date(chunk)
            day_diffs.append(abs((chunk_date - primary_date).days) if chunk_date else 0)

        max_diff = max(day_diffs) if day_diffs else 1
        max_diff = max(max_diff, 1)  # avoid division by zero

        # Weighted score: 70% original rank preservation, 30% date diversity.
        scored: list[tuple[float, int, Any]] = []
        n = len(candidates)
        for idx, (chunk, dd) in enumerate(zip(candidates, day_diffs)):
            rank_score = 1.0 - (idx / max(n - 1, 1))  # 1.0 for first, 0.0 for last
            date_score = dd / max_diff
            combined = 0.7 * rank_score + 0.3 * date_score
            scored.append((combined, idx, chunk))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [chunk for _, _, chunk in scored]

    def _rerank_by_index_proximity(
        self,
        primary_chunk: Any,
        candidates: list[Any],
        index_range: tuple[int, int],
    ) -> list[Any]:
        """Boost same-file candidates with index gap in the target range."""
        primary_file = _get_file(primary_chunk)
        primary_md = _get_metadata(primary_chunk)
        primary_index = primary_md.get("index")
        if not primary_file or primary_index is None or not candidates:
            return candidates

        try:
            primary_index = int(primary_index)
        except (TypeError, ValueError):
            return candidates

        min_gap, max_gap = index_range

        scored: list[tuple[int, int, Any]] = []
        for idx, chunk in enumerate(candidates):
            chunk_file = _get_file(chunk)
            chunk_md = _get_metadata(chunk)
            chunk_index = chunk_md.get("index")

            score = 0  # neutral for cross-file or missing index
            if chunk_file == primary_file and chunk_index is not None:
                try:
                    gap = abs(int(chunk_index) - primary_index)
                    if min_gap <= gap <= max_gap:
                        score = max_gap - gap + 1  # closer = higher score
                except (TypeError, ValueError):
                    pass

            scored.append((score, idx, chunk))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [chunk for _, _, chunk in scored]

    # ------------------------------------------------------------------
    # Profile enrichment & helpers
    # ------------------------------------------------------------------

    def _enrich_profile(self, queries: list[str], candidates: list[Any]) -> None:
        """Update the CorpusProfile with linking signals."""
        for query in queries:
            self.profile.record_query_effectiveness(query, len(candidates))

        for candidate in candidates:
            candidate_file = _get_file(candidate)
            if candidate_file:
                self.profile.update_connection_stats("cross_document")
            else:
                self.profile.update_connection_stats("unknown")

    def _compute_confidence(
        self,
        secondaries: list[Any],
        requested: int,
    ) -> float:
        if requested == 0:
            return 1.0
        return min(len(secondaries) / requested, 1.0)

    def _empty_bundle(
        self,
        primary_chunk: Any,
        hop_count: int,
    ) -> AnchorBundle:
        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=[],
            target_hop_count=hop_count,
            structural_hints={"linker": "metadata", "confidence": 0.0},
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_metadata(chunk: Any) -> dict[str, Any]:
    """Extract metadata dict from a chunk."""
    if hasattr(chunk, "metadata_dict"):
        return chunk.metadata_dict or {}
    elif hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            return raw
        elif isinstance(raw, (tuple, list)):
            return dict(raw)
    elif isinstance(chunk, dict):
        return chunk.get("metadata", {})
    return {}


def _get_file(chunk: Any) -> str:
    """Extract file metadata from a chunk."""
    md = _get_metadata(chunk)
    return str(md.get("file", "") or md.get("file_name", "") or "")


def _get_content(chunk: Any) -> str:
    """Extract text content from a chunk."""
    if hasattr(chunk, "content"):
        return chunk.content
    return str(chunk)


def _has_date_metadata(chunk: Any) -> bool:
    """Check if chunk has date fields usable for temporal reasoning."""
    md = _get_metadata(chunk)
    return bool(md.get("date_start") or md.get("date_end") or md.get("date"))


def _parse_date(chunk: Any) -> datetime.date | None:
    """Try to parse a date from chunk metadata."""
    md = _get_metadata(chunk)
    for key in ("date_start", "date_end", "date"):
        val = md.get(key)
        if val:
            try:
                return datetime.date.fromisoformat(str(val)[:10])
            except (ValueError, TypeError):
                continue
    return None


def _token_set_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity on lowercased whitespace-split token sets."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
