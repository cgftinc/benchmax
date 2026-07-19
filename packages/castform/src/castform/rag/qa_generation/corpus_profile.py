"""First-class corpus profile for threading context across pipeline stages.

``CorpusProfile`` consolidates the outputs of stages 2-3 (corpus summary,
entity extraction, capability detection) into a single mutable object.
The linker enriches it during linking; the generator reads the enriched
version for prompt conditioning.
"""

from __future__ import annotations

import functools
import logging
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from castform.rag.qa_generation.corpus_capabilities import CorpusCapabilities

logger = logging.getLogger(__name__)


@dataclass
class EntityPattern:
    """An extracted entity with document-frequency quality signal.

    Attributes:
        name: The entity string (e.g. "Redis", "TTL").
        type: One of "entity", "code_pattern", "domain_term".
        document_frequency: Fraction of sample chunks containing this entity (0.0-1.0).
        idf_weight: log(N/df) — higher values indicate more discriminative entities.
    """

    name: str
    type: str  # "entity" | "code_pattern" | "domain_term"
    document_frequency: float = 0.0
    idf_weight: float = 0.0
    quality_score: float = 0.0
    semantic_score: float = 0.0  # avg KeyBERT cosine similarity (0 for heuristic)
    aliases: tuple[str, ...] = ()  # non-canonical names merged into this entity


@dataclass
class CorpusMetadataCensus:
    """Corpus-relative distributions for calibrating chunk scoring.

    Computed once over the profile pool after entity extraction.
    All percentile fields are derived from ``statistics.quantiles(data, n=4)``.
    """

    # Prevalence: fraction of chunks with each metadata type
    header_prevalence: float  # any of h1/h2/h3/title/section_header
    doc_id_prevalence: float  # file_name/document_id/source
    date_prevalence: float  # date/timestamp fields
    sequential_prevalence: float  # prev_chunk_id/next_chunk_id or file+index

    # Content length distribution (characters)
    content_length_p25: int
    content_length_p50: int
    content_length_p75: int

    # Entity density distribution (recognized entities per 1000 chars)
    entity_density_p25: float
    entity_density_p50: float
    entity_density_p75: float

    # Lexical diversity distribution (unique tokens / total tokens)
    lexical_diversity_p25: float
    lexical_diversity_p50: float
    lexical_diversity_p75: float

    # Derived classification
    metadata_regime: str  # "structured" | "mixed" | "unstructured"

    # Chunk count (from source)
    chunk_count: int

    @property
    def multi_file_corpus(self) -> bool:
        return self.doc_id_prevalence > 0.05


@dataclass
class CorpusProfile:
    """Shared mutable corpus context threaded across pipeline stages.

    Created by stages 2-3 (profiling + entity extraction), enriched by the
    linker during linking, and read by the generator for prompt conditioning.
    """

    # --- Set by initial profiling stage (stages 2-3) ---
    corpus_summary: str = ""
    corpus_queries: list[str] = field(default_factory=list)
    corpus_description: str = ""
    entity_patterns: list[EntityPattern] = field(default_factory=list)
    capabilities: CorpusCapabilities = field(
        default_factory=lambda: CorpusCapabilities(
            has_document_ids=False,
            has_section_headers=False,
            has_sequential_links=False,
            has_dates=False,
        )
    )

    # --- Entity quality signals (computed from profile_sample) ---
    entity_document_frequency: dict[str, float] = field(default_factory=dict)
    ubiquitous_df_threshold: float = 0.80

    # --- Token-level TF-IDF signals (computed from profile_sample) ---
    token_document_frequency: dict[str, int] = field(default_factory=dict)
    token_df_sample_size: int = 0

    # --- Search capability detection ---
    search_modes: set[str] = field(default_factory=lambda: {"lexical"})
    best_search_mode: str = "lexical"

    # --- Enriched by linker during linking ---
    discovered_entities: list[str] = field(default_factory=list)
    effective_query_templates: list[str] = field(default_factory=list)
    connection_distribution: dict[str, int] = field(default_factory=dict)
    _query_effectiveness: dict[str, list[int]] = field(default_factory=dict)

    # --- Entity-chunk graph (computed during profiling) ---
    entity_chunk_index: dict[str, set[str]] = field(default_factory=dict)
    # entity name (lower) -> set of chunk hashes containing it (unfiltered)
    chunk_entity_index: dict[str, list[str]] = field(default_factory=dict)
    # chunk hash -> list of entity names found in it (unfiltered)
    entity_cooccurrence: dict[tuple[str, str], int] = field(default_factory=dict)
    # (entity_a, entity_b) canonicalized pair -> count of chunks containing both

    # --- Census and suitability (computed after entity extraction) ---
    census: CorpusMetadataCensus | None = None
    chunk_suitability_scores: dict[str, float] = field(default_factory=dict)
    # key = chunk hash, value = suitability score [0, 1]

    # --- Convenience ---

    @property
    def corpus_queries_bulleted(self) -> str:
        """Queries formatted for prompt templates."""
        return "\n".join(f"- {q}" for q in self.corpus_queries)

    @functools.cached_property
    def entity_quality_map(self) -> dict[str, float]:
        """Cached mapping of entity name (lower) -> quality_score."""
        return {p.name.lower(): p.quality_score for p in self.entity_patterns}

    def get_discriminative_entities(
        self,
        max_df: float | None = None,
        min_df: float = 0.005,
    ) -> list[EntityPattern]:
        """Return entities within a useful DF band (not too rare, not ubiquitous)."""
        threshold = max_df if max_df is not None else self.ubiquitous_df_threshold
        return [e for e in self.entity_patterns if min_df <= e.document_frequency < threshold]

    def get_entity_names(self, *, discriminative_only: bool = True) -> list[str]:
        """Return entity name strings, optionally filtered by DF."""
        if discriminative_only:
            return [e.name for e in self.get_discriminative_entities()]
        return [e.name for e in self.entity_patterns]

    def get_domain_terms(self, *, discriminative_only: bool = True) -> list[str]:
        """Return domain term strings, optionally filtered by DF."""
        if discriminative_only:
            return [e.name for e in self.get_discriminative_entities() if e.type == "domain_term"]
        return [e.name for e in self.entity_patterns if e.type == "domain_term"]

    def get_tfidf_keyphrases(
        self,
        chunk_content: str,
        *,
        top_n: int = 5,
        min_token_len: int = 4,
    ) -> list[str]:
        """Extract top-N discriminative keyphrases from *chunk_content*.

        Uses corpus-level token document frequency to compute TF-IDF
        scores per token, returning the highest-scoring tokens.
        """
        import re

        if self.token_df_sample_size == 0:
            return []

        tokens = re.findall(r"\w+", chunk_content.lower())
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        df_threshold = self.token_df_sample_size * 0.8

        scored: list[tuple[float, str]] = []
        for tok, count in tf.items():
            if len(tok) < min_token_len:
                continue
            df = self.token_document_frequency.get(tok)
            if df is None or df >= df_threshold:
                continue
            tfidf = count * math.log(self.token_df_sample_size / max(df, 1))
            scored.append((tfidf, tok))

        scored.sort(key=lambda x: -x[0])
        return [tok for _, tok in scored[:top_n]]

    # --- Enrichment methods (called by linker) ---

    def add_discovered_entities(self, entities: list[str]) -> None:
        """Record entities discovered during linking."""
        for entity in entities:
            if entity and entity not in self.discovered_entities:
                self.discovered_entities.append(entity)

    def record_query_effectiveness(self, query: str, num_results: int) -> None:
        """Track how many results each query template returns."""
        self._query_effectiveness.setdefault(query, []).append(num_results)
        if num_results > 0 and query not in self.effective_query_templates:
            self.effective_query_templates.append(query)

    def update_connection_stats(self, connection_type: str) -> None:
        """Increment the count for a connection type discovered during linking."""
        self.connection_distribution[connection_type] = (
            self.connection_distribution.get(connection_type, 0) + 1
        )


_ENTITY_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "it", "if",
    "you", "your", "this", "that", "with", "from", "by", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "can", "could", "would", "should", "may", "not", "no", "or", "and", "but",
    "so", "as", "its", "our", "we", "re", "s", "t",
})


def _score_entity_quality(pattern: EntityPattern) -> float:
    """Score entity quality from DF + name heuristics + semantic signal. Returns [0, 1].

    When ``semantic_score`` is set (KeyBERT entities), it replaces the name
    quality heuristic — giving a direct, language-agnostic quality signal.

    Weights:
    - DF discriminativeness (40%): flat top at 0.02-0.10, taper to 0 at 0.40.
    - Type bonus (20%): entity > code_pattern > domain_term.
    - Name/semantic quality (40%): semantic_score when available, else name heuristics.
    """
    df = pattern.document_frequency

    # DF discriminativeness: sweet spot 0.02-0.10
    if df <= 0.0:
        df_score = 0.0
    elif df < 0.02:
        df_score = df / 0.02
    elif df <= 0.10:
        df_score = 1.0
    elif df <= 0.40:
        df_score = 1.0 - (df - 0.10) / 0.30
    else:
        df_score = 0.0

    # Type bonus
    if pattern.type == "entity":
        type_score = 1.0
    elif pattern.type == "code_pattern":
        type_score = 0.85
    else:  # domain_term
        type_score = 0.6

    # Name/semantic quality
    if pattern.semantic_score > 0.0:
        # KeyBERT entities: use semantic similarity directly (already 0-1 range)
        name_score = min(pattern.semantic_score / 0.6, 1.0)  # normalize: 0.6+ → 1.0
    elif pattern.type == "code_pattern":
        name_score = 0.8
    else:
        words = pattern.name.lower().split()
        if all(w in _ENTITY_STOPWORDS for w in words):
            name_score = 0.0
        elif len(pattern.name) < 4:
            name_score = 0.4
        else:
            name_score = 0.7  # neutral baseline

    return df_score * 0.4 + type_score * 0.2 + name_score * 0.4


def _compute_entity_specificity(
    entity_patterns: list[EntityPattern],
    entity_chunk_idx: dict[str, set[str]],
    chunk_entity_idx: dict[str, list[str]],
) -> None:
    """Second-pass quality refinement using co-occurrence entropy.

    Generic section headers (Setup, Troubleshooting) co-occur with random
    entities across many topics → high entropy → low specificity.
    Domain-specific entities (Feature Flags, Error Tracking) co-occur with a
    consistent set of related entities → low entropy → high specificity.

    This signal is language-agnostic: it relies only on the bipartite graph
    structure, not any English vocabulary.

    Mutates ``entity_patterns`` in place, blending specificity into
    ``quality_score``:  quality = base_quality * 0.7 + specificity * 0.3
    """
    for pattern in entity_patterns:
        name_lower = pattern.name.lower()
        chunks_with_entity = entity_chunk_idx.get(name_lower, set())

        if not chunks_with_entity:
            continue

        # Build frequency distribution of co-occurring entity names
        cooc_counts: dict[str, int] = {}
        for chunk_hash in chunks_with_entity:
            for other_name in chunk_entity_idx.get(chunk_hash, []):
                if other_name.lower() == name_lower:
                    continue
                key = other_name.lower()
                cooc_counts[key] = cooc_counts.get(key, 0) + 1

        if not cooc_counts:
            # Appears only in isolation — treat as maximally specific
            specificity = 1.0
        else:
            total = sum(cooc_counts.values())
            raw_entropy = 0.0
            for count in cooc_counts.values():
                p = count / total
                raw_entropy -= p * math.log(p)

            n_distinct = len(cooc_counts)
            max_entropy = math.log(n_distinct) if n_distinct > 1 else 1.0
            normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
            specificity = 1.0 - normalized_entropy

        pattern.quality_score = pattern.quality_score * 0.7 + specificity * 0.3


def build_entity_chunk_graph(
    entity_patterns: list[EntityPattern],
    chunks: list[Any],
) -> tuple[dict[str, set[str]], dict[str, list[str]], dict[tuple[str, str], int]]:
    """Single O(E×N) scan: build entity↔chunk bipartite graph and compute DF/quality.

    Mutates ``entity_patterns`` in place, setting ``document_frequency``,
    ``idf_weight``, and ``quality_score`` on each pattern.

    Returns (entity_chunk_index, chunk_entity_index, entity_cooccurrence).
    All indexes are **unfiltered** — consumers apply their own quality thresholds.
    """
    n = len(chunks)
    if n == 0:
        return {}, {}, {}

    entity_chunk_idx: dict[str, set[str]] = {}
    chunk_entity_idx: dict[str, list[str]] = {}

    for pattern in entity_patterns:
        name_lower = pattern.name.lower()
        matched: set[str] = set()

        for chunk in chunks:
            content = (chunk.content if hasattr(chunk, "content") else str(chunk)).lower()
            chunk_hash = getattr(chunk, "hash", str(id(chunk)))

            if pattern.type == "code_pattern":
                try:
                    if re.search(pattern.name, content, re.IGNORECASE):
                        matched.add(chunk_hash)
                        chunk_entity_idx.setdefault(chunk_hash, []).append(pattern.name)
                except re.error:
                    continue
            elif name_lower in content:
                matched.add(chunk_hash)
                chunk_entity_idx.setdefault(chunk_hash, []).append(pattern.name)
            elif any(alias.lower() in content for alias in pattern.aliases):
                matched.add(chunk_hash)
                chunk_entity_idx.setdefault(chunk_hash, []).append(pattern.name)

        entity_chunk_idx[name_lower] = matched
        count = len(matched)
        pattern.document_frequency = count / n
        pattern.idf_weight = math.log(n / max(count, 1))

    # Compute quality scores (needs DF to be set first)
    for pattern in entity_patterns:
        pattern.quality_score = _score_entity_quality(pattern)

    # Co-occurrence from the full graph
    cooccurrence: dict[tuple[str, str], int] = {}
    entity_names = sorted(
        name for name, hashes in entity_chunk_idx.items() if hashes
    )
    for i, e1 in enumerate(entity_names):
        for e2 in entity_names[i + 1:]:
            overlap = len(entity_chunk_idx[e1] & entity_chunk_idx[e2])
            if overlap > 0:
                cooccurrence[(e1, e2)] = overlap

    return entity_chunk_idx, chunk_entity_idx, cooccurrence


def compute_entity_document_frequency(
    entity_patterns: list[EntityPattern],
    sample_chunks: list[Any],
) -> None:
    """Compute DF and quality score for each entity. Thin wrapper around graph build.

    Mutates ``entity_patterns`` in place, setting ``document_frequency``,
    ``idf_weight``, and ``quality_score`` on each pattern.
    """
    build_entity_chunk_graph(entity_patterns, sample_chunks)


def compute_token_document_frequency(
    sample_chunks: list[Any],
    *,
    min_token_len: int = 3,
) -> tuple[dict[str, int], int]:
    """Compute per-token document frequency across a chunk sample.

    Returns (token_df_dict, sample_size).
    """
    import re

    n = len(sample_chunks)
    if n == 0:
        return {}, 0

    df: dict[str, int] = {}
    for chunk in sample_chunks:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        unique_tokens = set(re.findall(r"\w+", content.lower()))
        for tok in unique_tokens:
            if len(tok) >= min_token_len:
                df[tok] = df.get(tok, 0) + 1

    return df, n


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "about",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "so",
        "if",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "that",
        "this",
        "it",
        "its",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "any",
        "up",
        "down",
        "new",
        "old",
        "true",
        "false",
        "null",
        "none",
        "yes",
        "get",
        "set",
        "var",
        "let",
        "const",
        "def",
        "return",
        "class",
        "function",
        "import",
        "from",
        "type",
        "string",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "object",
        "value",
        "key",
        "name",
        "data",
        "item",
        "index",
        "error",
        "result",
        "response",
        "request",
        "input",
        "output",
        "file",
        "path",
        "config",
        "options",
        "args",
        "params",
        "default",
        "test",
        "example",
        "code",
        "note",
        "using",
        "use",
        "used",
        "see",
        "ignoring",
        "views",
        "alpha",
    }
)


def _is_quality_entity(name: str) -> bool:
    """Check if an entity name meets minimum quality bar."""
    if len(name) < 3:
        return False
    if name.lower() in _STOPWORDS:
        return False
    return True


def build_entity_patterns_from_extraction(
    entity_names: list[str],
    code_patterns: dict[str, str],
    domain_terms: list[str],
) -> list[EntityPattern]:
    """Convert raw extraction output into ``EntityPattern`` objects.

    Applies quality heuristics: minimum length, stopword rejection,
    and case-insensitive deduplication.
    """
    seen_lower: set[str] = set()
    patterns: list[EntityPattern] = []

    def _add(name: str, entity_type: str) -> None:
        name = name.strip()
        if not name:
            return
        # Code patterns skip text-quality checks (they're regexes).
        if entity_type != "code_pattern" and not _is_quality_entity(name):
            return
        key = name.lower()
        if key in seen_lower:
            return
        seen_lower.add(key)
        patterns.append(EntityPattern(name=name, type=entity_type))

    for name in entity_names:
        _add(name, "entity")
    for _category, regex_pattern in code_patterns.items():
        _add(regex_pattern, "code_pattern")
    for term in domain_terms:
        _add(term, "domain_term")
    return patterns


# Header-like metadata keys that likely contain topic names.
_HEADER_METADATA_KEYS = frozenset({
    "h1", "h2", "h3", "title", "section_header", "article_title",
    "heading", "topic", "subject",
})


def extract_metadata_entities(
    chunks: list[Any],
    *,
    min_chunks: int = 3,
    max_df: float = 0.15,
) -> list[EntityPattern]:
    """Extract entity candidates from chunk header/title metadata.

    Scans metadata for header-like fields (h1-h3, title, article_title, etc.)
    and returns unique values that appear in at least ``min_chunks`` chunks
    but no more than ``max_df`` fraction of all chunks.

    Language-agnostic: works on any corpus with header metadata.
    """
    from collections import Counter  # noqa: PLC0415

    header_counts: Counter[str] = Counter()
    n = len(chunks)

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None)
        if not metadata:
            continue
        seen_in_chunk: set[str] = set()
        items = metadata if isinstance(metadata, (list, tuple)) else []
        for key, value in items:
            if key not in _HEADER_METADATA_KEYS:
                continue
            name = str(value).strip()
            if not name or not _is_quality_entity(name):
                continue
            if name not in seen_in_chunk:
                header_counts[name] += 1
                seen_in_chunk.add(name)

    max_count = int(n * max_df) if n > 0 else 1
    seen_lower: set[str] = set()
    patterns: list[EntityPattern] = []
    for name, count in header_counts.most_common():
        if count < min_chunks:
            break
        if count > max_count:
            continue
        key = name.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        # Metadata headers start as domain_term — the graph scan will compute
        # content DF and quality_score.  Generic headers ("Setup", "FAQ") get
        # penalised by high DF; topic-specific ones ("Feature flags") survive.
        patterns.append(EntityPattern(name=name, type="domain_term"))

    logger.info(
        "Metadata extraction: %d entities from headers (%d unique headers scanned)",
        len(patterns),
        len(header_counts),
    )
    return patterns


_KEYBERT_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class _KeyphraseCandidate:
    name: str
    chunk_count: int
    avg_score: float
    chunk_hashes: set[str]


def _load_keybert_model(model_name: str) -> Any:
    """Lazy-load and cache a KeyBERT model backed by a SentenceTransformer."""
    if model_name in _KEYBERT_MODEL_CACHE:
        return _KEYBERT_MODEL_CACHE[model_name]
    from keybert import KeyBERT  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    st_model = SentenceTransformer(model_name)
    model = KeyBERT(model=st_model)
    _KEYBERT_MODEL_CACHE[model_name] = model
    return model


def _extract_chunk_keyphrases(
    chunks: list[Any],
    model: Any,
    top_k: int,
    chunk_embeddings: dict[str, Any],
    score_threshold: float = 0.4,
) -> dict[str, list[tuple[str, float]]]:
    """Per-chunk KeyBERT extraction.

    Returns {chunk_hash: [(keyphrase, score), ...]}.
    Keyphrases with score < score_threshold or where every token is a stopword are filtered.
    """
    result: dict[str, list[tuple[str, float]]] = {}
    for chunk in chunks:
        chunk_hash = getattr(chunk, "hash", str(id(chunk)))
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if not content.strip():
            result[chunk_hash] = []
            continue

        kwargs: dict[str, Any] = {
            "keyphrase_ngram_range": (1, 2),
            "use_mmr": True,
            "diversity": 0.5,
            "top_n": top_k,
        }
        emb = chunk_embeddings.get(chunk_hash)
        if emb is not None:
            import numpy as np  # noqa: PLC0415

            kwargs["doc_embeddings"] = np.asarray(emb).reshape(1, -1)

        try:
            keywords = model.extract_keywords(content, **kwargs)
        except Exception:
            keywords = []

        filtered = [
            (phrase, score)
            for phrase, score in keywords
            if score >= score_threshold
            and not all(tok in _STOPWORDS for tok in phrase.lower().split())
        ]
        result[chunk_hash] = filtered
    return result


def _aggregate_keyphrases(
    chunk_keyphrases: dict[str, list[tuple[str, float]]],
) -> list[_KeyphraseCandidate]:
    """Group by case-folded keyphrase; track chunk_count, avg_score, chunk_hashes.

    Keeps most common casing as canonical form.
    Returns candidates sorted by chunk_count * avg_score descending.
    """
    grouped: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for chunk_hash, keyphrases in chunk_keyphrases.items():
        for phrase, score in keyphrases:
            grouped[phrase.lower()].append((phrase, score, chunk_hash))

    candidates: list[_KeyphraseCandidate] = []
    for _folded, entries in grouped.items():
        casing_counts: dict[str, int] = {}
        for phrase, _, _ in entries:
            casing_counts[phrase] = casing_counts.get(phrase, 0) + 1
        canonical = max(casing_counts, key=lambda k: casing_counts[k])

        seen_chunks: set[str] = set()
        scores: list[float] = []
        chunk_hashes: set[str] = set()
        for _, score, chunk_hash in entries:
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                scores.append(score)
                chunk_hashes.add(chunk_hash)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        candidates.append(
            _KeyphraseCandidate(
                name=canonical,
                chunk_count=len(chunk_hashes),
                avg_score=avg_score,
                chunk_hashes=chunk_hashes,
            )
        )

    candidates.sort(key=lambda c: c.chunk_count * c.avg_score, reverse=True)
    return candidates


def _filter_notable(
    candidates: list[_KeyphraseCandidate],
    chunk_embeddings: dict[str, Any],
    *,
    min_chunks: int = 3,
    min_clusters: int = 2,
) -> list[_KeyphraseCandidate]:
    """Notability filter: require chunk count + embedding diversity.

    For corpora < 50 chunks: relax min_chunks to 2 and skip diversity clustering.
    """
    import numpy as np  # noqa: PLC0415

    n_chunks = len(chunk_embeddings)
    small_corpus = n_chunks < 50
    effective_min_chunks = 2 if small_corpus else min_chunks

    count_filtered = [c for c in candidates if c.chunk_count >= effective_min_chunks]

    if small_corpus:
        return count_filtered

    from sklearn.cluster import KMeans  # noqa: PLC0415

    result: list[_KeyphraseCandidate] = []
    for candidate in count_filtered:
        embeddings = [
            chunk_embeddings[h] for h in candidate.chunk_hashes if h in chunk_embeddings
        ]
        if len(embeddings) < min_clusters:
            continue
        emb_matrix = np.stack(embeddings)
        n_clusters = min(min_clusters, len(embeddings))
        try:
            km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
            km.fit(emb_matrix)
            if len(set(km.labels_)) >= min_clusters:
                result.append(candidate)
        except Exception:
            result.append(candidate)

    return result


_REGEX_CHARS = re.compile(r"[\\.*+?^${}()|[\]]")


def _candidates_to_entity_patterns(
    candidates: list[_KeyphraseCandidate],
) -> list[EntityPattern]:
    """Convert keyphrase candidates to EntityPattern with type classification.

    - Contains regex-like chars → "code_pattern"
    - Single capitalized word or multi-word with any capitalized word → "entity"
    - Else → "domain_term"
    """
    patterns: list[EntityPattern] = []
    seen_lower: set[str] = set()

    for candidate in candidates:
        name = candidate.name
        key = name.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)

        if _REGEX_CHARS.search(name):
            entity_type = "code_pattern"
        else:
            words = name.split()
            if any(w and w[0].isupper() for w in words):
                entity_type = "entity"
            else:
                entity_type = "domain_term"

        patterns.append(EntityPattern(name=name, type=entity_type))

    return patterns


_TYPE_PRIORITY: dict[str, int] = {"entity": 0, "code_pattern": 1, "domain_term": 2}


def _merge_synonym_entities(
    patterns: list[EntityPattern],
    model: Any,
    *,
    similarity_threshold: float = 0.95,
    max_cluster_size: int = 5,
) -> list[EntityPattern]:
    """Merge synonym entities using embedding similarity.

    Embeds all entity names, clusters by cosine similarity, then per cluster picks
    the canonical name (highest semantic_score, longest name as tiebreak), merges
    type (highest priority: entity > code_pattern > domain_term), and stores
    non-canonical names as aliases for graph scanning.

    Skips clusters larger than max_cluster_size to avoid over-merging.
    """
    if len(patterns) < 2:
        return patterns

    import numpy as np  # noqa: PLC0415
    from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

    names = [p.name for p in patterns]
    try:
        raw = model.model.embed(names)
        embeddings = np.asarray(raw, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(patterns):
            return patterns
    except Exception:
        return patterns

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = (embeddings / norms).astype(np.float64)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        linkage="average",
        metric="cosine",
    )
    try:
        labels = clustering.fit_predict(normalized)
    except Exception:
        return patterns

    # Group patterns by cluster label
    clusters: dict[int, list[EntityPattern]] = {}
    for pattern, label in zip(patterns, labels):
        clusters.setdefault(int(label), []).append(pattern)

    def _numeric_signature(name: str) -> tuple[str, ...]:
        """Extract numbers from a name — entities with different numbers should not merge."""
        return tuple(re.findall(r"\d+", name))

    result: list[EntityPattern] = []
    for cluster_patterns in clusters.values():
        if len(cluster_patterns) > max_cluster_size:
            result.extend(cluster_patterns)
            continue

        if len(cluster_patterns) == 1:
            result.append(cluster_patterns[0])
            continue

        # Sub-split: entities with different numeric content stay separate.
        # "article 21" and "article 22" have similar embeddings but are distinct.
        by_nums: dict[tuple[str, ...], list[EntityPattern]] = {}
        for p in cluster_patterns:
            sig = _numeric_signature(p.name)
            by_nums.setdefault(sig, []).append(p)

        for sub_group in by_nums.values():
            if len(sub_group) == 1:
                result.append(sub_group[0])
                continue

            # Canonical: highest semantic_score, longest name as tiebreak
            canonical = max(sub_group, key=lambda p: (p.semantic_score, len(p.name)))

            # Type: highest priority (lowest numeric value)
            best_type = min(
                (p.type for p in sub_group),
                key=lambda t: _TYPE_PRIORITY.get(t, 99),
            )

            aliases = tuple(p.name for p in sub_group if p.name != canonical.name)
            best_score = max(p.semantic_score for p in sub_group)

            result.append(
                EntityPattern(
                    name=canonical.name,
                    type=best_type,
                    semantic_score=best_score,
                    aliases=aliases,
                )
            )

    return result


def extract_entities(
    chunks: list[Any],
    *,
    top_k_per_chunk: int = 20,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    score_threshold: float = 0.4,
    notability_min_chunks: int = 3,
    diversity_min_clusters: int = 2,
) -> tuple[list[EntityPattern], dict[str, set[str]], dict[str, list[str]]]:
    """Extract entity patterns using KeyBERT + metadata headers.

    Automatically samples a subset of chunks for KeyBERT extraction
    (7% of corpus, floor 500, cap 3000), then runs a cheap O(E×N) scan
    with the discovered entities against the full corpus to build
    the entity↔chunk bipartite graph.

    Also extracts entities from chunk header metadata (h1-h3, title, etc.)
    and merges them with KeyBERT candidates.

    Args:
        chunks: Full corpus chunks. A sample is drawn internally for
            KeyBERT extraction; the full set is used for the graph scan
            and metadata extraction.
        top_k_per_chunk: Max keyphrases to extract per chunk.
        model_name: SentenceTransformer model name for KeyBERT.
        score_threshold: Minimum KeyBERT similarity score to keep a keyphrase.
        notability_min_chunks: Min chunk count a keyphrase must appear in.
        diversity_min_clusters: Min KMeans clusters required for diversity gate.

    Returns:
        (entity_patterns, entity_chunk_index, chunk_entity_index)
        where entity_patterns have document_frequency and quality_score populated.
    """
    if not chunks:
        return [], {}, {}

    import numpy as np  # noqa: PLC0415

    # Auto-sample for KeyBERT extraction
    n_total = len(chunks)
    sample_size = max(500, min(int(n_total * 0.07), 3000))
    if n_total <= sample_size:
        sample_chunks = chunks
    else:
        rng = random.Random(42)
        sample_chunks = rng.sample(chunks, sample_size)

    model = _load_keybert_model(model_name)
    n_sample = len(sample_chunks)

    contents = [
        chunk.content if hasattr(chunk, "content") else str(chunk)
        for chunk in sample_chunks
    ]
    chunk_hashes = [getattr(chunk, "hash", str(id(chunk))) for chunk in sample_chunks]

    raw_embeddings: Any = model.model.embed(contents)
    chunk_embeddings: dict[str, Any] = {
        h: np.asarray(raw_embeddings[i]) for i, h in enumerate(chunk_hashes)
    }

    chunk_keyphrases = _extract_chunk_keyphrases(
        sample_chunks, model, top_k_per_chunk, chunk_embeddings, score_threshold
    )
    candidates = _aggregate_keyphrases(chunk_keyphrases)
    notable = _filter_notable(
        candidates,
        chunk_embeddings,
        min_chunks=notability_min_chunks,
        min_clusters=diversity_min_clusters,
    )

    patterns = _candidates_to_entity_patterns(notable)

    # Populate semantic_score from KeyBERT avg_score
    candidate_by_name: dict[str, _KeyphraseCandidate] = {
        c.name.lower(): c for c in notable
    }
    for pattern in patterns:
        candidate = candidate_by_name.get(pattern.name.lower())
        if candidate is not None:
            pattern.semantic_score = candidate.avg_score

    # Merge metadata-derived entities (headers, titles)
    metadata_patterns = extract_metadata_entities(chunks)
    kb_names = {p.name.lower() for p in patterns}
    for mp in metadata_patterns:
        if mp.name.lower() not in kb_names:
            patterns.append(mp)
            kb_names.add(mp.name.lower())

    # Merge semantic synonyms before scanning (e.g. "feature flag" / "feature flags")
    patterns = _merge_synonym_entities(patterns, model)

    # Cheap full-corpus scan with only the notable entities
    entity_chunk_index, chunk_entity_index, _ = build_entity_chunk_graph(
        patterns, chunks
    )

    logger.info(
        "KeyBERT extraction: %d notable keyphrases from %d chunks",
        len(notable),
        n_sample,
    )
    return patterns, entity_chunk_index, chunk_entity_index


def diverse_profile_sample(
    source: Any,
    corpus_size: int,
    min_chars: int,
    rng: random.Random,
) -> list[Any]:
    """Build a diverse profile sample with stratified + random fill.

    Pool size is proportional to corpus size: 5% of corpus, floor 500, cap 10K.
    Ensures coverage across documents by stratifying on document ID metadata,
    then fills the remaining budget randomly. Degrades gracefully to pure random
    sampling when no document IDs exist.

    Args:
        source: A ``ChunkSource`` instance.
        corpus_size: Total number of chunks in the corpus (from ``source.get_chunk_count()``).
        min_chars: Minimum character count per chunk.
        rng: Seeded random generator for reproducibility.

    Returns:
        List of sampled chunks (up to pool_size).
    """
    pool_size = min(corpus_size, max(500, min(10_000, int(corpus_size * 0.05))))
    oversample = min(pool_size * 3, pool_size + 1000)
    all_chunks = source.sample_chunks(oversample, min_chars=min_chars)
    if not all_chunks:
        return []

    if len(all_chunks) <= pool_size:
        return list(all_chunks)

    # Phase 1: Stratified — ensure coverage across documents
    by_doc: dict[str, list[Any]] = defaultdict(list)
    for chunk in all_chunks:
        doc_id = _get_doc_id(chunk)
        by_doc[doc_id].append(chunk)

    n_docs = len(by_doc)
    stratified_budget = pool_size // 2
    per_doc = max(1, stratified_budget // max(n_docs, 1))

    stratified: list[Any] = []
    for doc_chunks in by_doc.values():
        k = min(per_doc, len(doc_chunks))
        stratified.extend(rng.sample(doc_chunks, k))

    # Phase 2: Random fill from remaining chunks
    selected_ids = {id(c) for c in stratified}
    remaining = [c for c in all_chunks if id(c) not in selected_ids]
    fill_count = min(pool_size - len(stratified), len(remaining))
    if fill_count > 0:
        random_fill = rng.sample(remaining, fill_count)
        stratified.extend(random_fill)

    return stratified[:pool_size]


def detect_search_capabilities(source: Any) -> tuple[set[str], str]:
    """Detect available search modes from a ChunkSource.

    Returns:
        Tuple of (available_modes, best_mode).
    """
    modes: set[str] = {"lexical"}
    try:
        caps = source.get_search_capabilities()
        modes = set(caps.get("modes", {"lexical"}))
    except Exception:
        logger.debug("Could not detect search capabilities, defaulting to lexical")

    if "hybrid" in modes:
        best = "hybrid"
    elif "vector" in modes:
        best = "vector"
    else:
        best = "lexical"

    return modes, best


def _get_doc_id(chunk: Any) -> str:
    """Extract a document identifier from chunk metadata."""
    md: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        md = chunk.metadata_dict or {}
    elif hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            md = raw
        elif isinstance(raw, (tuple, list)):
            md = dict(raw)
    elif isinstance(chunk, dict):
        md = chunk.get("metadata", {})

    for key in ("document_id", "file_name", "file", "source", "doc_id"):
        val = md.get(key)
        if val is not None:
            return str(val)
    return "unknown"


def _get_metadata_dict(chunk: Any) -> dict[str, Any]:
    """Extract metadata dict from any chunk-like object."""
    if hasattr(chunk, "metadata_dict"):
        return chunk.metadata_dict or {}
    if hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (tuple, list)):
            return dict(raw)
    if isinstance(chunk, dict):
        return chunk.get("metadata", {})
    return {}


_MARKUP_PATTERN = re.compile(r"\[.*?\]\(.*?\)|https?://\S+|<[^>]+>|[|>»›‹«]")
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"was this .{0,20}(useful|helpful)",
        r"click\s+here",
        r"subscribe\s*(to|now)",
        r"table of contents",
        r"back to top",
        r"previous\s*[|/]\s*next",
        r"skip to (content|main|nav)",
        r"copyright|©|all rights reserved",
        r"rate this",
        r"share (this|on)",
        r"follow us",
        r"terms (of|and) (service|use)",
    ]
]


def _compute_answerability(content: str) -> float:
    """Score content answerability using language-agnostic heuristics.

    Composite of 4 sub-signals measuring whether the content contains
    substantive, answerable information vs. boilerplate/navigation/stubs.
    Returns float in [0, 1].
    """
    if not content or len(content) < 20:
        return 0.0

    # Sub-signal A: Sentence density (0.35 weight)
    # Latin enders: only count . ! ? when followed by whitespace or end-of-string.
    # Filters code dots (foo.bar), version numbers (1.2.3), URLs (foo.com).
    # CJK/Devanagari enders: always unambiguous, count all.
    latin_enders = len(re.findall(r"[.!?](?=\s|$)", content))
    cjk_enders = len(re.findall(r"[。！？।]", content))
    ender_count = latin_enders + cjk_enders
    enders_per_1k = ender_count / max(len(content) / 1000, 0.001)
    sentence_density = min(1.0, enders_per_1k / 6.0)

    # Sub-signal B: Markup density — inverted (0.30 weight)
    markup_chars = sum(len(m) for m in _MARKUP_PATTERN.findall(content))
    markup_fraction = markup_chars / len(content)
    markup_score = max(0.0, 1.0 - markup_fraction * 3.0)

    # Sub-signal C: Long-line ratio (0.25 weight)
    lines = [line for line in content.split("\n") if line.strip()]
    if lines:
        long_line_ratio = sum(1 for line in lines if len(line.strip()) > 60) / len(lines)
    else:
        long_line_ratio = 0.0

    # Sub-signal D: Negative pattern penalty (0.10 weight)
    neg_matches = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(content))
    negative_score = max(0.0, 1.0 - neg_matches * 0.33)

    return (
        0.35 * sentence_density
        + 0.30 * markup_score
        + 0.25 * long_line_ratio
        + 0.10 * negative_score
    )


_HEADER_KEYS: frozenset[str] = frozenset({"h1", "h2", "h3", "title", "section_header", "header"})
_DOC_ID_KEYS: frozenset[str] = frozenset({"file_name", "document_id", "source", "file", "doc_id"})
_DATE_KEYS: frozenset[str] = frozenset(
    {
        "date", "timestamp", "created_at", "updated_at", "published_at",
        "modified_at", "date_start", "date_end",
    }
)
_SEQ_KEYS: frozenset[str] = frozenset({"prev_chunk_id", "next_chunk_id"})
_INDEX_KEYS: frozenset[str] = frozenset({"chunk_index", "index", "position"})


def compute_header_prevalence(pool: list[Any]) -> float:
    """Compute fraction of chunks with header metadata. Lightweight, one pass.

    Used before full entity extraction to derive metadata_regime for
    ``select_diverse``. Returns 0.0 for an empty pool.
    """
    if not pool:
        return 0.0
    count = sum(
        1 for chunk in pool if any(_get_metadata_dict(chunk).get(k) for k in _HEADER_KEYS)
    )
    return count / len(pool)


def compute_metadata_census(
    pool: list[Any],
    entity_names: list[str],
    chunk_count: int,
) -> CorpusMetadataCensus:
    """Compute corpus-relative distributions from the profile pool.

    Single pass over *pool* computing metadata presence, content length,
    entity density, and lexical diversity. Uses ``statistics.quantiles``
    for p25/p50/p75. Returns a degenerate census (zeroed percentiles,
    regime="unstructured") if pool has fewer than 5 chunks.

    Args:
        pool: Profile pool chunks.
        entity_names: Extracted entity name strings (used for density).
        chunk_count: Total corpus size (stored on the census).

    Returns:
        A populated ``CorpusMetadataCensus``.
    """
    if len(pool) < 5:
        return CorpusMetadataCensus(
            header_prevalence=0.0,
            doc_id_prevalence=0.0,
            date_prevalence=0.0,
            sequential_prevalence=0.0,
            content_length_p25=0,
            content_length_p50=0,
            content_length_p75=0,
            entity_density_p25=0.0,
            entity_density_p50=0.0,
            entity_density_p75=0.0,
            lexical_diversity_p25=0.0,
            lexical_diversity_p50=0.0,
            lexical_diversity_p75=0.0,
            metadata_regime="unstructured",
            chunk_count=chunk_count,
        )

    entity_names_lower = [name.lower() for name in entity_names]

    header_count = 0
    doc_id_count = 0
    date_count = 0
    sequential_count = 0
    content_lengths: list[int] = []
    entity_densities: list[float] = []
    lexical_diversities: list[float] = []

    for chunk in pool:
        md = _get_metadata_dict(chunk)
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        content_lower = content.lower()

        if any(md.get(k) for k in _HEADER_KEYS):
            header_count += 1
        if any(md.get(k) for k in _DOC_ID_KEYS):
            doc_id_count += 1
        if any(md.get(k) for k in _DATE_KEYS):
            date_count += 1

        has_prev_next = bool(md.get("prev_chunk_id") or md.get("next_chunk_id"))
        has_file_index = bool(
            any(md.get(k) for k in _DOC_ID_KEYS) and any(md.get(k) for k in _INDEX_KEYS)
        )
        if has_prev_next or has_file_index:
            sequential_count += 1

        length = len(content)
        content_lengths.append(length)

        entity_count = sum(1 for name in entity_names_lower if name in content_lower)
        entity_densities.append(entity_count / max(length / 1000, 0.001))

        tokens = re.findall(r"\w+", content_lower)
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        lexical_diversities.append(unique_tokens / max(total_tokens, 1))

    n = len(pool)
    header_prevalence = header_count / n
    doc_id_prevalence = doc_id_count / n
    date_prevalence = date_count / n
    sequential_prevalence = sequential_count / n

    cl_q = statistics.quantiles(content_lengths, n=4)
    ed_q = statistics.quantiles(entity_densities, n=4)
    ld_q = statistics.quantiles(lexical_diversities, n=4)

    if header_prevalence > 0.70:
        regime = "structured"
    elif header_prevalence >= 0.20:
        regime = "mixed"
    elif doc_id_prevalence > 0.5 and date_prevalence > 0.5:
        regime = "mixed"  # no headers but rich metadata (e.g. email)
    else:
        regime = "unstructured"

    return CorpusMetadataCensus(
        header_prevalence=header_prevalence,
        doc_id_prevalence=doc_id_prevalence,
        date_prevalence=date_prevalence,
        sequential_prevalence=sequential_prevalence,
        content_length_p25=int(cl_q[0]),
        content_length_p50=int(cl_q[1]),
        content_length_p75=int(cl_q[2]),
        entity_density_p25=ed_q[0],
        entity_density_p50=ed_q[1],
        entity_density_p75=ed_q[2],
        lexical_diversity_p25=ld_q[0],
        lexical_diversity_p50=ld_q[1],
        lexical_diversity_p75=ld_q[2],
        metadata_regime=regime,
        chunk_count=chunk_count,
    )


def select_diverse(
    pool: list[Any],
    n: int,
    *,
    rng: random.Random,
    stratify_key: Callable[[Any], str] | None = None,
    min_jaccard_distance: float = 0.3,
) -> list[Any]:
    """Select *n* maximally diverse chunks from *pool* using greedy max-diversity.

    Precomputes token sets and seeds with the broadest-vocabulary chunk.
    Each subsequent selection maximises the minimum Jaccard distance to
    already-selected chunks. Candidates below *min_jaccard_distance* from
    any selected chunk are skipped; any remaining budget is filled with
    leftover chunks in arbitrary order.

    When *stratify_key* is provided the pool is grouped by key and
    ``ceil(n / num_groups)`` chunks are selected per group, then the
    result is trimmed or filled to exactly *n*.

    Args:
        pool: Source chunks to select from.
        n: Target number of chunks to return.
        rng: Seeded random generator (used for fill-phase shuffles).
        stratify_key: Optional callable that maps a chunk to a group label.
        min_jaccard_distance: Diversity floor; candidates closer than this
            to any already-selected chunk are skipped in the greedy pass.

    Returns:
        List of selected chunks (may be shorter than *n* if pool is small).
    """
    if not pool or n <= 0:
        return []
    if len(pool) <= n:
        return list(pool)

    def _token_set(chunk: Any) -> frozenset[str]:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        return frozenset(re.findall(r"\w+", content.lower()))

    def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - len(a & b) / union

    def _greedy_diverse(candidates: list[Any], k: int) -> list[Any]:
        if not candidates or k <= 0:
            return []
        if len(candidates) <= k:
            return list(candidates)

        # Precompute token sets — not recomputed inside the loop
        token_sets = [_token_set(c) for c in candidates]

        # Seed with the chunk having the most unique tokens
        seed_idx = max(range(len(candidates)), key=lambda i: len(token_sets[i]))
        selected: list[Any] = [candidates[seed_idx]]
        selected_sets: list[frozenset[str]] = [token_sets[seed_idx]]
        remaining: set[int] = set(range(len(candidates)))
        remaining.discard(seed_idx)

        while len(selected) < k and remaining:
            best_idx: int | None = None
            best_min_dist = -1.0

            for ri in remaining:
                min_dist = min(_jaccard_distance(token_sets[ri], s) for s in selected_sets)
                if min_dist < min_jaccard_distance:
                    continue
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = ri

            if best_idx is None:
                break  # All remaining candidates below diversity threshold

            selected.append(candidates[best_idx])
            selected_sets.append(token_sets[best_idx])
            remaining.discard(best_idx)

        # Fill remaining budget from leftover chunks in arbitrary order
        if len(selected) < k and remaining:
            fill = [candidates[i] for i in remaining]
            rng.shuffle(fill)
            selected.extend(fill[: k - len(selected)])

        return selected

    if stratify_key is not None:
        by_key: dict[str, list[Any]] = defaultdict(list)
        for chunk in pool:
            by_key[stratify_key(chunk)].append(chunk)

        num_groups = len(by_key)
        per_group = math.ceil(n / num_groups)

        result: list[Any] = []
        used_ids: set[int] = set()
        for group_chunks in by_key.values():
            for c in _greedy_diverse(group_chunks, per_group):
                if id(c) not in used_ids:
                    result.append(c)
                    used_ids.add(id(c))

        if len(result) > n:
            return result[:n]
        if len(result) < n:
            remaining_chunks = [c for c in pool if id(c) not in used_ids]
            rng.shuffle(remaining_chunks)
            result.extend(remaining_chunks[: n - len(result)])
        return result

    return _greedy_diverse(pool, n)


def _percentile_normalize(value: float, p25: float, p50: float, p75: float) -> float:
    """Map *value* to [0, 1] using census percentile boundaries.

    Division-by-zero is guarded with ``max(..., 1e-9)`` so degenerate censuses
    (all-zero percentiles) return a safe value rather than crashing.
    """
    eps = 1e-9
    if value <= p25:
        return value / max(p25, eps) * 0.25
    if value <= p50:
        return 0.25 + (value - p25) / max(p50 - p25, eps) * 0.25
    if value <= p75:
        return 0.50 + (value - p50) / max(p75 - p50, eps) * 0.25
    return min(1.0, 0.75 + (value - p75) / max(p75, eps) * 0.25)


def compute_chunk_suitability(
    chunk: Any,
    census: CorpusMetadataCensus,
    profile: CorpusProfile,
) -> float:
    """Score a chunk's suitability for QA generation relative to corpus distributions.

    Combines five signals — metadata presence, entity density, lexical diversity,
    content length, and TF-IDF distinctiveness — using regime-dependent weight
    vectors. Returns a float in [0, 1].

    Args:
        chunk: The chunk to score.
        census: Precomputed corpus metadata census for normalization.
        profile: Corpus profile supplying entity names and TF-IDF data.

    Returns:
        Suitability score in [0, 1].
    """
    md = _get_metadata_dict(chunk)
    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    content_lower = content.lower()

    # 1. Metadata richness: count of distinct metadata categories [0, 1]
    richness = 0
    if any(md.get(k) for k in _HEADER_KEYS):
        richness += 1
    if any(md.get(k) for k in _DOC_ID_KEYS):
        richness += 1
    if any(md.get(k) for k in _DATE_KEYS):
        richness += 1
    has_seq = bool(md.get("prev_chunk_id") or md.get("next_chunk_id"))
    has_file_idx = bool(
        any(md.get(k) for k in _DOC_ID_KEYS) and any(md.get(k) for k in _INDEX_KEYS)
    )
    if has_seq or has_file_idx:
        richness += 1
    metadata_score = richness / 4.0

    # 2. Entity density: quality-weighted entities per 1000 chars, percentile-normalised
    chunk_hash = getattr(chunk, "hash", str(id(chunk)))
    eq_map = profile.entity_quality_map
    if chunk_hash in profile.chunk_entity_index:
        # Fast path: read from pre-computed entity-chunk graph
        chunk_entities = profile.chunk_entity_index[chunk_hash]
        quality_weighted_count = sum(eq_map.get(e.lower(), 0.5) for e in chunk_entities)
    else:
        # Fallback: inline scan for chunks not in the graph (API backends, on-the-fly)
        quality_weighted_count = sum(
            score for name, score in eq_map.items() if name in content_lower
        )
    raw_entity_density = quality_weighted_count / max(len(content) / 1000, 0.001)
    entity_density = _percentile_normalize(
        raw_entity_density,
        census.entity_density_p25,
        census.entity_density_p50,
        census.entity_density_p75,
    )

    # 3. Lexical diversity: unique tokens / total tokens, percentile-normalised
    tokens = re.findall(r"\w+", content_lower)
    raw_diversity = len(set(tokens)) / max(len(tokens), 1)
    lexical_diversity = _percentile_normalize(
        raw_diversity,
        census.lexical_diversity_p25,
        census.lexical_diversity_p50,
        census.lexical_diversity_p75,
    )

    # 4. Content length: percentile-normalised (saturating curve handles outliers)
    raw_length = float(len(content))
    content_length_score = _percentile_normalize(
        raw_length,
        float(census.content_length_p25),
        float(census.content_length_p50),
        float(census.content_length_p75),
    )

    # 5. TF-IDF distinctiveness: mean IDF of top-5 keyphrases
    keyphrases = profile.get_tfidf_keyphrases(content, top_n=5)
    if keyphrases and profile.token_df_sample_size > 0:
        n_docs = profile.token_df_sample_size
        idfs = [
            math.log(n_docs / max(profile.token_document_frequency.get(kp, 1), 1))
            for kp in keyphrases
        ]
        mean_idf = sum(idfs) / len(idfs)
        max_idf = math.log(max(profile.token_df_sample_size, 2))
        tfidf_distinctiveness = min(1.0, mean_idf / max(max_idf, 0.001))
    else:
        tfidf_distinctiveness = 0.0

    # 6. Answerability: content vs. boilerplate/navigation/stubs
    answerability = _compute_answerability(content)

    # Regime-dependent weight vectors (6 signals)
    regime = census.metadata_regime
    if regime == "structured":
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.20, 0.20, 0.10, 0.10, 0.15, 0.25
        )
    elif regime == "mixed" and census.header_prevalence < 0.20:
        # Email-like: "mixed" via date/doc_id, not headers.
        # Answerability is likely uniform for uniform-prose corpora.
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.10, 0.25, 0.20, 0.10, 0.20, 0.15
        )
    elif regime == "mixed":
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.10, 0.20, 0.15, 0.10, 0.15, 0.30
        )
    else:  # unstructured
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.00, 0.25, 0.20, 0.10, 0.20, 0.25
        )

    return (
        w_meta * metadata_score
        + w_ent * entity_density
        + w_lex * lexical_diversity
        + w_len * content_length_score
        + w_tfidf * tfidf_distinctiveness
        + w_ans * answerability
    )
