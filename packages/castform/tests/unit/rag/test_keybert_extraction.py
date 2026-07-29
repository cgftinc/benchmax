"""Tests for KeyBERT-based entity extraction."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from castform.rag.qa_generation.corpus_profile import (
    EntityPattern,
    _aggregate_keyphrases,
    _candidates_to_entity_patterns,
    _extract_chunk_keyphrases,
    _filter_notable,
    _KeyphraseCandidate,
    _merge_synonym_entities,
    build_entity_chunk_graph,
    extract_entities,
)


@dataclass
class FakeChunk:
    content: str
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = str(id(self))


def _make_mock_model(embedding_dim: int = 384) -> MagicMock:
    """Return a KeyBERT mock whose .model.encode() and .model.embed() return random embeddings."""
    model = MagicMock()
    rng = np.random.default_rng(42)

    def fake_encode(texts: list[str], **kwargs: object) -> np.ndarray:
        return rng.random((len(texts), embedding_dim)).astype(np.float32)

    def fake_embed(texts: list[str], **kwargs: object) -> np.ndarray:
        return rng.random((len(texts), embedding_dim)).astype(np.float32)

    model.model.encode = fake_encode
    model.model.embed = fake_embed
    return model


# ---------------------------------------------------------------------------
# _aggregate_keyphrases
# ---------------------------------------------------------------------------


class TestAggregateKeyphrases:
    def test_deduplicates_across_chunks(self) -> None:
        chunk_keyphrases = {
            "c1": [("Machine Learning", 0.9), ("neural network", 0.7)],
            "c2": [("machine learning", 0.8), ("Deep Learning", 0.6)],
            "c3": [("Machine Learning", 0.85)],
        }
        candidates = _aggregate_keyphrases(chunk_keyphrases)

        names = {c.name.lower() for c in candidates}
        assert "machine learning" in names

        ml = next(c for c in candidates if c.name.lower() == "machine learning")
        assert ml.chunk_count == 3
        assert ml.avg_score == pytest.approx((0.9 + 0.8 + 0.85) / 3, rel=1e-5)

    def test_most_common_casing_wins(self) -> None:
        chunk_keyphrases = {
            "c1": [("Redis", 0.8)],
            "c2": [("Redis", 0.7)],
            "c3": [("redis", 0.6)],
        }
        candidates = _aggregate_keyphrases(chunk_keyphrases)
        redis = next(c for c in candidates if c.name.lower() == "redis")
        assert redis.name == "Redis"

    def test_sorted_by_count_times_score(self) -> None:
        chunk_keyphrases = {
            "c1": [("rare term", 0.9)],
            "c2": [("common term", 0.5)],
            "c3": [("common term", 0.5)],
            "c4": [("common term", 0.5)],
        }
        candidates = _aggregate_keyphrases(chunk_keyphrases)
        assert candidates[0].name.lower() == "common term"

    def test_empty_input_returns_empty(self) -> None:
        assert _aggregate_keyphrases({}) == []

    def test_single_chunk_single_keyphrase(self) -> None:
        result = _aggregate_keyphrases({"c1": [("Kubernetes", 0.95)]})
        assert len(result) == 1
        assert result[0].name == "Kubernetes"
        assert result[0].chunk_count == 1


# ---------------------------------------------------------------------------
# _candidates_to_entity_patterns
# ---------------------------------------------------------------------------


class TestCandidatesToEntityPatterns:
    def _make(self, name: str) -> _KeyphraseCandidate:
        return _KeyphraseCandidate(
            name=name, chunk_count=3, avg_score=0.7, chunk_hashes={"a", "b", "c"}
        )

    def test_capitalized_word_is_entity(self) -> None:
        patterns = _candidates_to_entity_patterns([self._make("Redis")])
        assert patterns[0].type == "entity"

    def test_multi_word_with_capitals_is_entity(self) -> None:
        patterns = _candidates_to_entity_patterns([self._make("Machine Learning")])
        assert patterns[0].type == "entity"

    def test_lowercase_multi_word_is_domain_term(self) -> None:
        patterns = _candidates_to_entity_patterns([self._make("feature flags")])
        assert patterns[0].type == "domain_term"

    def test_regex_chars_become_code_pattern(self) -> None:
        patterns = _candidates_to_entity_patterns([self._make(r"posthog\.\w+")])
        assert patterns[0].type == "code_pattern"

    def test_deduplicates_case_insensitive(self) -> None:
        candidates = [self._make("Redis"), self._make("redis")]
        patterns = _candidates_to_entity_patterns(candidates)
        assert len(patterns) == 1


# ---------------------------------------------------------------------------
# _filter_notable
# ---------------------------------------------------------------------------


class TestFilterNotable:
    def _make_candidate(self, name: str, chunk_hashes: set[str]) -> _KeyphraseCandidate:
        return _KeyphraseCandidate(
            name=name,
            chunk_count=len(chunk_hashes),
            avg_score=0.8,
            chunk_hashes=chunk_hashes,
        )

    def test_min_chunks_filters_rare_candidates(self) -> None:
        rng = np.random.default_rng(0)
        chunk_embeddings = {f"c{i}": rng.random(384).astype(np.float32) for i in range(100)}

        common = self._make_candidate("common", set(list(chunk_embeddings.keys())[:10]))
        rare = self._make_candidate("rare", {"c0", "c1"})

        result = _filter_notable([common, rare], chunk_embeddings, min_chunks=3, min_clusters=2)
        names = {c.name for c in result}
        assert "common" in names
        assert "rare" not in names

    def test_small_corpus_relaxes_min_chunks(self) -> None:
        rng = np.random.default_rng(0)
        chunk_embeddings = {f"c{i}": rng.random(384).astype(np.float32) for i in range(30)}

        candidate = self._make_candidate("term", {"c0", "c1"})
        result = _filter_notable([candidate], chunk_embeddings, min_chunks=3, min_clusters=2)
        assert len(result) == 1

    def test_small_corpus_skips_diversity_check(self) -> None:
        # All embeddings identical → diversity would fail, but small corpus skips it
        emb = np.ones(384, dtype=np.float32)
        chunk_embeddings = {f"c{i}": emb for i in range(20)}
        candidate = self._make_candidate("term", {"c0", "c1"})
        result = _filter_notable([candidate], chunk_embeddings, min_chunks=2, min_clusters=2)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _extract_chunk_keyphrases (with mocked model)
# ---------------------------------------------------------------------------


class TestExtractChunkKeyphrases:
    def test_returns_per_chunk_keyphrases(self) -> None:
        chunks = [
            FakeChunk("Redis is used for caching"),
            FakeChunk("Kubernetes orchestrates containers"),
        ]
        rng = np.random.default_rng(42)
        chunk_embeddings = {c.hash: rng.random(384).astype(np.float32) for c in chunks}

        mock_model = MagicMock()
        mock_model.extract_keywords.return_value = [("Redis", 0.9), ("caching", 0.7)]

        result = _extract_chunk_keyphrases(
            chunks, mock_model, top_k=5, chunk_embeddings=chunk_embeddings
        )

        assert set(result.keys()) == {c.hash for c in chunks}
        for keyphrases in result.values():
            assert isinstance(keyphrases, list)

    def test_stopword_only_keyphrases_filtered(self) -> None:
        chunks = [FakeChunk("some content here")]
        chunk_embeddings = {chunks[0].hash: np.random.default_rng(0).random(384).astype(np.float32)}

        mock_model = MagicMock()
        mock_model.extract_keywords.return_value = [
            ("the a", 0.9),  # all stopwords → filtered even though score passes
            ("Redis cache", 0.8),  # not all stopwords → kept
        ]

        result = _extract_chunk_keyphrases(
            chunks, mock_model, top_k=5, chunk_embeddings=chunk_embeddings
        )
        keyphrases = result[chunks[0].hash]
        names = [p for p, _ in keyphrases]
        assert "the a" not in names
        assert "Redis cache" in names

    def test_score_threshold_filters_low_scores(self) -> None:
        chunks = [FakeChunk("some content here")]
        chunk_embeddings: dict[str, np.ndarray] = {}
        mock_model = MagicMock()
        mock_model.extract_keywords.return_value = [
            ("below threshold", 0.3),  # score < 0.4 → filtered
            ("above threshold", 0.5),  # score >= 0.4 → kept
        ]

        result = _extract_chunk_keyphrases(
            chunks, mock_model, top_k=5, chunk_embeddings=chunk_embeddings, score_threshold=0.4
        )
        names = [p for p, _ in result[chunks[0].hash]]
        assert "below threshold" not in names
        assert "above threshold" in names

    def test_empty_content_produces_empty_list(self) -> None:
        chunks = [FakeChunk("")]
        chunk_embeddings: dict[str, np.ndarray] = {}
        mock_model = MagicMock()

        result = _extract_chunk_keyphrases(
            chunks, mock_model, top_k=5, chunk_embeddings=chunk_embeddings
        )
        assert result[chunks[0].hash] == []


# ---------------------------------------------------------------------------
# extract_entities (integration with mock model)
# ---------------------------------------------------------------------------


class TestExtractKeybertEntities:
    def test_returns_entity_patterns(self) -> None:
        chunks = [
            FakeChunk(f"Redis is a caching system used in production environment chunk {i}")
            for i in range(60)
        ]
        mock_model = _make_mock_model()
        mock_model.extract_keywords = MagicMock(
            return_value=[("Redis", 0.9), ("caching system", 0.7), ("production environment", 0.6)]
        )

        with patch(
            "castform.rag.qa_generation.corpus_profile._load_keybert_model", return_value=mock_model
        ):
            patterns, entity_chunk_idx, chunk_entity_idx = extract_entities(
                chunks, notability_min_chunks=3
            )

        assert isinstance(patterns, list)
        assert all(isinstance(p, EntityPattern) for p in patterns)
        assert isinstance(entity_chunk_idx, dict)
        assert isinstance(chunk_entity_idx, dict)

    def test_empty_chunks_returns_empty(self) -> None:
        patterns, entity_chunk_idx, chunk_entity_idx = extract_entities([])
        assert patterns == []
        assert entity_chunk_idx == {}
        assert chunk_entity_idx == {}

    def test_type_classification_applied(self) -> None:
        chunks = [
            FakeChunk(f"Kubernetes orchestrates containers in cloud environments {i}")
            for i in range(60)
        ]
        mock_model = _make_mock_model()
        mock_model.extract_keywords = MagicMock(
            return_value=[("Kubernetes", 0.95), ("cloud environments", 0.7)]
        )

        with patch(
            "castform.rag.qa_generation.corpus_profile._load_keybert_model", return_value=mock_model
        ):
            patterns, _, _ = extract_entities(chunks, notability_min_chunks=3)

        types = {p.type for p in patterns}
        assert types <= {"entity", "domain_term", "code_pattern"}

    def test_df_and_quality_populated(self) -> None:
        chunks = [FakeChunk(f"Redis caching layer chunk {i}") for i in range(60)]
        mock_model = _make_mock_model()
        mock_model.extract_keywords = MagicMock(return_value=[("Redis", 0.9)])

        with patch(
            "castform.rag.qa_generation.corpus_profile._load_keybert_model", return_value=mock_model
        ):
            patterns, entity_chunk_idx, chunk_entity_idx = extract_entities(
                chunks, notability_min_chunks=3
            )

        assert len(patterns) >= 1
        for p in patterns:
            assert p.document_frequency > 0.0
            assert p.quality_score > 0.0
        assert len(entity_chunk_idx) == len(patterns)
        assert len(chunk_entity_idx) > 0


# ---------------------------------------------------------------------------
# _merge_synonym_entities
# ---------------------------------------------------------------------------


def _make_embedding_model(name_to_embedding: dict[str, np.ndarray]) -> MagicMock:
    """Return a KeyBERT mock whose .model.embed() returns controlled embeddings by name."""
    model = MagicMock()
    rng = np.random.default_rng(0)
    dim = next(iter(name_to_embedding.values())).shape[0]

    def fake_embed(texts: list[str], **kwargs: object) -> np.ndarray:
        rows = []
        for t in texts:
            if t in name_to_embedding:
                rows.append(name_to_embedding[t])
            else:
                rows.append(rng.random(dim).astype(np.float32))
        return np.stack(rows)

    model.model.embed = fake_embed
    return model


class TestMergeSynonymEntities:
    def _make_pattern(self, name: str, semantic_score: float = 0.7) -> EntityPattern:
        return EntityPattern(name=name, type="domain_term", semantic_score=semantic_score)

    def test_synonyms_merge_into_one_with_alias(self) -> None:
        # "feature flags" and "feature flag" should merge (nearly identical embeddings)
        dim = 16
        base = np.ones(dim, dtype=np.float32)
        similar = base + np.array([0.01] + [0.0] * (dim - 1), dtype=np.float32)
        embeddings = {
            "feature flags": base / np.linalg.norm(base),
            "feature flag": similar / np.linalg.norm(similar),
        }
        model = _make_embedding_model(embeddings)

        patterns = [
            EntityPattern(name="feature flags", type="domain_term", semantic_score=0.8),
            EntityPattern(name="feature flag", type="domain_term", semantic_score=0.7),
        ]
        result = _merge_synonym_entities(patterns, model, similarity_threshold=0.82)

        assert len(result) == 1
        canonical = result[0]
        # Highest semantic_score wins
        assert canonical.name == "feature flags"
        assert "feature flag" in canonical.aliases

    def test_dissimilar_entities_do_not_merge(self) -> None:
        # "Redis" and "PostgreSQL" are orthogonal — cosine similarity = 0
        dim = 16
        redis_emb = np.zeros(dim, dtype=np.float32)
        redis_emb[0] = 1.0
        postgres_emb = np.zeros(dim, dtype=np.float32)
        postgres_emb[1] = 1.0
        embeddings = {"Redis": redis_emb, "PostgreSQL": postgres_emb}
        model = _make_embedding_model(embeddings)

        patterns = [
            EntityPattern(name="Redis", type="entity", semantic_score=0.9),
            EntityPattern(name="PostgreSQL", type="entity", semantic_score=0.85),
        ]
        result = _merge_synonym_entities(patterns, model, similarity_threshold=0.82)

        assert len(result) == 2
        names = {p.name for p in result}
        assert "Redis" in names
        assert "PostgreSQL" in names
        for p in result:
            assert p.aliases == ()

    def test_max_cluster_size_guard(self) -> None:
        # 6 very similar patterns — should NOT be merged because cluster > max_cluster_size=5
        dim = 16
        base = np.ones(dim, dtype=np.float32)
        base /= np.linalg.norm(base)
        embeddings = {
            f"term{i}": base + np.array([0.001 * i] + [0.0] * (dim - 1), dtype=np.float32)
            for i in range(6)
        }
        # Normalize
        for k, v in embeddings.items():
            embeddings[k] = v / np.linalg.norm(v)
        model = _make_embedding_model(embeddings)

        patterns = [
            EntityPattern(name=f"term{i}", type="domain_term", semantic_score=0.7) for i in range(6)
        ]
        result = _merge_synonym_entities(
            patterns, model, similarity_threshold=0.82, max_cluster_size=5
        )

        # All 6 kept because cluster > max_cluster_size
        assert len(result) == 6

    def test_single_pattern_returned_unchanged(self) -> None:
        model = MagicMock()
        patterns = [EntityPattern(name="Redis", type="entity", semantic_score=0.9)]
        result = _merge_synonym_entities(patterns, model)
        assert len(result) == 1
        assert result[0].name == "Redis"

    def test_empty_input_returns_empty(self) -> None:
        model = MagicMock()
        assert _merge_synonym_entities([], model) == []

    def test_type_priority_entity_wins(self) -> None:
        # When merging, "entity" type wins over "domain_term"
        dim = 16
        base = np.ones(dim, dtype=np.float32)
        similar = base + np.array([0.01] + [0.0] * (dim - 1), dtype=np.float32)
        embeddings = {
            "Feature Flags": base / np.linalg.norm(base),
            "feature flags": similar / np.linalg.norm(similar),
        }
        model = _make_embedding_model(embeddings)

        patterns = [
            EntityPattern(name="Feature Flags", type="entity", semantic_score=0.6),
            EntityPattern(name="feature flags", type="domain_term", semantic_score=0.8),
        ]
        result = _merge_synonym_entities(patterns, model, similarity_threshold=0.82)

        assert len(result) == 1
        # canonical chosen by semantic_score: "feature flags" (0.8 > 0.6)
        assert result[0].name == "feature flags"
        # type: "entity" wins (higher priority)
        assert result[0].type == "entity"


# ---------------------------------------------------------------------------
# aliases checked in build_entity_chunk_graph
# ---------------------------------------------------------------------------


class TestBuildEntityChunkGraphAliases:
    def test_aliases_matched_in_graph_scan(self) -> None:
        # "feature flags" (canonical) with alias "feature flag"
        # Chunk contains "feature flag" (the alias) but not "feature flags"
        chunk = FakeChunk("enable feature flag for new users")
        pattern = EntityPattern(
            name="feature flags",
            type="domain_term",
            aliases=("feature flag",),
        )
        entity_idx, chunk_idx, _ = build_entity_chunk_graph([pattern], [chunk])

        assert chunk.hash in entity_idx.get("feature flags", set())
        assert chunk.hash in chunk_idx

    def test_canonical_name_still_matched(self) -> None:
        chunk = FakeChunk("feature flags are toggled at runtime")
        pattern = EntityPattern(
            name="feature flags",
            type="domain_term",
            aliases=("feature flag",),
        )
        entity_idx, chunk_idx, _ = build_entity_chunk_graph([pattern], [chunk])

        assert chunk.hash in entity_idx.get("feature flags", set())

    def test_chunk_without_name_or_alias_not_matched(self) -> None:
        chunk = FakeChunk("completely unrelated content about databases")
        pattern = EntityPattern(
            name="feature flags",
            type="domain_term",
            aliases=("feature flag",),
        )
        entity_idx, _, _ = build_entity_chunk_graph([pattern], [chunk])

        assert chunk.hash not in entity_idx.get("feature flags", set())
