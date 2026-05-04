"""Tests for the hop-count validity filter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from benchmax.rag.qa_generation.filters.hop_count_validity import (
    HopCountValidityConfig,
    HopCountValidityFilter,
)
from benchmax.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA


def _make_context(max_refinements: int = 2) -> MagicMock:
    ctx = MagicMock()
    ctx.config.refinement.max_refinements_per_item = max_refinements
    ctx.setdefault = lambda key, default: default
    return ctx


def _make_item(
    *,
    question: str = "How do A and B interact?",
    answer: str = "A produces X, B consumes X.",
    qa_type: str = "multi_hop",
    chunks: list[dict] | None = None,
    refinement_count: int = 0,
    verdict: FilterVerdict | None = None,
) -> GeneratedQA:
    if chunks is None:
        chunks = [
            {"id": "c1", "metadata": {}, "content": "A produces X."},
            {"id": "c2", "metadata": {}, "content": "B consumes X."},
        ]
    return GeneratedQA(
        qa={
            "question": question,
            "answer": answer,
            "qa_type": qa_type,
            "reference_chunks": chunks,
            "min_hop_count": len(chunks),
            "is_co_located": None,
            "filter_status": None,
            "filter_reasoning": None,
            "no_context_answer": None,
            "eval_scores": {},
        },
        generation_metadata={
            "qa_type_target": qa_type,
            "refinement_count": refinement_count,
        },
        filter_verdict=verdict,
    )


def _mock_judge_response(
    answerable: bool,
    confidence: float = 0.9,
    missing_facts: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "confidence": confidence,
            "reasoning": "test reasoning",
            "missing_facts": missing_facts or [],
        }
    )


class TestHopCountValidityFilter:
    def _make_filter(self, mode: str = "primary_only") -> HopCountValidityFilter:
        return HopCountValidityFilter(
            cfg=HopCountValidityConfig(
                enabled=True,
                mode=mode,
                judge_model="test-model",
                judge_api_key="test-key",
                judge_base_url="http://test",
                batch_enabled=False,
            )
        )

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_passes_when_chunk_is_essential(self, mock_judge):
        mock_judge.return_value = {
            "answerable": False,
            "confidence": 0.9,
            "reasoning": "Missing fact X from omitted chunk",
            "missing_facts": ["X from c2"],
        }
        filt = self._make_filter()
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].is_passed
        meta = result[0].filter_verdict.metadata
        assert meta["hop_count_validated"] is True

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_fails_when_chunk_is_redundant(self, mock_judge):
        # leave_one_out mode: a subset that is still answerable means that chunk is redundant.
        mock_judge.return_value = {
            "answerable": True,
            "confidence": 0.95,
            "reasoning": "Can answer without one of the chunks",
            "missing_facts": [],
        }
        filt = self._make_filter(mode="leave_one_out")
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].needs_refinement
        meta = result[0].filter_verdict.metadata
        assert meta["hop_count_validated"] is False
        assert len(meta["redundant_chunks"]) > 0

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_rejects_after_max_refinements(self, mock_judge):
        # leave_one_out mode: rejection after exhausting refinements.
        mock_judge.return_value = {
            "answerable": True,
            "confidence": 0.95,
            "reasoning": "Redundant",
            "missing_facts": [],
        }
        filt = self._make_filter(mode="leave_one_out")
        item = _make_item(refinement_count=2)
        ctx = _make_context(max_refinements=2)

        result = filt.evaluate([item], ctx)
        assert result[0].is_rejected

    def test_skips_lookup_type(self):
        filt = self._make_filter()
        item = _make_item(qa_type="lookup")
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].filter_verdict is None

    def test_skips_single_chunk_items(self):
        filt = self._make_filter()
        item = _make_item(chunks=[{"id": "c1", "metadata": {}, "content": "Only one."}])
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].filter_verdict is None

    def test_skips_already_rejected_items(self):
        filt = self._make_filter()
        rejected_verdict = FilterVerdict(
            status="rejected",
            reason="grounding_rejected",
            reasoning="Not grounded",
        )
        item = _make_item(verdict=rejected_verdict)
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].filter_verdict.reason == "grounding_rejected"

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_leave_one_out_mode(self, mock_judge):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "answerable": False,
                "confidence": 0.9,
                "reasoning": "Essential",
                "missing_facts": [f"fact_{call_count}"],
            }

        mock_judge.side_effect = side_effect
        filt = self._make_filter(mode="leave_one_out")
        item = _make_item(
            chunks=[
                {"id": "c1", "metadata": {}, "content": "Chunk A content."},
                {"id": "c2", "metadata": {}, "content": "Chunk B content."},
                {"id": "c3", "metadata": {}, "content": "Chunk C content."},
            ]
        )
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].is_passed
        meta = result[0].filter_verdict.metadata
        assert meta["subsets_tested"] == 3
        assert call_count == 3

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_primary_only_single_hop_skips_filter(self, mock_judge):
        # When the primary chunk alone suffices, the question is single-hop.
        # The multi-hop validator should skip it (no verdict) rather than flag it.
        mock_judge.return_value = {
            "answerable": True,
            "confidence": 0.9,
            "reasoning": "Can answer with primary chunk alone",
            "missing_facts": [],
        }
        filt = self._make_filter()
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        # No verdict set — item passed through the filter without being flagged.
        assert result[0].filter_verdict is None

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_leave_one_out_provides_regeneration_feedback(self, mock_judge):
        # leave_one_out mode: a redundant chunk should produce actionable feedback.
        mock_judge.return_value = {
            "answerable": True,
            "confidence": 0.9,
            "reasoning": "Can answer without c2",
            "missing_facts": [],
        }
        filt = self._make_filter(mode="leave_one_out")
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        verdict = result[0].filter_verdict
        assert "redundant" in verdict.reasoning.lower()
        assert verdict.metadata.get("refinement_hint")

    def test_disabled_filter_passes_through(self):
        filt = HopCountValidityFilter(cfg=HopCountValidityConfig(enabled=False))
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].filter_verdict is None


class TestLopsidedOverlapPreGate:
    """Tests for the word-overlap pre-gate (_check_lopsided_overlap)."""

    def _make_filter_leave_one_out(self) -> HopCountValidityFilter:
        return HopCountValidityFilter(
            cfg=HopCountValidityConfig(
                enabled=True,
                mode="leave_one_out",
                judge_model="test-model",
                judge_api_key="test-key",
                judge_base_url="http://test",
                batch_enabled=False,
                lopsided_high_threshold=0.60,
                lopsided_low_threshold=0.15,
            )
        )

    def test_lopsided_item_is_flagged(self):
        """A QA where one chunk covers 70% of answer words and another covers 10% is lopsided."""
        # Craft an answer whose words mostly overlap with c1 but barely with c2.
        answer = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        # c1 shares 7/10 answer words → overlap ~70%
        c1_content = "alpha beta gamma delta epsilon zeta eta other1 other2 other3"
        # c2 shares 1/10 answer word → overlap ~10%
        c2_content = "theta completely different unrelated words here for testing"
        item = _make_item(
            answer=answer,
            chunks=[
                {"id": "c1", "metadata": {}, "content": c1_content},
                {"id": "c2", "metadata": {}, "content": c2_content},
            ],
        )
        filt = self._make_filter_leave_one_out()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        verdict = result[0].filter_verdict
        assert verdict is not None
        assert verdict.status in ("needs_refinement", "rejected")
        assert "lopsided" in verdict.reasoning.lower()
        assert verdict.metadata.get("failure_type") == "lopsided_chunk_contribution"

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_balanced_overlap_passes_gate(self, mock_judge):
        """A QA with balanced overlap (40%/35%) should not be flagged as lopsided."""
        mock_judge.return_value = {
            "answerable": False,
            "confidence": 0.9,
            "reasoning": "Essential",
            "missing_facts": ["some fact"],
        }
        # answer words — 20 unique tokens
        answer = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        # c1 covers 4/10 = 40% of answer words
        c1_content = "alpha beta gamma delta other1 other2 other3 other4"
        # c2 covers ~3/10 = 30%+ of answer words (above 15% low threshold)
        c2_content = "epsilon zeta eta other5 other6 other7 other8"
        item = _make_item(
            answer=answer,
            chunks=[
                {"id": "c1", "metadata": {}, "content": c1_content},
                {"id": "c2", "metadata": {}, "content": c2_content},
            ],
        )
        filt = self._make_filter_leave_one_out()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        # Lopsided gate should not have fired; judge was called and returned passed.
        verdict = result[0].filter_verdict
        assert verdict is not None
        assert verdict.status == "passed"

    def test_single_chunk_item_skipped(self):
        """Single-chunk items skip the filter entirely (no verdict set)."""
        item = _make_item(
            chunks=[{"id": "c1", "metadata": {}, "content": "Only one chunk here."}]
        )
        filt = self._make_filter_leave_one_out()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].filter_verdict is None

    def test_lopsided_gate_only_in_leave_one_out_mode(self):
        """Lopsided overlap check must not fire in primary_only mode."""
        # Construct a QA that would be flagged as lopsided in leave_one_out mode.
        answer = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        c1_content = "alpha beta gamma delta epsilon zeta eta other1 other2"
        c2_content = "theta completely different words here"
        item = _make_item(
            answer=answer,
            chunks=[
                {"id": "c1", "metadata": {}, "content": c1_content},
                {"id": "c2", "metadata": {}, "content": c2_content},
            ],
        )
        filt_primary = HopCountValidityFilter(
            cfg=HopCountValidityConfig(
                enabled=True,
                mode="primary_only",
                judge_model="test-model",
                judge_api_key="test-key",
                judge_base_url="http://test",
                batch_enabled=False,
            )
        )
        ctx = _make_context()

        with patch.object(
            HopCountValidityFilter,
            "_judge_subset",
            return_value={
                "answerable": False,
                "confidence": 0.9,
                "reasoning": "Essential",
                "missing_facts": ["X"],
            },
        ):
            result = filt_primary.evaluate([item], ctx)

        # In primary_only mode the lopsided gate is skipped;
        # the item should not carry a lopsided verdict.
        verdict = result[0].filter_verdict
        if verdict is not None:
            assert verdict.metadata.get("failure_type") != "lopsided_chunk_contribution"


class TestDifficultyScoring:
    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_difficulty_score_in_passed_verdict(self, mock_judge):
        mock_judge.return_value = {
            "answerable": False,
            "confidence": 0.9,
            "reasoning": "Essential chunk",
            "missing_facts": ["fact X"],
        }
        filt = HopCountValidityFilter(
            cfg=HopCountValidityConfig(
                enabled=True,
                mode="leave_one_out",
                judge_model="test",
                judge_api_key="test",
                judge_base_url="http://test",
                batch_enabled=False,
            )
        )
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].is_passed
        meta = result[0].filter_verdict.metadata
        assert "difficulty_score" in meta
        assert 0.0 <= meta["difficulty_score"] <= 1.0

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_higher_hop_count_means_harder(self, mock_judge):
        mock_judge.return_value = {
            "answerable": False,
            "confidence": 0.9,
            "reasoning": "Essential",
            "missing_facts": ["X"],
        }

        def run_with_chunks(n_chunks):
            filt = HopCountValidityFilter(
                cfg=HopCountValidityConfig(
                    enabled=True,
                    mode="leave_one_out",
                    judge_model="test",
                    judge_api_key="test",
                    judge_base_url="http://test",
                    batch_enabled=False,
                )
            )
            chunks = [
                {"id": f"c{i}", "metadata": {}, "content": f"Chunk {i}."} for i in range(n_chunks)
            ]
            item = _make_item(chunks=chunks)
            ctx = _make_context()
            result = filt.evaluate([item], ctx)
            return result[0].filter_verdict.metadata["difficulty_score"]

        score_2hop = run_with_chunks(2)
        score_3hop = run_with_chunks(3)
        assert score_3hop > score_2hop


class TestRemovedReferenceChunksTracking:
    """Verify that demotion writes removed chunks to removed_reference_chunks."""

    def _make_filter_leave_one_out(self) -> HopCountValidityFilter:
        return HopCountValidityFilter(
            cfg=HopCountValidityConfig(
                enabled=True,
                mode="leave_one_out",
                judge_model="test-model",
                judge_api_key="test-key",
                judge_base_url="http://test",
                batch_enabled=False,
            )
        )

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_demotion_populates_removed_reference_chunks(self, mock_judge):
        """When a redundant chunk is stripped, it appears in removed_reference_chunks."""

        def side_effect(question, answer, subset, stats):
            # subset contains the chunks being tested (omitting one at a time)
            subset_ids = {c["id"] for c in subset}
            # c3 is redundant: answerable even when c3 is absent (subset = [c1, c2])
            if subset_ids == {"c1", "c2"}:
                return {"answerable": True, "confidence": 0.9, "reasoning": "c3 redundant", "missing_facts": []}
            return {"answerable": False, "confidence": 0.9, "reasoning": "Essential", "missing_facts": ["X"]}

        mock_judge.side_effect = side_effect
        filt = self._make_filter_leave_one_out()
        item = _make_item(
            chunks=[
                {"id": "c1", "metadata": {}, "content": "Chunk A."},
                {"id": "c2", "metadata": {}, "content": "Chunk B."},
                {"id": "c3", "metadata": {}, "content": "Chunk C — redundant."},
            ]
        )
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].is_passed
        # reference_chunks should only contain the 2 essential chunks
        ref_ids = {c["id"] for c in result[0].qa["reference_chunks"]}
        assert ref_ids == {"c1", "c2"}
        # removed_reference_chunks must track the stripped chunk
        removed = result[0].qa.get("removed_reference_chunks", [])
        assert len(removed) == 1
        assert removed[0]["chunk"]["id"] == "c3"
        assert removed[0]["reason"] == "redundant"
        assert removed[0]["filter"] == "hop_count_validity"

    @patch.object(HopCountValidityFilter, "_judge_subset")
    def test_no_removed_chunks_when_all_essential(self, mock_judge):
        """When no chunks are redundant, removed_reference_chunks is not populated."""
        mock_judge.return_value = {
            "answerable": False,
            "confidence": 0.9,
            "reasoning": "Essential",
            "missing_facts": ["X"],
        }
        filt = self._make_filter_leave_one_out()
        item = _make_item()
        ctx = _make_context()

        result = filt.evaluate([item], ctx)
        assert result[0].is_passed
        assert result[0].qa.get("removed_reference_chunks") is None
