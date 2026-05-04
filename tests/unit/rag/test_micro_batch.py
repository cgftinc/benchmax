"""Tests for micro-batch pipeline: checkpoint, serialization, config, re-queue."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from benchmax.rag.qa_generation.pipeline_config import (
    PipelineConfig,
    MicroBatchConfig,
    PlatformConfig,
    TargetsConfig,
)
from benchmax.rag.qa_generation.checkpoint import (
    CheckpointManager,
    Manifest,
    compute_config_hash,
    deserialize_generated_qa,
    serialize_generated_qa,
)
from benchmax.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    question: str = "What is RAG?",
    answer: str = "Retrieval-Augmented Generation is a technique.",
    qa_type: str = "lookup",
    verdict_status: str | None = "passed",
    meta: dict | None = None,
    chunks: list[dict] | None = None,
    journey_events: list[dict] | None = None,
) -> GeneratedQA:
    qa = {
        "question": question,
        "answer": answer,
        "qa_type": qa_type,
        "reference_chunks": chunks or [{"id": "c1", "content": "chunk text", "metadata": {}}],
    }
    verdict = None
    if verdict_status is not None:
        verdict = FilterVerdict(
            status=verdict_status,
            reason="test_reason",
            reasoning="some reasoning",
            metadata={"score": 0.9},
        )
    return GeneratedQA(
        qa=qa,
        generation_metadata=meta or {"task_id": "task_00000"},
        filter_verdict=verdict,
        regeneration_history=[{"type": "initial", "round": 0}],
        journey_events=list(journey_events or []),
    )


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_with_verdict(self) -> None:
        item = _make_item(verdict_status="passed")
        serialized = serialize_generated_qa(item)
        restored = deserialize_generated_qa(serialized)

        assert restored.qa["question"] == item.qa["question"]
        assert restored.qa["answer"] == item.qa["answer"]
        assert restored.generation_metadata == item.generation_metadata
        assert restored.regeneration_history == item.regeneration_history
        assert restored.journey_events == item.journey_events
        assert restored.filter_verdict is not None
        assert restored.filter_verdict.status == "passed"
        assert restored.filter_verdict.reason == "test_reason"
        assert restored.filter_verdict.metadata == {"score": 0.9}

    def test_roundtrip_without_verdict(self) -> None:
        item = _make_item(verdict_status=None)
        serialized = serialize_generated_qa(item)
        restored = deserialize_generated_qa(serialized)

        assert restored.filter_verdict is None
        assert restored.qa["question"] == item.qa["question"]

    def test_serialized_is_json_safe(self) -> None:
        item = _make_item()
        serialized = serialize_generated_qa(item)
        json_str = json.dumps(serialized, ensure_ascii=False)
        restored = deserialize_generated_qa(json.loads(json_str))
        assert restored.qa["question"] == item.qa["question"]

    def test_dunder_keys_stripped_from_qa(self) -> None:
        item = _make_item()
        serialized = serialize_generated_qa(item)
        assert "__generation_metadata" in serialized
        restored = deserialize_generated_qa(serialized)
        assert "__generation_metadata" not in restored.qa
        assert "__filter_verdict" not in restored.qa
        assert "__regeneration_history" not in restored.qa
        assert "__journey_events" not in restored.qa

    def test_non_serializable_metadata_stripped(self) -> None:
        """AnchorBundle or other complex objects in metadata should not crash."""

        class FakeAnchor:
            pass

        item = _make_item(
            meta={
                "task_id": "task_00001",
                "anchor_bundle": FakeAnchor(),
                "nested": {"ok": 1, "bad": FakeAnchor()},
            }
        )
        serialized = serialize_generated_qa(item)
        # Should not raise
        json_str = json.dumps(serialized, ensure_ascii=False)
        restored = deserialize_generated_qa(json.loads(json_str))
        assert restored.generation_metadata["task_id"] == "task_00001"
        # Non-serializable values become None
        assert restored.generation_metadata["anchor_bundle"] is None
        assert restored.generation_metadata["nested"]["ok"] == 1
        assert restored.generation_metadata["nested"]["bad"] is None

    def test_journey_events_roundtrip(self) -> None:
        item = _make_item(
            journey_events=[
                {
                    "stage": "generator",
                    "event_type": "generated",
                    "task_id": "task_00000",
                    "refinement_count": 0,
                    "qa_type_before": "lookup",
                    "qa_type_after": "lookup",
                    "reason_code": "",
                    "details": {"generation_mode": "llm_direct"},
                }
            ]
        )
        serialized = serialize_generated_qa(item)
        restored = deserialize_generated_qa(serialized)
        assert restored.journey_events == item.journey_events


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_deterministic(self) -> None:
        h1 = compute_config_hash(total_samples=100, corpus_id="abc")
        h2 = compute_config_hash(total_samples=100, corpus_id="abc")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        h1 = compute_config_hash(total_samples=100, corpus_id="abc")
        h2 = compute_config_hash(total_samples=200, corpus_id="abc")
        assert h1 != h2

    def test_distribution_changes_hash(self) -> None:
        h1 = compute_config_hash(
            total_samples=100,
            corpus_id="abc",
            primary_type_distribution={"lookup": 0.5, "multi_hop": 0.5},
        )
        h2 = compute_config_hash(
            total_samples=100,
            corpus_id="abc",
            primary_type_distribution={"lookup": 0.3, "multi_hop": 0.7},
        )
        assert h1 != h2

    def test_reasoning_mode_changes_hash(self) -> None:
        h1 = compute_config_hash(
            total_samples=100,
            corpus_id="abc",
            reasoning_mode_distribution={"factual": 0.5, "inference": 0.5},
        )
        h2 = compute_config_hash(
            total_samples=100,
            corpus_id="abc",
            reasoning_mode_distribution={"factual": 0.3, "inference": 0.7},
        )
        assert h1 != h2

    def test_hop_distribution_changes_hash(self) -> None:
        h1 = compute_config_hash(
            total_samples=100,
            corpus_id="abc",
            hop_distribution={2: 0.5, 3: 0.5},
        )
        h2 = compute_config_hash(
            total_samples=100,
            corpus_id="abc",
            hop_distribution={2: 0.8, 3: 0.2},
        )
        assert h1 != h2


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_roundtrip(self) -> None:
        m = Manifest(
            config_hash="abc123",
            completed_batch_count=5,
            total_passed=400,
            total_rejected=100,
            iteration_count=3,
            accepted_by_type={"lookup": 120, "multi_hop": 280},
            accepted_by_reasoning_mode={"factual": 100, "inference": 180},
            accepted_by_hop_count={"2": 140, "3": 140},
        )
        d = m.to_dict()
        m2 = Manifest.from_dict(d)
        assert m2.config_hash == m.config_hash
        assert m2.completed_batch_count == m.completed_batch_count
        assert m2.total_passed == m.total_passed
        assert m2.total_rejected == m.total_rejected
        assert m2.iteration_count == m.iteration_count
        assert m2.accepted_by_type == m.accepted_by_type
        assert m2.accepted_by_reasoning_mode == m.accepted_by_reasoning_mode
        assert m2.accepted_by_hop_count == m.accepted_by_hop_count


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_and_load_manifest(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpt", config_hash="hash1")
        manifest = Manifest(config_hash="hash1", completed_batch_count=2)
        mgr.save_manifest(manifest)

        loaded = mgr.load_manifest()
        assert loaded is not None
        assert loaded.config_hash == "hash1"
        assert loaded.completed_batch_count == 2

    def test_resume_empty(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpt", config_hash="hash1")
        state = mgr.resume_state()
        assert state.passed_items == []
        assert state.completed_batch_count == 0

    def test_resume_with_hash_mismatch(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpt", config_hash="old_hash")
        mgr.save_manifest(Manifest(config_hash="old_hash", completed_batch_count=5))

        # New manager with different hash.
        mgr2 = CheckpointManager(tmp_path / "ckpt", config_hash="new_hash")
        state = mgr2.resume_state()
        assert state.passed_items == []
        assert state.completed_batch_count == 0
        # Checkpoint dir should be cleaned up.
        assert not (tmp_path / "ckpt" / "manifest.json").exists()

    def test_save_batch_and_resume(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpt", config_hash="h1")

        passed = [
            _make_item(
                question=f"Q{i}",
                journey_events=[
                    {
                        "stage": "generator",
                        "event_type": "generated",
                        "task_id": f"task_{i}",
                        "refinement_count": 0,
                        "qa_type_before": "lookup",
                        "qa_type_after": "lookup",
                        "reason_code": "",
                        "details": {},
                    }
                ],
            )
            for i in range(3)
        ]
        rejected = [_make_item(question="bad", verdict_status="rejected")]

        mgr.save_batch(0, passed, rejected, regens_count=1)

        # Verify files exist.
        assert mgr.passed_path.exists()
        assert (tmp_path / "ckpt" / "batch_0000_rejected.jsonl").exists()

        # Resume should load the passed items.
        state = mgr.resume_state()
        assert len(state.passed_items) == 3
        assert state.completed_batch_count == 1
        assert state.passed_items[0].qa["question"] == "Q0"
        assert state.passed_items[0].journey_events[0]["event_type"] == "generated"

    def test_multiple_batches_append(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpt", config_hash="h1")

        batch1_passed = [_make_item(question="Q1")]
        batch2_passed = [_make_item(question="Q2"), _make_item(question="Q3")]

        mgr.save_batch(0, batch1_passed, [])
        mgr.save_batch(1, batch2_passed, [])

        state = mgr.resume_state()
        assert len(state.passed_items) == 3
        questions = [item.qa["question"] for item in state.passed_items]
        assert questions == ["Q1", "Q2", "Q3"]

    def test_cleanup(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path / "ckpt", config_hash="h1")
        mgr.save_manifest(Manifest(config_hash="h1"))
        assert (tmp_path / "ckpt").exists()
        mgr.cleanup()
        assert not (tmp_path / "ckpt").exists()


# ---------------------------------------------------------------------------
# MicroBatchConfig
# ---------------------------------------------------------------------------


class TestMicroBatchConfig:
    def test_defaults(self) -> None:
        cfg = MicroBatchConfig()
        assert cfg.batch_size == 0  # 0 = auto-computed
        assert cfg.resume is True
        assert cfg.max_iterations == 50
        assert cfg.max_parallel_batches == 0  # 0 = auto-computed

    def test_on_pipeline_config(self) -> None:
        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            micro_batch=MicroBatchConfig(batch_size=50),
        )
        assert cfg.micro_batch.batch_size == 50


# ---------------------------------------------------------------------------
# Replacement task creation
# ---------------------------------------------------------------------------


class TestComputeNextBatch:
    def test_creates_correct_count(self) -> None:
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        chunk_mock = MagicMock()
        chunk_mock.hash = "seed_123"
        source.sample_chunks.return_value = [chunk_mock] * 5

        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(total_samples=10),
        )

        tasks = compute_next_batch(
            target_type_counts={"lookup": 3, "multi_hop": 7},
            accepted_type_counts={"lookup": 0, "multi_hop": 0},
            target_mode_counts={"factual": 3, "inference": 4},
            accepted_mode_counts={},
            target_hop_counts={"2": 4, "3": 3},
            accepted_hop_counts={},
            batch_size=5,
            source=source,
            cfg=cfg,
            iteration_count=0,
        )
        assert len(tasks) == 5
        assert all(t.task_id.startswith("iter_0_") for t in tasks)

    def test_returns_empty_when_targets_met(self) -> None:
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(total_samples=10),
        )

        tasks = compute_next_batch(
            target_type_counts={"lookup": 3, "multi_hop": 7},
            accepted_type_counts={"lookup": 3, "multi_hop": 7},
            target_mode_counts={},
            accepted_mode_counts={},
            target_hop_counts={},
            accepted_hop_counts={},
            batch_size=5,
            source=source,
            cfg=cfg,
            iteration_count=0,
        )
        assert tasks == []

    def test_task_types_follow_remaining_deficits(self) -> None:
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        chunk_mock = MagicMock()
        chunk_mock.hash = "seed_789"
        source.sample_chunks.return_value = [chunk_mock] * 10

        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(
                total_samples=100,
                primary_type_distribution={"lookup": 0.5, "multi_hop": 0.5},
            ),
        )

        tasks = compute_next_batch(
            target_type_counts={"lookup": 50, "multi_hop": 50},
            accepted_type_counts={"lookup": 49, "multi_hop": 46},
            target_mode_counts={"factual": 25, "inference": 25},
            accepted_mode_counts={"factual": 23, "inference": 23},
            target_hop_counts={"2": 25, "3": 25},
            accepted_hop_counts={"2": 23, "3": 23},
            batch_size=10,
            source=source,
            cfg=cfg,
            iteration_count=0,
        )
        lookups = sum(1 for t in tasks if t.qa_type == "lookup")
        multi_hops = sum(1 for t in tasks if t.qa_type == "multi_hop")
        assert lookups == 1
        assert multi_hops == 4

    def test_fallback_when_source_empty(self) -> None:
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        source.sample_chunks.return_value = []

        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(total_samples=10),
        )

        tasks = compute_next_batch(
            target_type_counts={"lookup": 3, "multi_hop": 7},
            accepted_type_counts={"lookup": 0, "multi_hop": 0},
            target_mode_counts={"factual": 3, "inference": 4},
            accepted_mode_counts={},
            target_hop_counts={"2": 4, "3": 3},
            accepted_hop_counts={},
            batch_size=5,
            source=source,
            cfg=cfg,
            iteration_count=0,
        )
        assert len(tasks) == 5

    def test_parallel_waves_respect_proportional_allocation(self) -> None:
        """Simulate parallel batch submission: 5 batches computed against
        accepted_type_counts=0 should collectively match the target distribution,
        not starve one type by overallocating the other."""
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        chunk_mock = MagicMock()
        chunk_mock.hash = "seed_parallel"
        source.sample_chunks.return_value = [chunk_mock] * 50

        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(
                total_samples=250,
                primary_type_distribution={"lookup": 0.30, "multi_hop": 0.70},
            ),
        )

        # 5 concurrent batches, all computed before any have returned.
        combined: list = []
        for i in range(5):
            tasks = compute_next_batch(
                target_type_counts={"lookup": 75, "multi_hop": 175},
                accepted_type_counts={"lookup": 0, "multi_hop": 0},
                target_mode_counts={"factual": 88, "inference": 87},
                accepted_mode_counts={},
                target_hop_counts={"2": 88, "3": 61, "4": 26},
                accepted_hop_counts={},
                batch_size=50,
                source=source,
                cfg=cfg,
                iteration_count=i,
            )
            combined.extend(tasks)

        lookups = sum(1 for t in combined if t.qa_type == "lookup")
        multi_hops = sum(1 for t in combined if t.qa_type == "multi_hop")

        # Per-batch should be ~15 lookup / ~35 multi_hop (30/70 of 50).
        # Across 5 parallel waves: exactly 75 lookup and 175 multi_hop — the target.
        assert lookups == 75, f"expected 75 lookup tasks across 5 parallel batches, got {lookups}"
        assert multi_hops == 175, (
            f"expected 175 multi_hop tasks across 5 parallel batches, got {multi_hops}"
        )

    def test_overflow_redistributes_when_one_type_capped(self) -> None:
        """When one type is near its quota, the overflow should go to the other
        type (which still has room) instead of being dropped."""
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        chunk_mock = MagicMock()
        chunk_mock.hash = "seed_overflow"
        source.sample_chunks.return_value = [chunk_mock] * 20

        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(
                total_samples=100,
                primary_type_distribution={"lookup": 0.30, "multi_hop": 0.70},
            ),
        )

        # Remaining: lookup=1, multi_hop=19. Batch=10. Proportional would give
        # 3 lookup / 7 multi_hop, but lookup cap is 1 → overflow 2 → all to multi_hop.
        tasks = compute_next_batch(
            target_type_counts={"lookup": 30, "multi_hop": 70},
            accepted_type_counts={"lookup": 29, "multi_hop": 51},
            target_mode_counts={"factual": 35, "inference": 35},
            accepted_mode_counts={"factual": 25, "inference": 26},
            target_hop_counts={"2": 35, "3": 35},
            accepted_hop_counts={"2": 25, "3": 26},
            batch_size=10,
            source=source,
            cfg=cfg,
            iteration_count=0,
        )
        lookups = sum(1 for t in tasks if t.qa_type == "lookup")
        multi_hops = sum(1 for t in tasks if t.qa_type == "multi_hop")
        assert lookups == 1
        assert multi_hops == 9
        assert len(tasks) == 10

    def test_sequential_run_respects_distribution_over_iterations(self) -> None:
        """Sequential run (parallel=1) should also converge on the target
        distribution: simulate iterative batches, updating accepted counts
        after each, and check totals match target proportions."""
        from benchmax.rag.qa_generation.pipeline import compute_next_batch

        source = MagicMock()
        chunk_mock = MagicMock()
        chunk_mock.hash = "seed_seq"
        source.sample_chunks.return_value = [chunk_mock] * 50

        cfg = PipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(
                total_samples=100,
                primary_type_distribution={"lookup": 0.30, "multi_hop": 0.70},
            ),
        )

        accepted = {"lookup": 0, "multi_hop": 0}
        total_lookups = 0
        total_multi_hops = 0
        for i in range(10):
            tasks = compute_next_batch(
                target_type_counts={"lookup": 30, "multi_hop": 70},
                accepted_type_counts=dict(accepted),
                target_mode_counts={"factual": 35, "inference": 35},
                accepted_mode_counts={},
                target_hop_counts={"2": 35, "3": 35},
                accepted_hop_counts={},
                batch_size=10,
                source=source,
                cfg=cfg,
                iteration_count=i,
            )
            for t in tasks:
                if t.qa_type == "lookup":
                    total_lookups += 1
                    accepted["lookup"] += 1
                else:
                    total_multi_hops += 1
                    accepted["multi_hop"] += 1

        assert total_lookups == 30
        assert total_multi_hops == 70


class TestQuotaAcceptance:
    def test_acceptance_uses_effective_type_and_respects_quota(self) -> None:
        from benchmax.rag.qa_generation.pipeline import _accept_items_under_type_quota

        accepted_counts = {"lookup": 0, "multi_hop": 0}
        target_counts = {"lookup": 1, "multi_hop": 1}
        items = [
            _make_item(
                qa_type="lookup",
                meta={"task_id": "lookup_1", "qa_type_target": "lookup"},
                chunks=[{"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"}],
            ),
            _make_item(
                qa_type="multi_hop",
                meta={"task_id": "mh_demoted", "qa_type_target": "multi_hop"},
                chunks=[{"id": "c2", "metadata": {"file": "a.mdx"}, "content": "A"}],
            ),
            _make_item(
                qa_type="lookup",
                meta={"task_id": "lookup_promoted", "qa_type_target": "lookup"},
                chunks=[
                    {"id": "c3", "metadata": {"file": "a.mdx"}, "content": "A"},
                    {"id": "c4", "metadata": {"file": "b.mdx"}, "content": "B"},
                ],
            ),
        ]

        accepted, rejected = _accept_items_under_type_quota(
            items,
            accepted_type_counts=accepted_counts,
            target_type_counts=target_counts,
        )

        assert len(accepted) == 2
        assert len(rejected) == 1
        assert accepted_counts == {"lookup": 1, "multi_hop": 1}
        assert accepted[0].qa["qa_type"] == "lookup"
        assert accepted[1].qa["qa_type"] == "multi_hop"
        assert rejected[0].qa["qa_type"] == "lookup"
        assert rejected[0].filter_verdict is not None
        assert rejected[0].filter_verdict.reason == "type_quota_exceeded"

    def test_original_lookup_is_rejected_when_lookup_quota_is_full(self) -> None:
        from benchmax.rag.qa_generation.pipeline import _accept_items_under_type_quota

        item = _make_item(
            qa_type="lookup",
            meta={"task_id": "lookup_2", "qa_type_target": "lookup"},
        )
        accepted, rejected = _accept_items_under_type_quota(
            [item],
            accepted_type_counts={"lookup": 1, "multi_hop": 0},
            target_type_counts={"lookup": 1, "multi_hop": 1},
        )

        assert accepted == []
        assert len(rejected) == 1
        assert rejected[0].filter_verdict is not None
        assert rejected[0].filter_verdict.reason == "type_quota_exceeded"


class TestJourneyStats:
    def test_collects_summary_from_item_events(self) -> None:
        from benchmax.rag.qa_generation.pipeline import _collect_journey_stats

        passed = _make_item(
            qa_type="multi_hop",
            meta={"task_id": "pass_1", "qa_type_target": "lookup"},
            journey_events=[
                {
                    "stage": "generator",
                    "event_type": "generated",
                    "task_id": "pass_1",
                    "refinement_count": 0,
                    "qa_type_before": "lookup",
                    "qa_type_after": "lookup",
                    "reason_code": "",
                    "details": {},
                },
                {
                    "stage": "type_balancer",
                    "event_type": "relabelled",
                    "task_id": "pass_1",
                    "refinement_count": 0,
                    "qa_type_before": "lookup",
                    "qa_type_after": "multi_hop",
                    "reason_code": "lookup_to_multi_hop",
                    "details": {},
                },
                {
                    "stage": "type_balancer",
                    "event_type": "accepted",
                    "task_id": "pass_1",
                    "refinement_count": 0,
                    "qa_type_before": "lookup",
                    "qa_type_after": "multi_hop",
                    "reason_code": "",
                    "details": {},
                },
            ],
        )
        rejected = _make_item(
            qa_type="lookup",
            verdict_status="rejected",
            meta={"task_id": "rej_1", "qa_type_target": "multi_hop"},
            journey_events=[
                {
                    "stage": "type_balancer",
                    "event_type": "quota_rejected",
                    "task_id": "rej_1",
                    "refinement_count": 0,
                    "qa_type_before": "multi_hop",
                    "qa_type_after": "lookup",
                    "reason_code": "type_quota_exceeded",
                    "details": {"effective_qa_type": "lookup"},
                }
            ],
        )

        stats = _collect_journey_stats(passed_items=[passed], rejected_items=[rejected])

        assert stats["event_counts"]["accepted"] == 1
        assert stats["event_counts"]["quota_rejected"] == 1
        assert stats["relabeled"]["lookup_to_multi_hop"] == 1
        assert stats["quota_rejected_by_effective_type"]["lookup"] == 1
        assert stats["accepted_by_requested_type"]["lookup"] == 1
        assert stats["accepted_by_effective_type"]["multi_hop"] == 1
