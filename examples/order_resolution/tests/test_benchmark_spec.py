"""Predecessor seal and frozen v2 experiment identity."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
from order_resolution import benchmark_spec as spec

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sealed_root(tmp_path: Path) -> Path:
    """A clean copy of only the sealed v1 files."""

    for relative in spec.PREDECESSOR_SHA256:
        source = EXAMPLE_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


def test_verifies_the_live_worktree() -> None:
    result = spec.verify_predecessor(EXAMPLE_ROOT)
    assert result["files_verified"] == len(spec.PREDECESSOR_SHA256)
    assert result["predecessor"] == {
        "benchmark_id": "order-resolution-v1",
        "decision": "repair",
        "rollout_count": 972,
    }


def test_verifies_a_clean_copy(sealed_root: Path) -> None:
    assert spec.verify_predecessor(sealed_root)["files_verified"] == 6


@pytest.mark.parametrize("relative", sorted(spec.PREDECESSOR_SHA256))
def test_rejects_each_tampered_file(sealed_root: Path, relative: str) -> None:
    target = sealed_root / relative
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(spec.PredecessorSealError, match=relative):
        spec.verify_predecessor(sealed_root)


@pytest.mark.parametrize("relative", sorted(spec.PREDECESSOR_SHA256))
def test_rejects_each_missing_file(sealed_root: Path, relative: str) -> None:
    (sealed_root / relative).unlink()
    with pytest.raises(spec.PredecessorSealError, match="missing"):
        spec.verify_predecessor(sealed_root)


def test_rejects_a_truncated_file(sealed_root: Path) -> None:
    target = sealed_root / "data/eval.jsonl"
    target.write_bytes(target.read_bytes()[:-10])
    with pytest.raises(spec.PredecessorSealError, match="recorded bytes"):
        spec.verify_predecessor(sealed_root)


def _predecessor_manifest(tmp_path: Path, **overrides: object) -> Path:
    manifest = json.loads(
        (EXAMPLE_ROOT / spec.PREDECESSOR_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest.update(overrides)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_accepts_the_binding_repair_decision(tmp_path: Path) -> None:
    assert spec.verify_predecessor_decision(_predecessor_manifest(tmp_path)) == {
        "benchmark_id": "order-resolution-v1",
        "decision": "repair",
        "rollout_count": 972,
    }


@pytest.mark.parametrize("status", ["go", "harden", "no_rl_signal", "repair_again"])
def test_rejects_any_other_decision(tmp_path: Path, status: str) -> None:
    manifest = json.loads(
        (EXAMPLE_ROOT / spec.PREDECESSOR_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest["report"]["decision"]["status"] = status
    with pytest.raises(spec.PredecessorSealError, match="binding 'repair'"):
        spec.verify_predecessor_decision(_predecessor_manifest(tmp_path, report=manifest["report"]))


def test_rejects_an_incomplete_predecessor(tmp_path: Path) -> None:
    with pytest.raises(spec.PredecessorSealError, match="completed result"):
        spec.verify_predecessor_decision(_predecessor_manifest(tmp_path, status="failed"))


def test_rejects_a_changed_rollout_count(tmp_path: Path) -> None:
    with pytest.raises(spec.PredecessorSealError, match="971"):
        spec.verify_predecessor_decision(_predecessor_manifest(tmp_path, rollout_count=971))


def test_treats_schema_v1_as_the_implicit_predecessor_identity(tmp_path: Path) -> None:
    """v1 is never rewritten just to carry a benchmark id."""

    manifest = json.loads(
        (EXAMPLE_ROOT / spec.PREDECESSOR_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert "benchmark_id" not in manifest
    assert manifest["schema_version"] == 1
    with pytest.raises(spec.PredecessorSealError, match="not schema v1"):
        spec.verify_predecessor_decision(_predecessor_manifest(tmp_path, schema_version=2))


def test_rejects_a_foreign_benchmark_identity(tmp_path: Path) -> None:
    with pytest.raises(spec.PredecessorSealError, match="order-resolution-v2"):
        spec.verify_predecessor_decision(
            _predecessor_manifest(tmp_path, benchmark_id="order-resolution-v2")
        )


def test_rejects_an_unsupported_benchmark_id() -> None:
    with pytest.raises(spec.SpecError, match="order-resolution-v3"):
        spec.verify_predecessor(EXAMPLE_ROOT, benchmark_id="order-resolution-v3")


def test_v2_paths_never_collide_with_the_sealed_v1_set() -> None:
    sealed = set(spec.PREDECESSOR_SHA256)
    written = set(spec.ALLOWED_ARTIFACT_PATHS) | set(spec.ALLOWED_DATA_PATHS)
    assert not sealed & written
    assert all(path.startswith("data/v2/") for path in spec.ALLOWED_DATA_PATHS)
    assert all(path.startswith("artifacts/benchmark-v2/") for path in spec.ALLOWED_ARTIFACT_PATHS)
    assert len(written) == len(spec.ALLOWED_ARTIFACT_PATHS) + len(spec.ALLOWED_DATA_PATHS)


def test_canary_attempt_paths_are_frozen_and_distinct() -> None:
    assert spec.MAX_CANARY_ATTEMPTS == 2
    assert len(set(spec.CANARY_ATTEMPT_ROOTS)) == 2
    for root in spec.CANARY_ATTEMPT_ROOTS:
        for name in spec.WAVE_FILENAMES:
            assert f"{root}/{name}" in spec.ALLOWED_ARTIFACT_PATHS
    assert spec.CANARY_AUTHORIZATION_PATH not in {
        f"{root}/{name}" for root in spec.CANARY_ATTEMPT_ROOTS for name in spec.WAVE_FILENAMES
    }


def test_wave_geometry_matches_the_frozen_plan() -> None:
    assert spec.CANARY_TASKS_PER_ARM == 18
    assert spec.CANARY_ROLLOUTS == 72
    assert spec.FULL_ROLLOUTS == 360
    assert spec.STRESS_ROLLOUTS == 324
    assert spec.SIGNAL_PROBE_ROLLOUTS == 288
    assert spec.TOTAL_FULL_WAVE_ROLLOUTS == 972 == spec.PREDECESSOR_ROLLOUT_COUNT
    assert spec.TRAIN_ROW_COUNT == 180
    assert spec.EVAL_ROW_COUNT == 90


def test_reserved_training_indices_are_disjoint() -> None:
    reserved = (set(spec.SIGNAL_PROBE_INDICES), set(spec.CANARY_INDICES), {spec.ORACLE_DEMO_INDEX})
    assert set(spec.CANARY_INDICES).isdisjoint(spec.SIGNAL_PROBE_INDICES)
    assert spec.ORACLE_DEMO_INDEX not in set(spec.CANARY_INDICES)
    assert all(index < spec.TRAIN_ROWS_PER_CELL for group in reserved for index in group)


def test_full_gates_match_the_unchanged_v1_thresholds() -> None:
    assert dict(spec.FULL_GATES) == {
        "max_infrastructure_failure_rate": 0.02,
        "min_frontier_success_rate": 0.70,
        "min_base_success_rate": 0.15,
        "max_base_success_rate": 0.80,
        "min_frontier_base_gap": 0.10,
        "min_model_attributable_base_failures": 10,
        "min_mixed_signal_groups": 9,
        "min_sibling_success_rate": 0.10,
        "max_sibling_success_rate": 0.90,
    }


def test_arms_cover_both_frontiers_and_both_small_prompts() -> None:
    assert [arm.id for arm in spec.ARMS] == [
        "small_base",
        "small_two_shot",
        "frontier_gpt",
        "frontier_grok",
    ]
    assert {arm.model for arm in spec.ARMS} == set(spec.REQUIRED_MODELS)
    two_shot = next(arm for arm in spec.ARMS if arm.id == spec.TWO_SHOT_ARM_ID)
    assert two_shot.model == spec.SMALL_MODEL
    assert two_shot.prompt == spec.TWO_SHOT_PROMPT


def test_template_banks_are_split_partitioned_and_stratum_matched() -> None:
    for split in ("train", "eval"):
        assert set(spec.PROMPT_TEMPLATES[split]) == set(spec.REQUEST_SHAPES)
        for shape, bank in spec.PROMPT_TEMPLATES[split].items():
            assert len(bank) == len(spec.PROMPT_STRATA), (split, shape)
            assert len(set(bank)) == len(bank), (split, shape)
    for shape in spec.REQUEST_SHAPES:
        train = set(spec.PROMPT_TEMPLATES["train"][shape])
        held_out = set(spec.PROMPT_TEMPLATES["eval"][shape])
        assert train.isdisjoint(held_out), shape


def test_template_banks_use_the_same_placeholders_per_shape() -> None:
    """Both splits must state the same facts; only the wording differs."""

    for shape in spec.REQUEST_SHAPES:
        placeholders = {
            split: [
                set(re.findall(r"\{(\w+)\}", template))
                for template in spec.PROMPT_TEMPLATES[split][shape]
            ]
            for split in ("train", "eval")
        }
        for position in range(len(spec.PROMPT_STRATA)):
            assert placeholders["train"][position] == placeholders["eval"][position], (
                shape,
                position,
            )


def test_clarification_shapes_omit_exactly_one_fact() -> None:
    def fields(shape: str) -> set[str]:
        return set(re.findall(r"\{(\w+)\}", spec.PROMPT_TEMPLATES["train"][shape][0]))

    assert fields("change_address.full") - fields("change_address.no_postal") == {"postal_code"}
    assert fields("replace_variant.sized") - fields("replace_variant.no_size") == {"size"}
    assert fields("cancel_item.identified") - fields("cancel_item.ambiguous") == {"product"}


def test_request_shape_maps_every_cell() -> None:
    shapes = {
        spec.request_shape(*cell.rsplit("-", 1))
        for cell in spec.CELLS
    }
    assert shapes == set(spec.REQUEST_SHAPES)
    assert spec.request_shape("cancel_item", "clarify") == "cancel_item.ambiguous"
    assert spec.request_shape("cancel_item", "deny") == "cancel_item.identified"
    with pytest.raises(spec.SpecError, match="unknown action family"):
        spec.request_shape("refund_order", "execute")


def test_prompt_template_cycles_strata_evenly_across_a_cell() -> None:
    for split, per_cell in (("train", spec.TRAIN_ROWS_PER_CELL), ("eval", spec.EVAL_ROWS_PER_CELL)):
        strata = [
            spec.prompt_template(split, "cancel_item", "execute", index)[2]
            for index in range(per_cell)
        ]
        assert set(strata) == set(spec.PROMPT_STRATA)
        assert len(set(Counter(strata).values())) == 1


def test_catalog_namespace_constants_support_disjoint_pools() -> None:
    train = range(*spec.TRAIN_PRODUCT_RANGE)
    held_out = range(*spec.EVAL_PRODUCT_RANGE)
    assert set(train).isdisjoint(held_out)
    assert len(train) + len(held_out) == spec.CATALOG_PRODUCTS
    assert len(spec.CATALOG_ADJECTIVES) * len(spec.CATALOG_NOUNS) == spec.CATALOG_PRODUCTS
    # The distractor offset must shift both name tokens and stay inside a pool.
    assert spec.DISTRACTOR_OFFSET % len(spec.CATALOG_NOUNS) != 0
    assert spec.DISTRACTOR_OFFSET % len(train) != 0
    assert spec.DISTRACTOR_OFFSET % len(held_out) != 0


def test_oracle_demo_task_ids_are_the_two_declared_cells() -> None:
    assert spec.oracle_demo_task_ids() == (
        "train-cancel_item-clarify-00",
        "train-replace_variant-deny-00",
    )
    assert all(cell in spec.CELLS for cell in spec.ORACLE_DEMO_CELLS)


def test_spec_identity_is_json_serializable_and_seals_the_predecessor() -> None:
    identity = json.loads(json.dumps(spec.spec_identity(), sort_keys=True))
    assert identity["schema_version"] == 2
    assert identity["benchmark_id"] == "order-resolution-v2"
    assert identity["predecessor"]["decision"] == "repair"
    assert identity["predecessor"]["sha256"] == dict(spec.PREDECESSOR_SHA256)
    assert identity["usage_accounting"]["mode"] == "omitted"
