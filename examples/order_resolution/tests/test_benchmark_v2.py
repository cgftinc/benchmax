"""Versioned, append-only v2 execution: paths, state machine, and verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from order_resolution import benchmark_spec as spec
from order_resolution.benchmark import (
    BenchmarkStateError,
    assert_bundle_matches_abi,
    assert_no_secrets,
    build_two_shot_example,
    check_v2_report,
    environment_abi,
    environment_abi_sha256,
    evaluate_canary,
    load_v2_oracle_demos,
    read_spec,
    v2_decision,
    v2_rollout_id,
    v2_task_selection,
    verify_v2_benchmark,
)
from order_resolution.hosting import CANONICAL_RUNTIME_DEPENDENCIES
from order_resolution.order_env import world_id_for_rollout

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXAMPLE_ROOT / "data" / "v2"
SPEC_PATH = EXAMPLE_ROOT / spec.SPEC_PATH
MIRRORED = ("data", "order_resolution", "templates", "alembic.ini", "migrations")


def _requires_spec() -> None:
    if not SPEC_PATH.exists():
        pytest.skip("freeze the v2 spec first: main.py freeze-benchmark")


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    """An example root that shares the real sealed inputs but its own artifacts."""

    _requires_spec()
    root = tmp_path / "example"
    root.mkdir()
    for name in MIRRORED:
        os.symlink(EXAMPLE_ROOT / name, root / name)
    (root / "artifacts").mkdir()
    for name in ("baseline.json", "baseline.raw.jsonl", "baseline.html"):
        os.symlink(EXAMPLE_ROOT / "artifacts" / name, root / "artifacts" / name)
    (root / spec.ARTIFACT_ROOT).mkdir(parents=True)
    os.symlink(SPEC_PATH, root / spec.SPEC_PATH)
    return root


# --- pure helpers -----------------------------------------------------------


def test_rollout_ids_are_unique_across_waves_attempts_and_nonces() -> None:
    base = dict(
        spec_sha256="a" * 64,
        run_nonce="nonce",
        arm="small_base",
        repetition=0,
        task_id="train-cancel_item-execute-18",
        sample=0,
    )
    ids = {
        v2_rollout_id(wave="canary", attempt=1, **base),
        v2_rollout_id(wave="canary", attempt=2, **base),
        v2_rollout_id(wave="full", attempt=None, **base),
        v2_rollout_id(wave="canary", attempt=1, **{**base, "run_nonce": "other"}),
        v2_rollout_id(wave="canary", attempt=1, **{**base, "spec_sha256": "b" * 64}),
        v2_rollout_id(wave="canary", attempt=1, **{**base, "arm": "frontier_gpt"}),
        v2_rollout_id(wave="canary", attempt=1, **{**base, "sample": 1}),
        v2_rollout_id(wave="canary", attempt=1, **{**base, "repetition": 1}),
    }
    assert len(ids) == 8
    assert all(value.startswith("order-resolution-v2-") for value in ids)
    # A v1 rollout id could never collide: it carries no benchmark prefix.
    assert not any(value.startswith("full-") for value in ids)


def test_environment_abi_excludes_secrets_and_constructor_values() -> None:
    abi = environment_abi(EXAMPLE_ROOT)
    serialized = json.dumps(abi)
    for marker in ("postgres", "runtime_database_url", "password", "@", "neon"):
        assert marker not in serialized.lower()
    assert abi["limits"] == {"max_turns": 8, "max_tool_calls": 16}
    assert abi["pip_dependencies"] == list(CANONICAL_RUNTIME_DEPENDENCIES)
    assert environment_abi_sha256(EXAMPLE_ROOT) == environment_abi_sha256(EXAMPLE_ROOT)


def test_branch_specific_bundles_share_one_abi() -> None:
    """Two child branches produce different bundle bytes and the same contract."""

    frozen = environment_abi_sha256(EXAMPLE_ROOT)
    for digest in ("bundle-branch-a", "bundle-branch-b"):
        assert_bundle_matches_abi(
            {
                "digest": digest,
                "class": "OrderResolutionEnv",
                "pip_dependencies": list(CANONICAL_RUNTIME_DEPENDENCIES),
                "secret_boundary": "ok",
            },
            example_root=EXAMPLE_ROOT,
            expected_abi_sha256=frozen,
        )
    with pytest.raises(BenchmarkStateError, match="ABI drifted"):
        assert_bundle_matches_abi(
            {"class": "OrderResolutionEnv", "secret_boundary": "ok"},
            example_root=EXAMPLE_ROOT,
            expected_abi_sha256="0" * 64,
        )


def test_bundle_must_pass_its_secret_boundary() -> None:
    with pytest.raises(BenchmarkStateError, match="secret-boundary"):
        assert_bundle_matches_abi(
            {
                "class": "OrderResolutionEnv",
                "pip_dependencies": list(CANONICAL_RUNTIME_DEPENDENCIES),
                "secret_boundary": "failed",
            },
            example_root=EXAMPLE_ROOT,
            expected_abi_sha256=environment_abi_sha256(EXAMPLE_ROOT),
        )


@pytest.mark.parametrize(
    "text",
    [
        '{"url": "postgresql://role:secret@host/db"}',
        '{"key": "napi_abcdefghijklmnopqrstuvwx"}',
        '{"NEON_API_KEY": "x"}',
    ],
)
def test_secret_scan_rejects_credentials(text: str) -> None:
    with pytest.raises(BenchmarkStateError, match="secret-like"):
        assert_no_secrets(text, label="artifact")


def test_secret_scan_accepts_redacted_content() -> None:
    assert_no_secrets('{"branch_id": "br-little-moon", "deleted": true}', label="artifact")


# --- gate evaluation --------------------------------------------------------


def _canary_records(
    *,
    base: int = 8,
    gpt: int = 14,
    grok: int = 10,
    two_shot_classes: tuple[str, ...] = ("execute", "clarify", "deny"),
    infrastructure: int = 0,
    invariants: int = 0,
    count: int = spec.CANARY_ROLLOUTS,
) -> list[dict[str, Any]]:
    """Synthesize a canary wave with controllable per-arm outcomes."""

    quotas = {"small_base": base, "frontier_gpt": gpt, "frontier_grok": grok}
    # Cell-major ordering, so a quota covers every cell once before doubling up.
    tasks = [(cell, index) for index in spec.CANARY_INDICES for cell in spec.CELLS]
    records: list[dict[str, Any]] = []
    for arm in spec.ARMS:
        remaining = quotas.get(arm.id, 0)
        classes_left = set(two_shot_classes)
        for cell, index in tasks:
            outcome_class = cell.rsplit("-", 1)[1]
            if arm.id == spec.TWO_SHOT_ARM_ID:
                success = outcome_class in classes_left
                classes_left.discard(outcome_class)
            else:
                success = remaining > 0
                if success:
                    remaining -= 1
            records.append(
                {
                    "phase": "canary",
                    "arm": arm.id,
                    "model": arm.model,
                    "repetition": 0,
                    "rollout_id": f"{arm.id}-{cell}-{index}",
                    "world_id": f"world-{arm.id}-{cell}-{index}",
                    "task_id": f"train-{cell}-{index:02d}",
                    "cell": cell,
                    "action_family": cell.rsplit("-", 1)[0],
                    "outcome_class": outcome_class,
                    "task_success": 1.0 if success else 0.0,
                    "termination_reason": "finished",
                    "error_present": False,
                    "rewards": {"_forbidden_mutation": 0.0, "_invariant_failure": 0.0},
                    "sample": 0,
                }
            )
    for record in records[:infrastructure]:
        record["error_present"] = True
    for record in records[:invariants]:
        record["rewards"] = {**record["rewards"], "_invariant_failure": 1.0}
    return records[:count]


def test_canary_proceeds_when_every_predeclared_condition_holds() -> None:
    result = evaluate_canary(_canary_records(), spec.CANARY_GATES)
    assert result["status"] == "proceed", result["failed_gates"]
    assert result["failed_gates"] == []
    assert result["qualifying_frontiers"] == ["frontier_gpt"]


def test_infrastructure_failure_precedes_every_product_verdict() -> None:
    result = evaluate_canary(_canary_records(infrastructure=1), spec.CANARY_GATES)
    assert result["status"] == "infrastructure_failure"


def test_invariant_failure_is_a_binding_product_result() -> None:
    result = evaluate_canary(_canary_records(invariants=1), spec.CANARY_GATES)
    assert result["status"] == "repair_again"


def test_too_strong_base_hardens() -> None:
    result = evaluate_canary(_canary_records(base=16, gpt=18), spec.CANARY_GATES)
    assert result["status"] == "harden"


def test_weak_frontier_reports_no_headroom() -> None:
    result = evaluate_canary(_canary_records(gpt=9, grok=9), spec.CANARY_GATES)
    assert result["status"] == "no_headroom"
    assert "one_frontier_meets_all_conditions" in result["failed_gates"]


def test_two_shot_contract_failure_is_binding_and_not_retryable() -> None:
    result = evaluate_canary(
        _canary_records(two_shot_classes=("execute", "clarify")), spec.CANARY_GATES
    )
    assert result["status"] == "repair_again"
    assert "two_shot_covers_every_outcome_class" in result["failed_gates"]


def test_missing_records_never_report_proceed() -> None:
    result = evaluate_canary(_canary_records(count=spec.CANARY_ROLLOUTS - 1), spec.CANARY_GATES)
    assert result["status"] == "repair_again"
    assert "exact_record_count_and_unique_ids" in result["failed_gates"]


def _summaries(base_rate: float, frontier_rate: float, **overrides) -> dict[str, Any]:
    def summary(rate: float) -> dict[str, Any]:
        return {
            "n": 90,
            "success_rate": rate,
            "infrastructure_failure_rate": overrides.get("infrastructure", 0.0),
            "model_attributable_failures": overrides.get("attributable", 30),
        }

    return {
        "small_base": summary(base_rate),
        "small_two_shot": summary(base_rate),
        "frontier_gpt": summary(frontier_rate),
        "frontier_grok": summary(frontier_rate - 0.05),
    }


@pytest.mark.parametrize(
    ("base_rate", "frontier_rate", "signal", "extra", "expected"),
    [
        (0.40, 0.80, True, {}, "go"),
        (0.90, 0.95, True, {}, "harden"),
        (0.05, 0.80, True, {}, "repair_again"),
        (0.40, 0.60, True, {}, "no_headroom"),
        (0.40, 0.45, True, {}, "no_headroom"),
        (0.40, 0.80, False, {}, "no_headroom"),
        (0.40, 0.80, True, {"infrastructure": 0.05}, "repair_again"),
        (0.40, 0.80, True, {"attributable": 3}, "repair_again"),
    ],
)
def test_full_decision_uses_the_unchanged_v1_thresholds(
    base_rate: float, frontier_rate: float, signal: bool, extra: dict, expected: str
) -> None:
    decision = v2_decision(_summaries(base_rate, frontier_rate, **extra), {"passes": signal})
    assert decision["status"] == expected
    assert set(decision["gates"]) == {
        "infrastructure_below_2_percent",
        "frontier_at_least_70_percent",
        "base_between_15_and_80_percent",
        "frontier_base_gap_at_least_10_points",
        "at_least_10_model_attributable_base_failures",
        "signal_probe_passes",
    }


# --- spec and selections ----------------------------------------------------


def test_frozen_spec_pins_sources_data_and_abi() -> None:
    _requires_spec()
    payload, digest = read_spec(SPEC_PATH)
    assert payload["schema_version"] == 2
    assert payload["benchmark_id"] == "order-resolution-v2"
    assert payload["artifact_creation_mode"] == "exclusive"
    assert payload["predecessor"]["decision"] == "repair"
    assert payload["environment"]["abi_sha256"] == environment_abi_sha256(EXAMPLE_ROOT)
    assert len(digest) == 64
    assert set(payload["selections"]) == {
        "canary_task_ids",
        "eval_task_ids",
        "stress_task_ids",
        "signal_probe_task_ids",
        "report_demo_task_ids",
        "oracle_demo_task_ids",
    }
    assert len(payload["selections"]["canary_task_ids"]) == 18
    assert len(payload["selections"]["eval_task_ids"]) == 90
    assert len(payload["selections"]["stress_task_ids"]) == 27
    assert len(payload["selections"]["signal_probe_task_ids"]) == 36


def test_spec_rejects_a_foreign_or_v1_specification(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(BenchmarkStateError, match="not a schema-v2"):
        read_spec(path)
    path.write_text(
        json.dumps({"schema_version": 2, "benchmark_id": "order-resolution-v3"}), encoding="utf-8"
    )
    with pytest.raises(BenchmarkStateError, match="another benchmark"):
        read_spec(path)


def test_task_selection_keeps_canary_disjoint_from_probe_and_demos() -> None:
    train = [json.loads(line) for line in (DATA_DIR / "train.jsonl").read_text().splitlines()]
    held_out = [json.loads(line) for line in (DATA_DIR / "eval.jsonl").read_text().splitlines()]
    selection = v2_task_selection(train, held_out)
    canary = set(selection["canary_task_ids"])
    assert canary.isdisjoint(selection["signal_probe_task_ids"])
    assert canary.isdisjoint(selection["oracle_demo_task_ids"])
    assert canary.isdisjoint(selection["eval_task_ids"])
    assert len(canary) == 18


def test_task_selection_rejects_an_incomplete_grid() -> None:
    train = [json.loads(line) for line in (DATA_DIR / "train.jsonl").read_text().splitlines()]
    held_out = [json.loads(line) for line in (DATA_DIR / "eval.jsonl").read_text().splitlines()]
    with pytest.raises(BenchmarkStateError, match="nine cells"):
        v2_task_selection(train[:20], held_out)


# --- demos ------------------------------------------------------------------


def test_two_shot_recomputes_the_canonical_example_id() -> None:
    demos = load_v2_oracle_demos(DATA_DIR / "oracle_traces.jsonl")
    row = json.loads((DATA_DIR / "eval.jsonl").read_text().splitlines()[0])
    from benchmax.envs import Example, canonical_example_id

    target = Example(id=canonical_example_id(row), payload=row)
    augmented = build_two_shot_example(target, demos)
    assert augmented.id != target.id
    assert augmented.id == canonical_example_id(augmented.payload)
    system = [m for m in augmented.payload["prompt_messages"] if m["role"] == "system"]
    assert len(system) == 1
    assert augmented.payload["prompt_messages"][-1] == row["prompt_messages"][-1]


def test_two_shot_rejects_mixed_benchmark_versions() -> None:
    demos = load_v2_oracle_demos(DATA_DIR / "oracle_traces.jsonl")
    row = json.loads((DATA_DIR / "eval.jsonl").read_text().splitlines()[0])
    from benchmax.envs import Example

    legacy = {**row, "benchmark_id": "order-resolution-v1"}
    with pytest.raises(ValueError, match="mix benchmark versions"):
        build_two_shot_example(Example(id="x", payload=legacy), demos)


def test_oracle_demo_loader_requires_both_live_v2_demos(tmp_path: Path) -> None:
    traces = [
        json.loads(line) for line in (DATA_DIR / "oracle_traces.jsonl").read_text().splitlines()
    ]
    keep = {trace["task_id"]: trace for trace in traces}
    path = tmp_path / "oracle_traces.jsonl"
    partial = [keep[spec.oracle_demo_task_ids()[0]]]
    path.write_text("".join(json.dumps(row) + "\n" for row in partial), encoding="utf-8")
    with pytest.raises(BenchmarkStateError, match="missing frozen v2 oracle demos"):
        load_v2_oracle_demos(path)

    demoted = [{**keep[task_id], "reward": 0.5} for task_id in spec.oracle_demo_task_ids()]
    path.write_text("".join(json.dumps(row) + "\n" for row in demoted), encoding="utf-8")
    with pytest.raises(BenchmarkStateError, match="does not score 1.0"):
        load_v2_oracle_demos(path)


# --- sealed wave verification ----------------------------------------------


def _write_wave(mirror: Path, *, wave: str, attempt: int | None, status: str) -> Path:
    """Build a sealed canary wave whose records reconcile with the frozen data."""

    from order_resolution.benchmark import _seal_payload, _sha256_file

    payload, spec_sha256 = read_spec(SPEC_PATH)
    root = mirror / (
        spec.CANARY_ATTEMPT_ROOTS[(attempt or 1) - 1] if wave == "canary" else spec.FULL_ROOT
    )
    root.mkdir(parents=True, exist_ok=True)
    rows = {
        row["task_id"]: row
        for line in (DATA_DIR / "train.jsonl").read_text().splitlines()
        for row in [json.loads(line)]
    }
    nonce = "testnonce"
    records = []
    for arm in spec.ARMS:
        for task_id in payload["selections"]["canary_task_ids"]:
            row = rows[task_id]
            rollout_id = v2_rollout_id(
                spec_sha256=spec_sha256,
                wave=wave,
                attempt=attempt,
                run_nonce=nonce,
                arm=arm.id,
                repetition=0,
                task_id=task_id,
                sample=0,
            )
            demos = load_v2_oracle_demos(DATA_DIR / "oracle_traces.jsonl")
            from benchmax.envs import Example

            example = Example(id=task_id, payload=row)
            if arm.prompt == spec.TWO_SHOT_PROMPT:
                example = build_two_shot_example(example, demos)
            initial = len(example.payload["prompt_messages"])
            reply = row["expected_reply"]
            success = _synthetic_success(arm.id, row, status)
            records.append(
                {
                    "phase": "canary",
                    "arm": arm.id,
                    "model": arm.model,
                    "repetition": 0,
                    "group_id": f"canary-{arm.id}-r0-{task_id}",
                    "rollout_id": rollout_id,
                    "world_id": world_id_for_rollout(rollout_id),
                    "benchmark_id": spec.BENCHMARK_ID,
                    "wave": wave,
                    "attempt": attempt,
                    "sample": 0,
                    "task_id": task_id,
                    "cell": row["cell"],
                    "action_family": row["action_family"],
                    "outcome_class": row["outcome_class"],
                    "expected_disposition": row["expected_disposition"],
                    "predicted_disposition": reply["disposition"],
                    "task_success": 1.0 if success else 0.0,
                    "rewards": {"_forbidden_mutation": 0.0, "_invariant_failure": 0.0},
                    "termination_reason": "finished",
                    "error_present": status == "infrastructure_failure",
                    "latency_seconds": 1.0,
                    "initial_message_count": initial,
                    "reply_call_count": 1,
                    "tool_call_count": 1,
                    "invalid_tool_call_count": 0,
                    "messages": [
                        *example.payload["prompt_messages"],
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-0",
                                    "type": "function",
                                    "function": {
                                        "name": "reply_to_customer",
                                        "arguments": json.dumps(reply, sort_keys=True),
                                    },
                                }
                            ],
                        },
                    ],
                }
            )
    raw_path = root / "rollouts.raw.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in sorted(records, key=lambda record: record["rollout_id"])
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "benchmark_id": spec.BENCHMARK_ID,
        "spec_sha256": spec_sha256,
        "wave": wave,
        "attempt": attempt,
        "run_nonce": nonce,
        "started_at": "2026-08-06T00:00:00Z",
        "completed_at": "2026-08-06T00:10:00Z",
        "neon": {"branch_id": "br-test", "deleted": True, "deleted_at": "2026-08-06T00:11:00Z"},
        "bundle": {"digest": "bundle-test", "secret_boundary": "ok"},
        "environment_abi_sha256": payload["environment"]["abi_sha256"],
        "datasets": dict(payload["datasets"]),
        "selections": dict(payload["selections"]),
        "execution": {
            "concurrency": spec.CANARY_CONCURRENCY,
            "group_size": 1,
            "stress_repeats": 0,
            "training_group_size": 0,
        },
        "models": {
            "arms": [
                {"id": arm.id, "model": arm.model, "endpoint": "https://x/v1", "prompt": arm.prompt}
                for arm in spec.ARMS
            ]
        },
        "gates": dict(spec.CANARY_GATES),
        "artifacts": {"raw_rollouts": raw_path.name, "seal": "seal.json"},
        "rollout_count": len(records),
    }
    loaded = [json.loads(line) for line in raw_path.read_text().splitlines()]
    manifest["canary"] = evaluate_canary(loaded, manifest["gates"])
    manifest["status"] = manifest["canary"]["status"]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    seal = _seal_payload({"manifest": manifest_path, "raw": raw_path, "seal": root / "seal.json"})
    (root / "seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", "utf-8")
    assert _sha256_file(manifest_path) == seal["sha256"]["manifest"]
    return manifest_path


def _synthetic_success(arm_id: str, row: dict, status: str) -> bool:
    """A passing shape: strong GPT, weaker Grok, base at the band's floor."""

    index = int(row["task_id"].rsplit("-", 1)[1])
    first = index == spec.CANARY_INDICES[0]
    if arm_id == "frontier_gpt":
        return True
    if arm_id == "frontier_grok":
        return first
    if arm_id == "small_two_shot":
        return first
    return first and row["outcome_class"] == "execute"


def test_sealed_canary_verifies_and_gates_on_status(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    result = verify_v2_benchmark(manifest_path, example_root=mirror)
    assert result["wave"] == "canary"
    assert result["rollouts"] == spec.CANARY_ROLLOUTS
    assert result["status"] == "proceed"
    assert result["failed_gates"] == []
    verify_v2_benchmark(manifest_path, example_root=mirror, require_status="proceed")
    with pytest.raises(BenchmarkStateError, match="required status"):
        verify_v2_benchmark(manifest_path, example_root=mirror, require_status="no_headroom")


# --- exclusive creation and the retry/authorization state machine ------------


def test_wave_paths_refuse_to_start_over_an_existing_artifact(mirror: Path) -> None:
    from order_resolution.benchmark import _wave_paths

    root = mirror / spec.CANARY_ATTEMPT_ROOTS[0]
    root.mkdir(parents=True)
    paths = _wave_paths(mirror, root / "manifest.json", "canary")
    assert set(paths) == {"manifest", "raw", "seal"}
    for name in ("manifest.json", "rollouts.raw.jsonl", "seal.json"):
        (root / name).write_text("{}", encoding="utf-8")
        with pytest.raises(BenchmarkStateError, match="already exists"):
            _wave_paths(mirror, root / "manifest.json", "canary")
        (root / name).unlink()


def test_wave_paths_reject_unfrozen_locations(mirror: Path) -> None:
    from order_resolution.benchmark import _wave_paths

    with pytest.raises(BenchmarkStateError, match="not a permitted v2 artifact path"):
        _wave_paths(mirror, mirror / spec.ARTIFACT_ROOT / "canary" / "attempt-03" / "manifest.json",
                    "canary")
    with pytest.raises(BenchmarkStateError, match="named manifest.json"):
        _wave_paths(mirror, mirror / spec.CANARY_ATTEMPT_ROOTS[0] / "seal.json", "canary")


def test_create_exclusive_never_overwrites(tmp_path: Path) -> None:
    from order_resolution.benchmark import _create_exclusive

    path = tmp_path / "seal.json"
    _create_exclusive(path, "{}\n", ())
    with pytest.raises(BenchmarkStateError, match="refusing to overwrite"):
        _create_exclusive(path, "{}\n", ())
    assert path.read_text(encoding="utf-8") == "{}\n"


def test_create_exclusive_refuses_secret_bearing_payloads(tmp_path: Path) -> None:
    from order_resolution.benchmark import _create_exclusive

    path = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="secret-bearing"):
        _create_exclusive(path, '{"url": "postgresql://x:y@z/db"}', ["postgresql://x:y@z/db"])
    assert not path.exists()


def _preconditions(mirror: Path, **overrides):
    from order_resolution.benchmark import _assert_wave_preconditions

    _, spec_sha256 = read_spec(SPEC_PATH)
    arguments = {
        "example_root": mirror,
        "spec_sha256": spec_sha256,
        "wave": "canary",
        "attempt": 1,
        "manifest_path": mirror / spec.CANARY_ATTEMPT_ROOTS[0] / "manifest.json",
        "authorization_path": mirror / spec.CANARY_AUTHORIZATION_PATH,
        "requires_infrastructure_failure": None,
        "requires_canary": None,
    }
    return _assert_wave_preconditions(**{**arguments, **overrides})


def test_attempt_1_needs_no_prior_and_rejects_a_stray_prior(mirror: Path) -> None:
    assert _preconditions(mirror) is None
    with pytest.raises(BenchmarkStateError, match="attempt 1 never requires"):
        _preconditions(mirror, requires_infrastructure_failure=Path("x"))


def test_attempt_2_requires_a_sealed_infrastructure_failure(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    attempt_2 = mirror / spec.CANARY_ATTEMPT_ROOTS[1] / "manifest.json"
    with pytest.raises(BenchmarkStateError, match="requires --requires-infrastructure-failure"):
        _preconditions(mirror, attempt=2, manifest_path=attempt_2)
    # A product result never unlocks the second attempt.
    with pytest.raises(BenchmarkStateError, match="only infrastructure_failure retries"):
        _preconditions(
            mirror,
            attempt=2,
            manifest_path=attempt_2,
            requires_infrastructure_failure=manifest_path,
        )


def test_attempt_2_accepts_a_sealed_infrastructure_failure(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="infrastructure_failure")
    assert json.loads(manifest_path.read_text())["status"] == "infrastructure_failure"
    prior = _preconditions(
        mirror,
        attempt=2,
        manifest_path=mirror / spec.CANARY_ATTEMPT_ROOTS[1] / "manifest.json",
        requires_infrastructure_failure=manifest_path,
    )
    assert prior["status"] == "infrastructure_failure"


def test_an_existing_authorization_closes_canary_execution(mirror: Path) -> None:
    authorization = mirror / spec.CANARY_AUTHORIZATION_PATH
    authorization.parent.mkdir(parents=True, exist_ok=True)
    authorization.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkStateError, match="canary execution is closed"):
        _preconditions(mirror)


def test_attempt_3_does_not_exist(mirror: Path) -> None:
    with pytest.raises(BenchmarkStateError, match="attempt must be 1..2"):
        _preconditions(mirror, attempt=3)


def test_full_wave_requires_a_passing_authorization(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    full_manifest = mirror / spec.FULL_ROOT / "manifest.json"
    with pytest.raises(BenchmarkStateError, match="requires --requires-canary"):
        _preconditions(mirror, wave="full", attempt=None, manifest_path=full_manifest)

    _, spec_sha256 = read_spec(SPEC_PATH)
    authorization = mirror / spec.CANARY_AUTHORIZATION_PATH
    from order_resolution.benchmark import _sha256_file

    payload = {
        "benchmark_id": spec.BENCHMARK_ID,
        "spec_sha256": spec_sha256,
        "attempt": 1,
        "attempt_status": "proceed",
        "attempt_manifest_path": spec.CANARY_ATTEMPT_ROOTS[0] + "/manifest.json",
        "attempt_manifest_sha256": _sha256_file(manifest_path),
    }
    authorization.write_text(json.dumps(payload), encoding="utf-8")
    accepted = _preconditions(
        mirror,
        wave="full",
        attempt=None,
        manifest_path=full_manifest,
        requires_canary=authorization,
    )
    assert accepted["attempt_status"] == "proceed"

    authorization.write_text(
        json.dumps({**payload, "attempt_status": "infrastructure_failure"}), encoding="utf-8"
    )
    with pytest.raises(BenchmarkStateError, match="does not record a passing canary"):
        _preconditions(
            mirror,
            wave="full",
            attempt=None,
            manifest_path=full_manifest,
            requires_canary=authorization,
        )

    authorization.write_text(json.dumps({**payload, "spec_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(BenchmarkStateError, match="different spec"):
        _preconditions(
            mirror,
            wave="full",
            attempt=None,
            manifest_path=full_manifest,
            requires_canary=authorization,
        )


def test_authorization_is_created_exactly_once(mirror: Path) -> None:
    from order_resolution.benchmark import _authorize_canary

    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    _, spec_sha256 = read_spec(SPEC_PATH)
    authorization = mirror / spec.CANARY_AUTHORIZATION_PATH
    arguments = dict(
        example_root=mirror,
        authorization_path=authorization,
        manifest_path=manifest_path,
        spec_sha256=spec_sha256,
        attempt=1,
        prior=None,
    )
    _authorize_canary(**arguments)
    payload = json.loads(authorization.read_text())
    assert payload["attempt_status"] == "proceed"
    assert payload["attempt_manifest_path"] == spec.CANARY_ATTEMPT_ROOTS[0] + "/manifest.json"
    assert payload["attempt_seal_sha256"]
    with pytest.raises(BenchmarkStateError, match="refusing to overwrite"):
        _authorize_canary(**arguments)


def test_attempt_2_authorization_must_embed_the_prior_failure(mirror: Path) -> None:
    from order_resolution.benchmark import _authorize_canary

    manifest_path = _write_wave(mirror, wave="canary", attempt=2, status="proceed")
    _, spec_sha256 = read_spec(SPEC_PATH)
    arguments = dict(
        example_root=mirror,
        authorization_path=mirror / spec.CANARY_AUTHORIZATION_PATH,
        manifest_path=manifest_path,
        spec_sha256=spec_sha256,
        attempt=2,
    )
    with pytest.raises(BenchmarkStateError, match="sealed attempt-1"):
        _authorize_canary(**arguments, prior=None)
    _authorize_canary(
        **arguments,
        prior={"status": "infrastructure_failure", "spec_sha256": spec_sha256},
    )
    payload = json.loads((mirror / spec.CANARY_AUTHORIZATION_PATH).read_text())
    assert payload["prior_infrastructure_failure"]["status"] == "infrastructure_failure"


def test_verification_rejects_a_tampered_raw_file(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    raw_path = manifest_path.parent / "rollouts.raw.jsonl"
    lines = raw_path.read_text().splitlines()
    raw_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkStateError, match="modified after sealing"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_verification_rejects_a_tampered_manifest(mirror: Path) -> None:
    """Editing a sealed verdict breaks the seal before any gate is consulted."""

    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "proceed"
    manifest["status"] = "infrastructure_failure"
    manifest["canary"]["status"] = "infrastructure_failure"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    with pytest.raises(BenchmarkStateError, match="modified after sealing"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_reseal_cannot_launder_a_rewritten_verdict(mirror: Path) -> None:
    """Even a re-sealed manifest must still reconcile with its raw rollouts."""

    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")

    def flip(manifest: dict) -> None:
        manifest["status"] = "infrastructure_failure"
        manifest["canary"]["status"] = "infrastructure_failure"

    _retamper(manifest_path, flip)
    with pytest.raises(BenchmarkStateError, match="gate evaluation does not reconcile"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_verification_requires_the_wave_to_sit_at_its_frozen_path(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    moved = mirror / spec.CANARY_ATTEMPT_ROOTS[1]
    moved.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "rollouts.raw.jsonl", "seal.json"):
        (moved / name).write_bytes((manifest_path.parent / name).read_bytes())
    with pytest.raises(BenchmarkStateError, match="not at its frozen path"):
        verify_v2_benchmark(moved / "manifest.json", example_root=mirror)


def test_verification_rejects_a_missing_seal(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    (manifest_path.parent / "seal.json").unlink()
    with pytest.raises(BenchmarkStateError, match="no seal.json"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_verification_rejects_an_undeleted_branch(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    _retamper(manifest_path, lambda manifest: manifest["neon"].update(deleted=False))
    with pytest.raises(BenchmarkStateError, match="not recorded as deleted"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_verification_rejects_a_wrong_arm_mapping(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")

    def swap(manifest: dict) -> None:
        manifest["models"]["arms"][0]["model"] = "gpt-5.6-sol"

    _retamper(manifest_path, swap)
    with pytest.raises(BenchmarkStateError, match="does not match the frozen mapping"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_verification_rejects_a_foreign_spec_digest(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    _retamper(manifest_path, lambda manifest: manifest.update(spec_sha256="0" * 64))
    with pytest.raises(BenchmarkStateError, match="does not reference the frozen spec"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def test_verification_rejects_a_drifted_environment_abi(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    _retamper(manifest_path, lambda manifest: manifest.update(environment_abi_sha256="0" * 64))
    with pytest.raises(BenchmarkStateError, match="different environment ABI"):
        verify_v2_benchmark(manifest_path, example_root=mirror)


def _retamper(manifest_path: Path, mutate) -> None:
    """Rewrite a manifest and its seal so only the intended check can fail."""

    from order_resolution.benchmark import _seal_payload

    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    root = manifest_path.parent
    seal = _seal_payload(
        {
            "manifest": manifest_path,
            "raw": root / "rollouts.raw.jsonl",
            "seal": root / "seal.json",
        }
    )
    (root / "seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", "utf-8")


def test_schema_v1_manifests_keep_their_verification_path(mirror: Path) -> None:
    result = verify_v2_benchmark(mirror / "artifacts" / "baseline.json", example_root=mirror)
    assert result["decision"] == "repair"
    assert result["rollouts"] == 972


def test_report_check_requires_the_full_wave(mirror: Path) -> None:
    manifest_path = _write_wave(mirror, wave="canary", attempt=1, status="proceed")
    with pytest.raises(BenchmarkStateError, match="only the full wave"):
        check_v2_report(manifest_path, example_root=mirror)
