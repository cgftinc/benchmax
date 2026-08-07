"""Deterministic split, leakage, timestamp, and oracle checks."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from benchmax.envs import Example
from order_resolution import benchmark_spec as spec
from order_resolution.benchmark_spec import verify_predecessor
from order_resolution.contract import oracle_rollout_id
from order_resolution.fixtures import (
    DEFAULT_SEED,
    StaleCatalogError,
    V2Catalog,
    V2Data,
    build_catalog,
    build_v2_catalog,
    check_data,
    check_v2_data,
    generate_data,
    generate_v2_data,
    initial_snapshot,
    oracle_after_snapshot,
    prompt_skeleton,
    read_olist_calibration,
    render_jsonl,
    sync_catalog_namespace,
    validate_v2_generated,
    write_data,
    write_v2_data,
)
from order_resolution.grading import grade_snapshots
from order_resolution.order_env import OrderResolutionEnv
from order_resolution.policy import render_system_contract, validate_reply
from order_resolution.schema import product_variants
from sqlalchemy.ext.asyncio import AsyncEngine


def test_catalog_has_realistic_linked_cardinality() -> None:
    catalog = build_catalog()
    variants = [variant for product in catalog for variant in product["variants"]]
    assert len(catalog) == 250
    assert len(variants) == 750
    assert len({product["product_id"] for product in catalog}) == 250
    assert len({variant["variant_id"] for variant in variants}) == 750
    assert all(isinstance(variant["price_minor"], int) for variant in variants)
    assert {variant["currency"] for variant in variants} == {"USD"}


def test_generation_is_byte_stable_and_balanced() -> None:
    first = generate_data(DEFAULT_SEED)
    second = generate_data(DEFAULT_SEED)
    assert first.hashes == second.hashes
    assert first.train == second.train
    assert first.eval == second.eval
    assert len(first.train) == 180
    assert len(first.eval) == 90
    assert set(Counter(row["cell"] for row in first.train).values()) == {20}
    assert set(Counter(row["cell"] for row in first.eval).values()) == {10}


def test_changed_seed_changes_semantic_fixture_and_prompt() -> None:
    first = generate_data(DEFAULT_SEED).train[0]
    changed = generate_data(DEFAULT_SEED + 1).train[0]
    assert first["scenario_seed"] != changed["scenario_seed"]
    assert first["fixture"]["requested_address"] != changed["fixture"]["requested_address"]
    assert first["prompt_messages"] != changed["prompt_messages"]


def test_split_has_no_family_entity_or_template_leakage() -> None:
    generated = generate_data(DEFAULT_SEED)
    for key in ("scenario_family_id", "prompt_template_id"):
        assert {row[key] for row in generated.train}.isdisjoint(
            {row[key] for row in generated.eval}
        )
    assert {row["fixture"]["ids"]["product_id"] for row in generated.train}.isdisjoint(
        {row["fixture"]["ids"]["product_id"] for row in generated.eval}
    )
    assert {row["fixture"]["ids"]["customer_id"] for row in generated.train}.isdisjoint(
        {row["fixture"]["ids"]["customer_id"] for row in generated.eval}
    )


def test_scenarios_have_fixed_valid_timestamps_and_expected_reply_contracts() -> None:
    generated = generate_data(DEFAULT_SEED)
    for row in (*generated.train, *generated.eval):
        as_of = datetime.fromisoformat(row["as_of"])
        order = row["fixture"]["initial_snapshot"]["orders"][row["fixture"]["ids"]["order_number"]]
        assert as_of.tzinfo is not None
        assert datetime.fromisoformat(order["created_at"]) < as_of
        assert row["expected_reply"]["missing_fields"] == sorted(
            row["expected_reply"]["missing_fields"]
        )
        assert row["prompt_messages"][0]["role"] == "system"
        assert row["prompt_messages"][1]["role"] == "user"


def test_every_oracle_terminal_state_passes_exact_grader() -> None:
    generated = generate_data(DEFAULT_SEED)
    traces = {trace["task_id"]: trace for trace in generated.oracle_traces}
    assert set(traces) == {row["task_id"] for row in generated.train}
    all_call_ids: list[str] = []
    for row in generated.train:
        grade = grade_snapshots(
            before=initial_snapshot(row),
            after=oracle_after_snapshot(row),
            required=row["required_state"],
            forbidden=row["forbidden_state"],
            expected_disposition=row["expected_disposition"],
            expected_reply=row["expected_reply"],
        )
        assert grade.task_success == 1.0, (row["task_id"], grade.failures)
        trace = traces[row["task_id"]]
        assert trace["reward"] == 1.0
        reply_calls = [
            call
            for message in trace["completion_messages"]
            for call in message.get("tool_calls", [])
            if call["function"]["name"] == "reply_to_customer"
        ]
        assert len(reply_calls) == 1
        all_call_ids.extend(
            call["id"]
            for message in trace["completion_messages"]
            for call in message.get("tool_calls", [])
        )
    assert len(all_call_ids) == len(set(all_call_ids))


def test_check_detects_drift(tmp_path: Path) -> None:
    write_data(tmp_path, seed=DEFAULT_SEED, force=True)
    assert check_data(tmp_path, seed=DEFAULT_SEED) == generate_data(DEFAULT_SEED).hashes
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="generated data drift"):
        check_data(tmp_path, seed=DEFAULT_SEED)


def test_frozen_eval_hash_does_not_move_with_generator(tmp_path: Path) -> None:
    write_data(tmp_path, seed=DEFAULT_SEED, force=True)
    frozen = (tmp_path / "eval.sha256").read_text(encoding="utf-8")
    write_data(tmp_path, seed=DEFAULT_SEED + 1, force=True)
    assert (tmp_path / "eval.sha256").read_text(encoding="utf-8") == frozen
    with pytest.raises(RuntimeError, match="frozen pre-run hash"):
        check_data(tmp_path, seed=DEFAULT_SEED + 1)


# ---------------------------------------------------------------------------
# order-resolution-v2
# ---------------------------------------------------------------------------

V2_DATA_DIR = Path(__file__).parents[1] / "data" / "v2"


@pytest.fixture(scope="module")
def v2() -> V2Data:
    return generate_v2_data()


def test_v2_generation_is_byte_stable_and_balanced(v2: V2Data) -> None:
    again = generate_v2_data()
    assert render_jsonl(v2.train) == render_jsonl(again.train)
    assert render_jsonl(v2.eval) == render_jsonl(again.eval)
    assert v2.hashes == again.hashes
    assert len(v2.train) == 180 and len(v2.eval) == 90
    assert set(Counter(row["cell"] for row in v2.train).values()) == {20}
    assert set(Counter(row["cell"] for row in v2.eval).values()) == {10}
    validate_v2_generated(v2)


def test_v2_catalog_is_content_addressed_without_a_digest_cycle(v2: V2Data) -> None:
    catalog = build_v2_catalog()
    assert catalog.generation_key_sha256 == v2.catalog.generation_key_sha256
    assert catalog.content_sha256 == v2.catalog.content_sha256
    assert catalog.generation_key_sha256 != catalog.content_sha256
    assert len(catalog.products) == 250
    assert len(catalog.variants()) == 750
    # Identifiers embed the generation key; the content digest covers them.
    assert all(
        product_id.startswith(f"p{catalog.id_prefix}-") for product_id in catalog.product_ids()
    )
    assert all(
        variant["variant_id"].startswith(f"v{catalog.id_prefix}-") for variant in catalog.variants()
    )
    assert len(set(catalog.product_ids())) == 250
    assert len({variant["variant_id"] for variant in catalog.variants()}) == 750
    assert len({variant["sku"] for variant in catalog.variants()}) == 750


def test_v2_catalog_namespace_moves_when_contents_change(monkeypatch) -> None:
    before = build_v2_catalog()
    monkeypatch.setattr(spec, "CATALOG_GENERATOR_VERSION", spec.CATALOG_GENERATOR_VERSION + 1)
    after = build_v2_catalog()
    assert after.generation_key_sha256 != before.generation_key_sha256
    assert set(after.product_ids()).isdisjoint(before.product_ids())


def test_v2_visible_state_carries_no_split_or_version_marker(v2: V2Data) -> None:
    """v1 leaked `item-train-000-a` and `OR-T00000`; v2 identifiers are opaque hashes."""

    for row in (*v2.train, *v2.eval):
        ids = row["fixture"]["ids"]
        visible = json.dumps(
            {
                "ids": ids,
                "prompt": row["prompt_messages"][-1]["content"],
                "expected_reply": row["expected_reply"],
            }
        ).lower()
        for marker in ("train", "eval", "order-resolution", "v2"):
            assert marker not in visible, (row["task_id"], marker)
        # Per-scenario identifiers are hash bodies, so they cannot encode a split.
        assert re.fullmatch(r"OR-[0-9A-F]{8}", ids["order_number"]), row["task_id"]
        for key in ("customer_id", "target_item_id", "distractor_item_id", "address_id"):
            kind, _, body = ids[key].partition("-")
            assert kind in {"customer", "item", "address"}
            assert re.fullmatch(r"[0-9a-f]{12}", body), (row["task_id"], key)


def test_v2_splits_share_no_entity_family_or_wording(v2: V2Data) -> None:
    for key in ("scenario_family_id", "prompt_template_id"):
        assert {row[key] for row in v2.train}.isdisjoint({row[key] for row in v2.eval})
    for key in ("target_product_id", "distractor_product_id", "customer_id", "order_number"):
        assert {row["fixture"]["ids"][key] for row in v2.train}.isdisjoint(
            {row["fixture"]["ids"][key] for row in v2.eval}
        )
    train_skeletons = {
        prompt_skeleton(row["prompt_messages"][-1]["content"]) for row in v2.train
    }
    eval_skeletons = {prompt_skeleton(row["prompt_messages"][-1]["content"]) for row in v2.eval}
    assert train_skeletons.isdisjoint(eval_skeletons)
    assert Counter(row["prompt_stratum"] for row in v2.train) == Counter(
        row["prompt_stratum"] for row in v2.train
    )
    assert set(Counter(row["prompt_stratum"] for row in v2.eval).values()) == {18}


def test_v2_item_tasks_have_exactly_one_visible_match(v2: V2Data) -> None:
    for row in (*v2.train, *v2.eval):
        if row["action_family"] == "change_address":
            continue
        message = row["prompt_messages"][-1]["content"].lower()
        names = [row["fixture"]["target_product_name"], row["fixture"]["distractor_product_name"]]
        matches = [name for name in names if name.lower() in message]
        if row["cell"] == "cancel_item-clarify":
            assert matches == [], row["task_id"]
        else:
            assert matches == [row["fixture"]["target_product_name"]], row["task_id"]
        assert set(names[0].split()).isdisjoint(names[1].split())


def test_v2_clarifications_omit_exactly_the_named_fact(v2: V2Data) -> None:
    for row in (*v2.train, *v2.eval):
        if row["outcome_class"] != "clarify":
            assert row["expected_reply"]["missing_fields"] == []
            continue
        message = row["prompt_messages"][-1]["content"].lower()
        address = row["fixture"]["requested_address"]
        if row["action_family"] == "change_address":
            assert row["expected_reply"]["missing_fields"] == ["shipping_address.postal_code"]
            assert address["postal_code"] not in message
            for field in ("line1", "city", "region", "country"):
                assert str(address[field]).lower() in message, (row["task_id"], field)
        elif row["action_family"] == "replace_variant":
            assert row["expected_reply"]["missing_fields"] == ["requested_options.size"]
            assert row["fixture"]["requested_size"] is None
            assert not any(size in message for size in spec.CATALOG_SIZES)
            assert row["fixture"]["target_product_name"].lower() in message
            assert row["expected_reply"]["order_item_id"] is not None
        else:
            assert row["expected_reply"]["missing_fields"] == ["order_item_id"]
            assert row["expected_reply"]["order_item_id"] is None


def test_v2_expected_replies_satisfy_the_published_policy(v2: V2Data) -> None:
    for row in (*v2.train, *v2.eval):
        reply = row["expected_reply"]
        validate_reply(
            disposition=reply["disposition"],
            outcome_code=reply["outcome_code"],
            order_item_id=reply["order_item_id"],
            missing_fields=reply["missing_fields"],
        )
        assert reply["disposition"] == row["expected_disposition"]
        assert reply["order_number"] == row["fixture"]["ids"]["order_number"]


def test_v2_system_prompt_is_the_published_contract(v2: V2Data) -> None:
    contract = render_system_contract()
    for row in (*v2.train, *v2.eval):
        system = row["prompt_messages"][0]
        assert system["role"] == "system"
        assert system["content"] == contract
        # The public contract must not carry this row's answer.
        assert row["expected_reply"]["outcome_code"] in contract
        assert row["fixture"]["ids"]["order_number"] not in contract


def test_v1_bytes_are_untouched_by_v2_generation(v2: V2Data) -> None:
    verify_predecessor(Path(__file__).parents[1])


# --- compiled oracle traces -------------------------------------------------


def _oracle_traces() -> list[dict]:
    path = V2_DATA_DIR / "oracle_traces.jsonl"
    if not path.exists():
        pytest.skip("compile the v2 oracles first via contract-test --compile-oracles")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_compiled_oracles_cover_the_training_split_with_full_reward(v2: V2Data) -> None:
    traces = _oracle_traces()
    assert {trace["task_id"] for trace in traces} == {row["task_id"] for row in v2.train}
    assert len(traces) == 180
    assert all(trace["reward"] == 1.0 for trace in traces)
    assert all(trace["benchmark_id"] == "order-resolution-v2" for trace in traces)


def test_compiled_oracles_pair_unique_call_ids_and_one_terminal_reply() -> None:
    call_ids: list[str] = []
    for trace in _oracle_traces():
        calls = [
            call
            for message in trace["completion_messages"]
            for call in message.get("tool_calls", [])
        ]
        results = [
            message for message in trace["completion_messages"] if message["role"] == "tool"
        ]
        assert len(calls) == len(results)
        assert [call["id"] for call in calls] == [
            message["tool_call_id"] for message in results
        ]
        replies = [call for call in calls if call["function"]["name"] == "reply_to_customer"]
        assert len(replies) == 1
        assert calls[-1] is replies[0], trace["task_id"]
        call_ids.extend(call["id"] for call in calls)
    assert len(call_ids) == len(set(call_ids))


def test_compiled_oracles_check_availability_before_replacing() -> None:
    for trace in _oracle_traces():
        names = [
            call["function"]["name"]
            for message in trace["completion_messages"]
            for call in message.get("tool_calls", [])
        ]
        assert names[0] == "get_order"
        if "replace_order_item_variant" in names:
            assert names.index("check_variant_availability") < names.index(
                "replace_order_item_variant"
            )


def test_compiled_oracles_use_only_observed_values(v2: V2Data) -> None:
    """Every argument must come from the customer or an earlier tool result."""

    free_text = {"reason"}
    rows = {row["task_id"]: row for row in v2.train}
    for trace in _oracle_traces():
        row = rows[trace["task_id"]]
        observed = " ".join(message["content"] for message in trace["prompt_messages"])
        for message in trace["completion_messages"]:
            for call in message.get("tool_calls", []):
                arguments = json.loads(call["function"]["arguments"])
                for key, value in _string_leaves(arguments):
                    if key in free_text:
                        continue
                    assert value in observed, (trace["task_id"], key, value)
            if message["role"] == "tool":
                observed += " " + message["content"]
        hidden = {
            value
            for key, value in row["fixture"]["ids"].items()
            if isinstance(value, str) and key not in {"order_number"}
        }
        first_call = json.loads(
            trace["completion_messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        assert hidden.isdisjoint(str(value) for _, value in _string_leaves(first_call))


def _string_leaves(value, key: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(key, value)]
    if isinstance(value, dict):
        return [pair for name, item in value.items() for pair in _string_leaves(item, name)]
    if isinstance(value, list):
        return [pair for item in value for pair in _string_leaves(item, key)]
    return []


def test_frozen_demo_cases_are_the_two_declared_cells() -> None:
    traces = {trace["task_id"] for trace in _oracle_traces()}
    assert set(spec.oracle_demo_task_ids()) <= traces
    assert spec.oracle_demo_task_ids() == (
        "train-cancel_item-clarify-00",
        "train-replace_variant-deny-00",
    )


def test_v2_check_detects_drift(tmp_path: Path) -> None:
    write_v2_data(tmp_path, force=True)
    assert check_v2_data(tmp_path, require_oracles=False)["train.jsonl"]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="generated v2 data drift"):
        check_v2_data(tmp_path, require_oracles=False)


def test_v2_check_requires_compiled_oracles(tmp_path: Path) -> None:
    write_v2_data(tmp_path, force=True)
    with pytest.raises(RuntimeError, match="missing compiled v2 oracle traces"):
        check_v2_data(tmp_path)


# --- catalog namespace reconciliation ---------------------------------------


def _small_catalog(prefix: str, *, name: str = "Test Item") -> V2Catalog:
    key = prefix + "0" * (64 - len(prefix))
    products = tuple(
        {
            "product_id": f"p{prefix}-{index:03d}",
            "name": f"{name} {index}",
            "category": "home",
            "variants": [
                {
                    "variant_id": f"v{prefix}-{index:03d}-0",
                    "sku": f"SKU-{prefix.upper()}-{index:03d}0",
                    "price_minor": 1_000,
                    "currency": "USD",
                    "options": {"size": "small"},
                    "active": True,
                }
            ],
        }
        for index in range(2)
    )
    return V2Catalog(products=products, generation_key_sha256=key, content_sha256="0" * 64)


@pytest.mark.asyncio
async def test_catalog_namespace_inserts_once_then_verifies(engine: AsyncEngine) -> None:
    catalog = _small_catalog("aaaaaaaaaaaa")
    async with engine.begin() as connection:
        assert await sync_catalog_namespace(connection, catalog) == "inserted"
    async with engine.begin() as connection:
        assert await sync_catalog_namespace(connection, catalog) == "verified"


@pytest.mark.asyncio
async def test_catalog_namespace_rejects_changed_content(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await sync_catalog_namespace(connection, _small_catalog("bbbbbbbbbbbb"))
    changed = _small_catalog("bbbbbbbbbbbb", name="Renamed Item")
    async with engine.begin() as connection:
        with pytest.raises(StaleCatalogError, match="do not match the frozen v2 catalog"):
            await sync_catalog_namespace(connection, changed)


@pytest.mark.asyncio
async def test_catalog_namespace_rejects_a_partial_set(engine: AsyncEngine) -> None:
    """A half-populated namespace is a blocking error, never a silent merge."""

    catalog = _small_catalog("cccccccccccc")
    async with engine.begin() as connection:
        await sync_catalog_namespace(connection, catalog)
        await connection.execute(
            sa.delete(product_variants).where(
                product_variants.c.variant_id == catalog.products[1]["variants"][0]["variant_id"]
            )
        )
    async with engine.begin() as connection:
        with pytest.raises(StaleCatalogError):
            await sync_catalog_namespace(connection, catalog)


@pytest.mark.asyncio
async def test_frozen_v2_catalog_reconciles_exactly(engine: AsyncEngine) -> None:
    catalog = build_v2_catalog()
    async with engine.begin() as connection:
        first = await sync_catalog_namespace(connection, catalog)
    async with engine.begin() as connection:
        assert await sync_catalog_namespace(connection, catalog) == "verified"
    assert first in {"inserted", "verified"}


# --- live service equality --------------------------------------------------


@pytest.mark.asyncio
async def test_recorded_oracle_results_match_a_live_replay(database, engine: AsyncEngine) -> None:
    """The stored trace is the service's own output, not a synthesized summary."""

    async with engine.begin() as connection:
        await sync_catalog_namespace(connection, build_v2_catalog())
    traces = {trace["task_id"]: trace for trace in _oracle_traces()}
    rows = {row["task_id"]: row for row in generate_v2_data().train}
    selected = [
        "train-cancel_item-execute-00",
        "train-change_address-clarify-00",
        "train-replace_variant-deny-00",
    ]
    env = OrderResolutionEnv(_database_url_for_tests())
    try:
        for task_id in selected:
            row = rows[task_id]
            example = Example(id=task_id, payload=row)
            rollout_id = oracle_rollout_id(task_id)
            recorded = [
                (
                    json.loads(call["function"]["arguments"]),
                    call["function"]["name"],
                    json.loads(result["content"]),
                )
                for call, result in _paired_calls(traces[task_id])
            ]
            async with env.rollout_context(rollout_id, example):
                for arguments, name, expected in recorded:
                    live = await env.run_tool(rollout_id, name, **arguments)
                    assert json.loads(json.dumps(live)) == expected, (task_id, name)
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_target_resolution_is_invariant_under_item_order(
    database, engine: AsyncEngine
) -> None:
    async with engine.begin() as connection:
        await sync_catalog_namespace(connection, build_v2_catalog())
    row = next(
        row for row in generate_v2_data().train if row["cell"] == "cancel_item-execute"
    )
    permuted = json.loads(json.dumps(row))
    snapshot = permuted["fixture"]["initial_snapshot"]
    for key in ("order_items", "allocations", "inventory"):
        snapshot[key] = dict(reversed(list(snapshot[key].items())))

    env = OrderResolutionEnv(_database_url_for_tests())
    try:
        results = []
        for suffix, candidate in (("a", row), ("b", permuted)):
            rollout_id = f"permutation-{suffix}"
            async with env.rollout_context(rollout_id, Example(id=rollout_id, payload=candidate)):
                results.append(
                    await env.run_tool(
                        rollout_id,
                        "get_order",
                        order_number=candidate["fixture"]["ids"]["order_number"],
                    )
                )
    finally:
        await env.aclose()
    assert results[0] == results[1]
    message = row["prompt_messages"][-1]["content"].lower()
    matched = [
        item for item in results[0]["items"] if str(item["product_name"]).lower() in message
    ]
    assert len(matched) == 1
    assert matched[0]["order_item_id"] == row["expected_reply"]["order_item_id"]


def _paired_calls(trace: dict) -> list[tuple[dict, dict]]:
    calls = [
        call for message in trace["completion_messages"] for call in message.get("tool_calls", [])
    ]
    results = [message for message in trace["completion_messages"] if message["role"] == "tool"]
    return list(zip(calls, results, strict=True))


def _database_url_for_tests() -> str:
    import os

    value = os.environ.get("ORDER_RESOLUTION_TEST_DATABASE_URL")
    if not value:
        pytest.skip("ORDER_RESOLUTION_TEST_DATABASE_URL is required")
    return value


def test_olist_calibration_retains_only_aggregates(tmp_path: Path) -> None:
    source = tmp_path / "olist_items.csv"
    source.write_text(
        "order_id,price,freight_value\nsecret-order-1,10.0,2.0\nsecret-order-2,20.0,4.0\n",
        encoding="utf-8",
    )
    summary = read_olist_calibration(source)
    assert summary == {
        "source": "local Olist aggregate calibration; no rows retained",
        "rows": 2,
        "price_min": 10.0,
        "price_max": 20.0,
        "price_mean": 15.0,
        "freight_mean": 3.0,
    }
    assert "secret-order" not in json.dumps(summary)
