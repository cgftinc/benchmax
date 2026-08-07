"""Deterministic catalog, scenario, split, and oracle-trace generation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from order_resolution import benchmark_spec as spec
from order_resolution.command_codes import CommandCode
from order_resolution.grading import grade_snapshots
from order_resolution.policy import (
    DISPOSITION_BY_CODE,
    INTENTS,
    ITEM_ID_RULES,
    ItemIdRule,
    render_system_contract,
    required_missing_fields,
    validate_reply,
)
from order_resolution.schema import (
    addresses,
    customers,
    inventory,
    inventory_allocations,
    metadata,
    order_items,
    orders,
    payments,
    product_variants,
    products,
    shipments,
    support_cases,
    support_messages,
    warehouses,
    worlds,
)

DEFAULT_SEED = 20260805
TRAIN_PER_CELL = 20
EVAL_PER_CELL = 10
SYSTEM_PROMPT = (
    "you are a post-purchase support agent. inspect the order, use only the typed business "
    "tools, follow fulfillment policy, and finish with exactly one structured customer reply."
)
ACTION_FAMILIES = ("cancel_item", "change_address", "replace_variant")
OUTCOME_CLASSES = ("execute", "clarify", "deny")
CELLS = tuple((action, outcome) for action in ACTION_FAMILIES for outcome in OUTCOME_CLASSES)
SIZES = ("small", "medium", "large")
COLORS = ("blue", "green", "black", "white", "red", "yellow", "purple", "orange")
DATA_FILES = ("train.jsonl", "eval.jsonl", "oracle_traces.jsonl")
EVAL_HASH_FILE = "eval.sha256"


@dataclass(frozen=True, slots=True)
class GeneratedData:
    train: tuple[dict[str, Any], ...]
    eval: tuple[dict[str, Any], ...]
    oracle_traces: tuple[dict[str, Any], ...]
    catalog: tuple[dict[str, Any], ...]
    hashes: dict[str, str]


def build_catalog() -> tuple[dict[str, Any], ...]:
    """Build 250 synthetic products and exactly 750 linked variants."""

    catalog: list[dict[str, Any]] = []
    for product_index in range(250):
        product_id = f"product-{product_index:03d}"
        base_price = 1_500 + (product_index % 35) * 125
        color = COLORS[product_index % len(COLORS)]
        variants = []
        for variant_index, size in enumerate(SIZES):
            variants.append(
                {
                    "variant_id": f"variant-{product_index:03d}-{variant_index}",
                    "sku": f"SKU-{product_index:03d}-{variant_index}",
                    "price_minor": base_price + (500 if variant_index == 2 else 0),
                    "currency": "USD",
                    "options": {"color": color, "size": size},
                    "active": True,
                }
            )
        catalog.append(
            {
                "product_id": product_id,
                "name": f"Everyday item {product_index:03d}",
                "category": ("apparel", "home", "outdoors", "office")[product_index % 4],
                "variants": variants,
            }
        )
    return tuple(catalog)


def generate_data(seed: int = DEFAULT_SEED) -> GeneratedData:
    """Generate both leakage-separated splits and executable oracle traces."""

    catalog = build_catalog()
    catalog_hash = _sha256_json(catalog)
    randomizer = random.Random(seed)
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    global_index = 0
    for action_family, outcome_class in CELLS:
        for split, count in (("train", TRAIN_PER_CELL), ("eval", EVAL_PER_CELL)):
            for cell_index in range(count):
                scenario_seed = randomizer.randrange(1, 2**31)
                product_pool = range(0, 160) if split == "train" else range(160, 250)
                product_index = list(product_pool)[global_index % len(product_pool)]
                row = _task_row(
                    seed=seed,
                    scenario_seed=scenario_seed,
                    split=split,
                    action_family=action_family,
                    outcome_class=outcome_class,
                    cell_index=cell_index,
                    global_index=global_index,
                    product=catalog[product_index],
                    catalog_hash=catalog_hash,
                )
                (train if split == "train" else eval_rows).append(row)
                global_index += 1
    oracle_traces = [_oracle_trace(row) for row in train]
    rendered = {
        "train.jsonl": render_jsonl(train),
        "eval.jsonl": render_jsonl(eval_rows),
        "oracle_traces.jsonl": render_jsonl(oracle_traces),
    }
    return GeneratedData(
        train=tuple(train),
        eval=tuple(eval_rows),
        oracle_traces=tuple(oracle_traces),
        catalog=catalog,
        hashes={
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in rendered.items()
        },
    )


def write_data(data_dir: Path, *, seed: int = DEFAULT_SEED, force: bool = False) -> GeneratedData:
    """Write deterministic JSONL, refusing accidental replacement without force."""

    generated = generate_data(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = [data_dir / name for name in DATA_FILES if (data_dir / name).exists()]
    if existing and not force:
        check_data(data_dir, seed=seed)
        return generated
    rows_by_name = {
        "train.jsonl": generated.train,
        "eval.jsonl": generated.eval,
        "oracle_traces.jsonl": generated.oracle_traces,
    }
    for name, rows in rows_by_name.items():
        (data_dir / name).write_text(render_jsonl(rows), encoding="utf-8")
    eval_hash_path = data_dir / EVAL_HASH_FILE
    if not eval_hash_path.exists():
        eval_hash_path.write_text(generated.hashes["eval.jsonl"] + "\n", encoding="utf-8")
    return generated


def check_data(data_dir: Path, *, seed: int = DEFAULT_SEED) -> dict[str, str]:
    """Prove balance, leakage separation, stable bytes, timestamps, and oracle success."""

    generated = generate_data(seed)
    _validate_generated(generated)
    expected = {
        "train.jsonl": render_jsonl(generated.train),
        "eval.jsonl": render_jsonl(generated.eval),
        "oracle_traces.jsonl": render_jsonl(generated.oracle_traces),
    }
    for name, content in expected.items():
        path = data_dir / name
        if not path.exists():
            raise RuntimeError(f"missing generated data file: {path}")
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"generated data drift: {path}")
    frozen_eval_hash = (data_dir / EVAL_HASH_FILE).read_text(encoding="utf-8").strip()
    if frozen_eval_hash != generated.hashes["eval.jsonl"]:
        raise RuntimeError("generated eval hash differs from the frozen pre-run hash")
    return generated.hashes


def render_jsonl(rows: Any) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for row in rows
    )


def read_olist_calibration(source: Path) -> dict[str, Any]:
    """Compute non-row-level local calibration statistics without retaining source data."""

    prices: list[float] = []
    freight: list[float] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("price"):
                prices.append(float(row["price"]))
            if row.get("freight_value"):
                freight.append(float(row["freight_value"]))
    if not prices:
        raise ValueError("Olist calibration file contains no price values")
    return {
        "source": "local Olist aggregate calibration; no rows retained",
        "rows": len(prices),
        "price_min": min(prices),
        "price_max": max(prices),
        "price_mean": sum(prices) / len(prices),
        "freight_mean": sum(freight) / len(freight) if freight else None,
    }


def initial_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(row["fixture"]["initial_snapshot"])


def case_id_for_world(world_id: str) -> str:
    return f"case-{hashlib.sha256(world_id.encode()).hexdigest()[:20]}"


async def seed_database_world(
    connection: AsyncConnection,
    *,
    row: dict[str, Any],
    world_id: str,
    ttl_seconds: int,
) -> str:
    """Seed one isolated operational world from a hidden deterministic scenario."""

    ids = row["fixture"]["ids"]
    snapshot = row["fixture"]["initial_snapshot"]
    # Validate the referenced catalog rows directly; never parse a catalog index
    # out of an identifier, which would couple seeding to an ID format.
    referenced = sorted(
        {item["variant_id"] for item in snapshot["order_items"].values()}
        | set(snapshot["inventory"])
        | {
            allocation["variant_id"]
            for allocation in snapshot["allocations"].values()
        }
    )
    present = set(
        (
            await connection.scalars(
                sa.select(product_variants.c.variant_id).where(
                    product_variants.c.variant_id.in_(referenced)
                )
            )
        ).all()
    )
    missing = [variant_id for variant_id in referenced if variant_id not in present]
    if missing:
        raise RuntimeError(
            f"parent catalog is missing {len(missing)} referenced variants: {missing[0]}"
        )
    warehouse_exists = await connection.scalar(
        sa.select(warehouses.c.warehouse_id).where(warehouses.c.warehouse_id == "warehouse-main")
    )
    if warehouse_exists is None:
        raise RuntimeError("immutable parent warehouse is not initialized")
    as_of = datetime.fromisoformat(row["as_of"])
    await connection.execute(
        sa.insert(worlds).values(
            world_id=world_id,
            scenario_id=row["task_id"],
            as_of=as_of,
            next_event_seq=0,
            retain_operational_state=False,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
        )
    )
    await connection.execute(
        sa.insert(customers).values(
            world_id=world_id,
            customer_id=ids["customer_id"],
            email=f"{ids['customer_id']}@example.test",
            name="Synthetic customer",
        )
    )
    original_address_id = snapshot["orders"][ids["order_number"]]["shipping_address_id"]
    await connection.execute(
        sa.insert(addresses).values(
            world_id=world_id,
            address_id=original_address_id,
            customer_id=ids["customer_id"],
            created_event_seq=0,
            **snapshot["addresses"][original_address_id],
        )
    )
    order_id = f"order-{hashlib.sha256(world_id.encode()).hexdigest()[:20]}"
    order = snapshot["orders"][ids["order_number"]]
    await connection.execute(
        sa.insert(orders).values(
            world_id=world_id,
            order_id=order_id,
            order_number=ids["order_number"],
            customer_id=ids["customer_id"],
            shipping_address_id=original_address_id,
            status=order["status"],
            currency=order["currency"],
            created_at=datetime.fromisoformat(order["created_at"]),
        )
    )
    for item_id, item in snapshot["order_items"].items():
        await connection.execute(
            sa.insert(order_items).values(
                world_id=world_id,
                order_item_id=item_id,
                order_id=order_id,
                **item,
            )
        )
    for variant_id, stock in snapshot["inventory"].items():
        await connection.execute(
            sa.insert(inventory).values(
                world_id=world_id,
                warehouse_id="warehouse-main",
                variant_id=variant_id,
                **stock,
            )
        )
    for item_id, allocation in snapshot["allocations"].items():
        await connection.execute(
            sa.insert(inventory_allocations).values(
                world_id=world_id,
                allocation_id=f"allocation-{item_id}",
                order_item_id=item_id,
                warehouse_id="warehouse-main",
                **allocation,
            )
        )
    await connection.execute(
        sa.insert(payments).values(
            world_id=world_id,
            payment_id=f"payment-{order_id}",
            order_id=order_id,
            status="captured",
            captured_minor=snapshot["payment"]["captured_minor"],
            currency="USD",
        )
    )
    shipment = snapshot["shipment"]
    await connection.execute(
        sa.insert(shipments).values(
            world_id=world_id,
            shipment_id=f"shipment-{order_id}",
            order_id=order_id,
            status=shipment["status"],
            carrier_handoff_at=(
                datetime.fromisoformat(shipment["carrier_handoff_at"])
                if shipment["carrier_handoff_at"]
                else None
            ),
            delivered_at=None,
        )
    )
    case_id = case_id_for_world(world_id)
    await connection.execute(
        sa.insert(support_cases).values(
            world_id=world_id,
            case_id=case_id,
            customer_id=ids["customer_id"],
            order_id=order_id,
            status="open",
            disposition=None,
            outcome_code=None,
            created_at=as_of,
        )
    )
    await connection.execute(
        sa.insert(support_messages).values(
            world_id=world_id,
            message_id=f"message-inbound-{hashlib.sha256(world_id.encode()).hexdigest()[:16]}",
            case_id=case_id,
            role="customer",
            message_kind="inbound",
            content=row["prompt_messages"][-1]["content"],
            reply_facts=None,
            event_seq=0,
            created_at=as_of,
        )
    )
    return case_id


async def seed_immutable_catalog(connection: AsyncConnection) -> None:
    """Seed the complete parent catalog before restricted child roles exist."""

    catalog = build_catalog()
    await connection.execute(
        pg_insert(products)
        .values(
            [
                {
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "category": product["category"],
                }
                for product in catalog
            ]
        )
        .on_conflict_do_nothing()
    )
    await connection.execute(
        pg_insert(product_variants)
        .values(
            [
                {"product_id": product["product_id"], **variant}
                for product in catalog
                for variant in product["variants"]
            ]
        )
        .on_conflict_do_nothing()
    )
    await connection.execute(
        pg_insert(warehouses)
        .values(warehouse_id="warehouse-main", name="Main warehouse")
        .on_conflict_do_nothing()
    )


async def delete_operational_world(connection: AsyncConnection, world_id: str) -> None:
    """Delete only disposable commerce rows while retaining result evidence."""

    for table in reversed(metadata.sorted_tables):
        if table.schema == "commerce":
            await connection.execute(sa.delete(table).where(table.c.world_id == world_id))


def oracle_after_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    before = initial_snapshot(row)
    after = copy.deepcopy(before)
    action = row["action_family"]
    outcome = row["outcome_class"]
    ids = row["fixture"]["ids"]
    if outcome == "execute" and action == "cancel_item":
        item = after["order_items"][ids["target_item_id"]]
        refund_minor = item["quantity"] * item["unit_price_minor"]
        item["status"] = "cancelled"
        after["allocations"][ids["target_item_id"]]["active"] = False
        after["inventory"][ids["old_variant_id"]]["reserved"] -= item["quantity"]
        after["orders"][ids["order_number"]]["status"] = "partially_cancelled"
        after["payment"]["refunded_minor"] += refund_minor
        after["refunds"][ids["target_item_id"]] = {
            "order_item_id": ids["target_item_id"],
            "amount_minor": refund_minor,
            "currency": "USD",
        }
    elif outcome == "execute" and action == "change_address":
        address_id = f"oracle-address-{row['task_id']}"
        after["addresses"][address_id] = row["fixture"]["requested_address"]
        after["orders"][ids["order_number"]]["shipping_address_id"] = address_id
        after["shipping_address"] = row["fixture"]["requested_address"]
    elif outcome == "execute" and action == "replace_variant":
        item = after["order_items"][ids["target_item_id"]]
        old_variant_id = item["variant_id"]
        item["variant_id"] = ids["new_variant_id"]
        after["allocations"][ids["target_item_id"]] = {
            "variant_id": ids["new_variant_id"],
            "quantity": item["quantity"],
            "active": True,
        }
        after["inventory"][old_variant_id]["reserved"] -= item["quantity"]
        after["inventory"][ids["new_variant_id"]]["reserved"] += item["quantity"]
    after["support_case"] = {
        "disposition": row["expected_disposition"],
        "outcome_code": row["expected_reply"]["outcome_code"],
    }
    after["reply"] = copy.deepcopy(row["expected_reply"])
    after["reply_count"] = 1
    return after


def _task_row(
    *,
    seed: int,
    scenario_seed: int,
    split: str,
    action_family: str,
    outcome_class: str,
    cell_index: int,
    global_index: int,
    product: dict[str, Any],
    catalog_hash: str,
) -> dict[str, Any]:
    cell = f"{action_family}-{outcome_class}"
    task_id = f"{split}-{cell}-{cell_index:02d}"
    scenario_family_id = f"family-{split}-{cell}-{cell_index:02d}"
    prompt_template_id = f"template-{split}-{cell}-{cell_index % 4}"
    as_of = datetime(2026, 8, 5, 12, tzinfo=UTC) + timedelta(
        minutes=global_index, seconds=scenario_seed % 60
    )
    customer_id = f"customer-{split}-{global_index:03d}"
    order_number = f"OR-{split[0].upper()}{global_index:05d}"
    target_item_id = f"item-{split}-{global_index:03d}-a"
    distractor_item_id = f"item-{split}-{global_index:03d}-b"
    old_variant, same_price_variant, premium_variant = product["variants"]
    deny_variant_id = (
        same_price_variant["variant_id"] if cell_index % 2 == 0 else premium_variant["variant_id"]
    )
    ids = {
        "customer_id": customer_id,
        "order_number": order_number,
        "target_item_id": target_item_id,
        "distractor_item_id": distractor_item_id,
        "product_id": product["product_id"],
        "old_variant_id": old_variant["variant_id"],
        "new_variant_id": (
            deny_variant_id
            if action_family == "replace_variant" and outcome_class == "deny"
            else same_price_variant["variant_id"]
        ),
    }
    requested_address = {
        "line1": f"{100 + scenario_seed % 800} Market St",
        "line2": None,
        "city": "Oakland",
        "region": "CA",
        "postal_code": f"{94600 + global_index % 50:05d}",
        "country": "US",
    }
    initial = _initial_state(
        ids=ids,
        old_variant=old_variant,
        new_variant=same_price_variant,
        premium_variant=premium_variant,
        as_of=as_of,
        handed_off=outcome_class == "deny" and action_family in {"cancel_item", "change_address"},
        replacement_outcome=outcome_class if action_family == "replace_variant" else None,
        replacement_deny_kind=("stock" if cell_index % 2 == 0 else "price"),
    )
    expected_disposition, outcome_code, missing_fields = _reply_facts(
        action_family, outcome_class, cell_index
    )
    expected_reply: dict[str, Any] = {
        "disposition": expected_disposition,
        "outcome_code": outcome_code,
        "order_number": order_number,
        "order_item_id": (
            target_item_id
            if action_family != "change_address" and outcome_class != "clarify"
            else None
        ),
        "missing_fields": sorted(missing_fields),
    }
    row = {
        "task_id": task_id,
        "scenario_family_id": scenario_family_id,
        "prompt_template_id": prompt_template_id,
        "generation_seed": seed,
        "scenario_seed": scenario_seed,
        "catalog_hash": catalog_hash,
        "split": split,
        "cell": cell,
        "action_family": action_family,
        "outcome_class": outcome_class,
        "as_of": as_of.isoformat(),
        "prompt_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _customer_message(
                    split=split,
                    scenario_seed=scenario_seed,
                    action_family=action_family,
                    outcome_class=outcome_class,
                    order_number=order_number,
                    product_name=product["name"],
                    requested_address=requested_address,
                ),
            },
        ],
        "expected_disposition": expected_disposition,
        "expected_reply": expected_reply,
        "fixture": {
            "ids": ids,
            "requested_address": requested_address,
            "expires_at": (as_of + timedelta(days=1)).isoformat(),
            "initial_snapshot": initial,
        },
    }
    row["required_state"] = _required_assertions(row)
    row["forbidden_state"] = _forbidden_assertions(row)
    return row


def _initial_state(
    *,
    ids: dict[str, str],
    old_variant: dict[str, Any],
    new_variant: dict[str, Any],
    premium_variant: dict[str, Any],
    as_of: datetime,
    handed_off: bool,
    replacement_outcome: str | None,
    replacement_deny_kind: str,
) -> dict[str, Any]:
    new_on_hand = 0 if replacement_outcome == "deny" and replacement_deny_kind == "stock" else 8
    return {
        "orders": {
            ids["order_number"]: {
                "status": "processing",
                "shipping_address_id": f"address-{ids['customer_id']}",
                "currency": "USD",
                "created_at": (as_of - timedelta(days=2)).isoformat(),
            }
        },
        "order_items": {
            ids["target_item_id"]: {
                "variant_id": old_variant["variant_id"],
                "quantity": 1,
                "unit_price_minor": old_variant["price_minor"],
                "status": "unfulfilled",
            },
            ids["distractor_item_id"]: {
                "variant_id": old_variant["variant_id"],
                "quantity": 1,
                "unit_price_minor": old_variant["price_minor"],
                "status": "unfulfilled",
            },
        },
        "addresses": {
            f"address-{ids['customer_id']}": {
                "line1": "1 Original Ave",
                "line2": None,
                "city": "Oakland",
                "region": "CA",
                "postal_code": "94612",
                "country": "US",
            }
        },
        "shipping_address": {
            "line1": "1 Original Ave",
            "line2": None,
            "city": "Oakland",
            "region": "CA",
            "postal_code": "94612",
            "country": "US",
        },
        "inventory": {
            old_variant["variant_id"]: {"on_hand": 20, "reserved": 2},
            new_variant["variant_id"]: {"on_hand": new_on_hand, "reserved": 0},
            premium_variant["variant_id"]: {"on_hand": 8, "reserved": 0},
        },
        "allocations": {
            ids["target_item_id"]: {
                "variant_id": old_variant["variant_id"],
                "quantity": 1,
                "active": True,
            },
            ids["distractor_item_id"]: {
                "variant_id": old_variant["variant_id"],
                "quantity": 1,
                "active": True,
            },
        },
        "payment": {"captured_minor": old_variant["price_minor"] * 2, "refunded_minor": 0},
        "shipment": {
            "status": "handed_off" if handed_off else "pending",
            "carrier_handoff_at": (
                (as_of - timedelta(minutes=5)).isoformat() if handed_off else None
            ),
        },
        "refunds": {},
        "support_case": {"disposition": None, "outcome_code": None},
        "reply": None,
        "reply_count": 0,
    }


def _reply_facts(action: str, outcome: str, cell_index: int) -> tuple[str, str, list[str]]:
    if outcome == "execute":
        return (
            "completed",
            {
                "cancel_item": "ITEM_CANCELLED",
                "change_address": "SHIPPING_ADDRESS_CHANGED",
                "replace_variant": "ITEM_VARIANT_REPLACED",
            }[action],
            [],
        )
    if outcome == "clarify":
        return (
            "needs_information",
            {
                "cancel_item": "NEEDS_ORDER_ITEM",
                "change_address": "NEEDS_POSTAL_CODE",
                "replace_variant": "NEEDS_VARIANT_OPTIONS",
            }[action],
            {
                "cancel_item": ["order_item_id"],
                "change_address": ["postal_code"],
                "replace_variant": ["requested_options"],
            }[action],
        )
    if action in {"cancel_item", "change_address"}:
        return "cannot_complete", "ALREADY_HANDED_TO_CARRIER", []
    return (
        "cannot_complete",
        "VARIANT_OUT_OF_STOCK" if cell_index % 2 == 0 else "PRICE_OR_PRODUCT_MISMATCH",
        [],
    )


def _customer_message(
    *,
    split: str,
    scenario_seed: int,
    action_family: str,
    outcome_class: str,
    order_number: str,
    product_name: str,
    requested_address: dict[str, Any],
) -> str:
    lead = "Please" if split == "train" else "Could you"
    if action_family == "cancel_item":
        item = "the item" if outcome_class == "clarify" else f"{product_name}"
        message = f"{lead} cancel {item} on order {order_number}?"
    elif action_family == "change_address":
        if outcome_class == "clarify":
            message = (
                f"{lead} send order {order_number} to {requested_address['line1']}, Oakland, CA?"
            )
        else:
            message = (
                f"{lead} change order {order_number} to {requested_address['line1']}, "
                f"{requested_address['city']}, {requested_address['region']} "
                f"{requested_address['postal_code']}, {requested_address['country']}."
            )
    else:
        option = "another size" if outcome_class == "clarify" else "medium"
        message = (
            f"{lead} replace {product_name} on order {order_number} with the {option} variant?"
        )
    return f"{message} Support reference {scenario_seed % 10_000:04d}."


def _required_assertions(row: dict[str, Any]) -> list[dict[str, Any]]:
    ids = row["fixture"]["ids"]
    assertions: list[dict[str, Any]] = [
        {
            "path": "support_case.disposition",
            "op": "eq",
            "value": row["expected_disposition"],
        },
        {
            "path": "support_case.outcome_code",
            "op": "eq",
            "value": row["expected_reply"]["outcome_code"],
        },
        {"path": "reply", "op": "eq", "value": row["expected_reply"]},
    ]
    if row["outcome_class"] != "execute":
        return assertions
    if row["action_family"] == "cancel_item":
        item = row["fixture"]["initial_snapshot"]["order_items"][ids["target_item_id"]]
        assertions.extend(
            [
                {
                    "path": f"order_items.{ids['target_item_id']}.status",
                    "op": "eq",
                    "value": "cancelled",
                },
                {
                    "path": f"inventory.{ids['old_variant_id']}.reserved",
                    "op": "eq",
                    "value": 1,
                },
                {
                    "path": f"refunds.{ids['target_item_id']}.amount_minor",
                    "op": "eq",
                    "value": item["quantity"] * item["unit_price_minor"],
                },
                {
                    "path": "payment.refunded_minor",
                    "op": "eq",
                    "value": item["quantity"] * item["unit_price_minor"],
                },
            ]
        )
    elif row["action_family"] == "change_address":
        assertions.append(
            {
                "path": "shipping_address",
                "op": "eq",
                "value": row["fixture"]["requested_address"],
            }
        )
    else:
        assertions.extend(
            [
                {
                    "path": f"order_items.{ids['target_item_id']}.variant_id",
                    "op": "eq",
                    "value": ids["new_variant_id"],
                },
                {
                    "path": f"inventory.{ids['new_variant_id']}.reserved",
                    "op": "eq",
                    "value": 1,
                },
            ]
        )
    return assertions


def _forbidden_assertions(row: dict[str, Any]) -> list[dict[str, Any]]:
    ids = row["fixture"]["ids"]
    if row["outcome_class"] != "execute":
        paths = (
            "orders",
            "order_items",
            "addresses",
            "shipping_address",
            "inventory",
            "allocations",
            "refunds",
        )
    else:
        paths = (
            f"order_items.{ids['distractor_item_id']}",
            "payment",
            "shipment",
        )
        if row["action_family"] == "cancel_item":
            paths = (
                f"order_items.{ids['distractor_item_id']}",
                "payment.captured_minor",
                "shipment",
            )
        elif row["action_family"] == "change_address":
            paths += ("order_items", "inventory", "allocations", "refunds")
        elif row["action_family"] == "replace_variant":
            paths += ("refunds", "addresses")
    return [{"path": path, "op": "unchanged"} for path in paths]


def _oracle_trace(row: dict[str, Any]) -> dict[str, Any]:
    tool_calls = _oracle_tool_calls(row)
    completion_messages: list[dict[str, Any]] = []
    for index, (name, arguments, result) in enumerate(tool_calls):
        call_id = f"call-{row['task_id']}-{index:02d}"
        completion_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments, sort_keys=True, separators=(",", ":")
                            ),
                        },
                    }
                ],
            }
        )
        completion_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, sort_keys=True, separators=(",", ":")),
            }
        )
    after = oracle_after_snapshot(row)
    grade = grade_snapshots(
        before=initial_snapshot(row),
        after=after,
        required=row["required_state"],
        forbidden=row["forbidden_state"],
        expected_disposition=row["expected_disposition"],
        expected_reply=row["expected_reply"],
    )
    return {
        "task_id": row["task_id"],
        "prompt_messages": row["prompt_messages"],
        "completion_messages": completion_messages,
        "final_snapshot_sha256": _sha256_json(after),
        "reward": grade.task_success,
    }


def _oracle_tool_calls(
    row: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    ids = row["fixture"]["ids"]
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "get_order",
            {"order_number": ids["order_number"]},
            {"ok": True, "order_number": ids["order_number"], "items": 2},
        )
    ]
    if row["outcome_class"] != "clarify":
        command = {
            "cancel_item": "cancel_order_item",
            "change_address": "change_shipping_address",
            "replace_variant": "replace_order_item_variant",
        }[row["action_family"]]
        arguments: dict[str, Any] = {"order_number": ids["order_number"]}
        if row["action_family"] != "change_address":
            arguments["order_item_id"] = ids["target_item_id"]
        if row["action_family"] == "cancel_item":
            arguments["reason"] = "customer request"
        elif row["action_family"] == "change_address":
            arguments["address"] = row["fixture"]["requested_address"]
        else:
            arguments.update(
                new_variant_id=ids["new_variant_id"], reason="customer requested variant"
            )
        result = {
            "ok": row["outcome_class"] == "execute",
            "code": row["expected_reply"]["outcome_code"],
        }
        calls.append((command, arguments, result))
    calls.append(
        (
            "reply_to_customer",
            row["expected_reply"],
            {"ok": True, "terminal": True, "rendered": "canonical customer reply"},
        )
    )
    return calls


def _validate_generated(generated: GeneratedData) -> None:
    if len(generated.train) != 180 or len(generated.eval) != 90:
        raise RuntimeError("generated split counts are not 180 train / 90 eval")
    for rows, expected in ((generated.train, TRAIN_PER_CELL), (generated.eval, EVAL_PER_CELL)):
        counts = Counter(row["cell"] for row in rows)
        if set(counts) != {f"{action}-{outcome}" for action, outcome in CELLS}:
            raise RuntimeError("generated task grid is incomplete")
        if set(counts.values()) != {expected}:
            raise RuntimeError("generated task grid is unbalanced")

    for key in ("scenario_family_id", "prompt_template_id"):
        if {row[key] for row in generated.train} & {row[key] for row in generated.eval}:
            raise RuntimeError(f"{key} overlaps train and eval")
    train_products = {row["fixture"]["ids"]["product_id"] for row in generated.train}
    eval_products = {row["fixture"]["ids"]["product_id"] for row in generated.eval}
    if train_products & eval_products:
        raise RuntimeError("product identities overlap train and eval")
    train_customers = {row["fixture"]["ids"]["customer_id"] for row in generated.train}
    eval_customers = {row["fixture"]["ids"]["customer_id"] for row in generated.eval}
    if train_customers & eval_customers:
        raise RuntimeError("customer identities overlap train and eval")

    for row in (*generated.train, *generated.eval):
        timestamp = datetime.fromisoformat(row["as_of"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise RuntimeError(f"task {row['task_id']} has a naive as_of timestamp")
        created_at = datetime.fromisoformat(
            row["fixture"]["initial_snapshot"]["orders"][row["fixture"]["ids"]["order_number"]][
                "created_at"
            ]
        )
        if created_at >= timestamp:
            raise RuntimeError(f"task {row['task_id']} has an invalid order timestamp")
        expires_at = datetime.fromisoformat(row["fixture"]["expires_at"])
        if expires_at <= timestamp:
            raise RuntimeError(f"task {row['task_id']} has an invalid expiration timestamp")
        handoff_at = row["fixture"]["initial_snapshot"]["shipment"]["carrier_handoff_at"]
        if handoff_at is not None and datetime.fromisoformat(handoff_at) > timestamp:
            raise RuntimeError(f"task {row['task_id']} has a future carrier handoff")
    trace_by_task = {trace["task_id"]: trace for trace in generated.oracle_traces}
    if set(trace_by_task) != {row["task_id"] for row in generated.train}:
        raise RuntimeError("oracle traces do not cover exactly the training split")
    if any(trace["reward"] != 1.0 for trace in generated.oracle_traces):
        raise RuntimeError("an oracle trace failed the exact grader")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# order-resolution-v2 generation
#
# v2 keeps the row schema and the shared world/snapshot mechanics above, and
# replaces the parts the audit found defective: identical order lines, split
# markers in visible identifiers, one renderer wearing two template labels,
# multi-fact clarifications, and synthetic oracle tool results.
# ---------------------------------------------------------------------------

V2_DATA_FILES = ("train.jsonl", "eval.jsonl")
CITIES = (
    ("Oakland", "CA", 94_600),
    ("Boulder", "CO", 80_300),
    ("Athens", "GA", 30_600),
    ("Salem", "OR", 97_300),
    ("Bangor", "ME", 4_400),
    ("Dover", "DE", 19_900),
    ("Tempe", "AZ", 85_200),
    ("Provo", "UT", 84_600),
)
STREETS = ("Market St", "Larkspur Ln", "Foundry Rd", "Cypress Way", "Bellevue Ave", "Kiln Ct")


class StaleCatalogError(RuntimeError):
    """The content-addressed catalog namespace disagrees with the frozen catalog."""


@dataclass(frozen=True, slots=True)
class V2Catalog:
    """A content-addressed catalog namespace and its two frozen digests."""

    products: tuple[dict[str, Any], ...]
    generation_key_sha256: str
    content_sha256: str

    @property
    def id_prefix(self) -> str:
        return self.generation_key_sha256[:12]

    def product_ids(self) -> tuple[str, ...]:
        return tuple(product["product_id"] for product in self.products)

    def variants(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"product_id": product["product_id"], **variant}
            for product in self.products
            for variant in product["variants"]
        )


@dataclass(frozen=True, slots=True)
class V2Data:
    """Deterministic v2 rows and the digests recorded in the frozen spec."""

    train: tuple[dict[str, Any], ...]
    eval: tuple[dict[str, Any], ...]
    catalog: V2Catalog
    hashes: dict[str, str]


def _catalog_contents() -> list[dict[str, Any]]:
    """Non-ID product/variant content; the generation key is derived from this."""

    contents: list[dict[str, Any]] = []
    for index in range(spec.CATALOG_PRODUCTS):
        adjective = spec.CATALOG_ADJECTIVES[index // len(spec.CATALOG_NOUNS)]
        noun = spec.CATALOG_NOUNS[index % len(spec.CATALOG_NOUNS)]
        base_price = 1_500 + (index % 35) * 125
        contents.append(
            {
                "name": f"{adjective} {noun}",
                "category": spec.CATALOG_CATEGORIES[index % len(spec.CATALOG_CATEGORIES)],
                "variants": [
                    {
                        "price_minor": base_price + (500 if size == "large" else 0),
                        "currency": "USD",
                        "options": {"size": size},
                        "active": True,
                    }
                    for size in spec.CATALOG_SIZES
                ],
            }
        )
    return contents


def build_v2_catalog() -> V2Catalog:
    """Build the v2 catalog namespace without a digest/identifier cycle.

    The generation key covers the generator version, seed, cardinalities, and
    canonical non-ID contents. Identifiers embed a prefix of that key, and the
    separate content digest is then taken over the complete rendered rows.
    """

    contents = _catalog_contents()
    generation_key = _sha256_json(
        {
            "generator_version": spec.CATALOG_GENERATOR_VERSION,
            "seed": spec.GENERATION_SEED,
            "products": spec.CATALOG_PRODUCTS,
            "variants_per_product": len(spec.CATALOG_SIZES),
            "contents": contents,
        }
    )
    prefix = generation_key[:12]
    products: list[dict[str, Any]] = []
    for index, content in enumerate(contents):
        product_id = f"p{prefix}-{index:03d}"
        products.append(
            {
                "product_id": product_id,
                "name": content["name"],
                "category": content["category"],
                "variants": [
                    {
                        "variant_id": f"v{prefix}-{index:03d}-{position}",
                        "sku": f"SKU-{prefix[:6].upper()}-{index:03d}{position}",
                        **variant,
                    }
                    for position, variant in enumerate(content["variants"])
                ],
            }
        )
    rendered = tuple(products)
    return V2Catalog(
        products=rendered,
        generation_key_sha256=generation_key,
        content_sha256=_sha256_json(rendered),
    )


def generate_v2_data() -> V2Data:
    """Generate both v2 splits deterministically from the frozen spec."""

    catalog = build_v2_catalog()
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for split, rows, per_cell, product_range in (
        ("train", train, spec.TRAIN_ROWS_PER_CELL, spec.TRAIN_PRODUCT_RANGE),
        ("eval", eval_rows, spec.EVAL_ROWS_PER_CELL, spec.EVAL_PRODUCT_RANGE),
    ):
        pool = list(range(*product_range))
        position = 0
        for cell in spec.CELLS:
            action_family, outcome_class = cell.rsplit("-", 1)
            for cell_index in range(per_cell):
                target = pool[position % len(pool)]
                distractor = pool[(position + spec.DISTRACTOR_OFFSET) % len(pool)]
                rows.append(
                    _v2_task_row(
                        catalog=catalog,
                        split=split,
                        action_family=action_family,
                        outcome_class=outcome_class,
                        cell_index=cell_index,
                        position=position,
                        target_product=catalog.products[target],
                        distractor_product=catalog.products[distractor],
                    )
                )
                position += 1
    hashes = {
        "train.jsonl": hashlib.sha256(render_jsonl(train).encode()).hexdigest(),
        "eval.jsonl": hashlib.sha256(render_jsonl(eval_rows).encode()).hexdigest(),
        "catalog_generation_key": catalog.generation_key_sha256,
        "catalog_content": catalog.content_sha256,
    }
    return V2Data(
        train=tuple(train), eval=tuple(eval_rows), catalog=catalog, hashes=dict(hashes)
    )


def _opaque(kind: str, *parts: str, length: int = 12) -> str:
    """A deterministic visible identifier that carries no split or version marker."""

    payload = "|".join((spec.BENCHMARK_ID, kind, *parts))
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def _v2_task_row(
    *,
    catalog: V2Catalog,
    split: str,
    action_family: str,
    outcome_class: str,
    cell_index: int,
    position: int,
    target_product: dict[str, Any],
    distractor_product: dict[str, Any],
) -> dict[str, Any]:
    cell = f"{action_family}-{outcome_class}"
    task_id = f"{split}-{cell}-{cell_index:02d}"
    template, shape, stratum = spec.prompt_template(split, action_family, outcome_class, cell_index)
    as_of = datetime(2026, 8, 6, 12, tzinfo=UTC) + timedelta(minutes=position)

    ordered_variant, same_price_variant, premium_variant = target_product["variants"]
    distractor_variant = distractor_product["variants"][0]
    deny_kind = "stock" if cell_index % 2 == 0 else "price"
    requested_size, requested_variant = _v2_replacement_request(
        outcome_class=outcome_class,
        deny_kind=deny_kind,
        same_price_variant=same_price_variant,
        premium_variant=premium_variant,
    )

    ids = {
        "customer_id": f"customer-{_opaque('customer', task_id)}",
        "order_number": f"OR-{_opaque('order', task_id, length=8).upper()}",
        "target_item_id": f"item-{_opaque('target-item', task_id)}",
        "distractor_item_id": f"item-{_opaque('distractor-item', task_id)}",
        "address_id": f"address-{_opaque('address', task_id)}",
        "target_product_id": target_product["product_id"],
        "distractor_product_id": distractor_product["product_id"],
        "ordered_variant_id": ordered_variant["variant_id"],
        "distractor_variant_id": distractor_variant["variant_id"],
        "requested_variant_id": (
            requested_variant["variant_id"] if requested_variant is not None else None
        ),
    }
    city, region, postal_base = CITIES[position % len(CITIES)]
    requested_address = {
        "line1": f"{100 + position % 800} {STREETS[position % len(STREETS)]}",
        "line2": None,
        "city": city,
        "region": region,
        "postal_code": f"{postal_base + position % 50:05d}",
        "country": "US",
    }
    initial = _v2_initial_state(
        ids=ids,
        as_of=as_of,
        ordered_variant=ordered_variant,
        same_price_variant=same_price_variant,
        premium_variant=premium_variant,
        distractor_variant=distractor_variant,
        handed_off=outcome_class == "deny" and action_family in {"cancel_item", "change_address"},
        replacement_out_of_stock=(
            action_family == "replace_variant" and outcome_class == "deny" and deny_kind == "stock"
        ),
    )
    expected_reply = _v2_expected_reply(
        action_family=action_family,
        outcome_class=outcome_class,
        deny_kind=deny_kind,
        order_number=ids["order_number"],
        target_item_id=ids["target_item_id"],
    )
    row: dict[str, Any] = {
        "benchmark_id": spec.BENCHMARK_ID,
        "task_id": task_id,
        "scenario_family_id": f"family-{split}-{cell}-{cell_index:02d}",
        "prompt_template_id": f"template-{split}-{shape}-{stratum}",
        "prompt_shape": shape,
        "prompt_stratum": stratum,
        "generation_seed": spec.GENERATION_SEED,
        "catalog_generation_key_sha256": catalog.generation_key_sha256,
        "catalog_content_sha256": catalog.content_sha256,
        "split": split,
        "cell": cell,
        "action_family": action_family,
        "outcome_class": outcome_class,
        "as_of": as_of.isoformat(),
        "prompt_messages": [
            {"role": "system", "content": render_system_contract()},
            {
                "role": "user",
                "content": template.format(
                    order_number=ids["order_number"],
                    product=target_product["name"],
                    size=requested_size or "",
                    **{key: value for key, value in requested_address.items() if key != "line2"},
                ),
            },
        ],
        "expected_disposition": expected_reply["disposition"],
        "expected_reply": expected_reply,
        "fixture": {
            "ids": ids,
            "target_product_name": target_product["name"],
            "distractor_product_name": distractor_product["name"],
            "requested_address": requested_address,
            "requested_size": requested_size,
            "expires_at": (as_of + timedelta(days=1)).isoformat(),
            "initial_snapshot": initial,
        },
    }
    row["required_state"] = _v2_required_assertions(row)
    row["forbidden_state"] = _forbidden_assertions(row)
    return row


def _v2_replacement_request(
    *,
    outcome_class: str,
    deny_kind: str,
    same_price_variant: dict[str, Any],
    premium_variant: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    """Which size the customer names, and the variant it resolves to."""

    if outcome_class == "clarify":
        return None, None
    if outcome_class == "deny" and deny_kind == "price":
        return str(premium_variant["options"]["size"]), premium_variant
    return str(same_price_variant["options"]["size"]), same_price_variant


def _v2_expected_reply(
    *,
    action_family: str,
    outcome_class: str,
    deny_kind: str,
    order_number: str,
    target_item_id: str,
) -> dict[str, Any]:
    intent = {
        "cancel_item": INTENTS[0],
        "change_address": INTENTS[1],
        "replace_variant": INTENTS[2],
    }[action_family]
    if outcome_class == "execute":
        outcome_code = str(intent.completed)
    elif outcome_class == "clarify":
        outcome_code = str(intent.needs_information)
    elif action_family == "replace_variant":
        outcome_code = str(
            CommandCode.VARIANT_OUT_OF_STOCK
            if deny_kind == "stock"
            else CommandCode.PRICE_OR_PRODUCT_MISMATCH
        )
    else:
        outcome_code = str(CommandCode.ALREADY_HANDED_TO_CARRIER)

    missing_fields = list(required_missing_fields(outcome_code))
    item_rule = ITEM_ID_RULES[outcome_code]
    if item_rule is ItemIdRule.FORBIDDEN:
        order_item_id = None
    elif item_rule is ItemIdRule.REQUIRED:
        order_item_id = target_item_id
    else:
        # ALREADY_HANDED_TO_CARRIER denies an item request and a whole-order one.
        order_item_id = target_item_id if action_family != "change_address" else None
    reply = {
        "disposition": str(DISPOSITION_BY_CODE[outcome_code]),
        "outcome_code": outcome_code,
        "order_number": order_number,
        "order_item_id": order_item_id,
        "missing_fields": sorted(missing_fields),
    }
    validate_reply(
        disposition=reply["disposition"],
        outcome_code=reply["outcome_code"],
        order_item_id=reply["order_item_id"],
        missing_fields=reply["missing_fields"],
    )
    return reply


def _v2_initial_state(
    *,
    ids: dict[str, Any],
    as_of: datetime,
    ordered_variant: dict[str, Any],
    same_price_variant: dict[str, Any],
    premium_variant: dict[str, Any],
    distractor_variant: dict[str, Any],
    handed_off: bool,
    replacement_out_of_stock: bool,
) -> dict[str, Any]:
    """Two visibly different order lines over one shared, observable world."""

    address = {
        "line1": "1 Original Ave",
        "line2": None,
        "city": "Fresno",
        "region": "CA",
        "postal_code": "93701",
        "country": "US",
    }
    captured = ordered_variant["price_minor"] + distractor_variant["price_minor"]
    return {
        "orders": {
            ids["order_number"]: {
                "status": "processing",
                "shipping_address_id": ids["address_id"],
                "currency": "USD",
                "created_at": (as_of - timedelta(days=2)).isoformat(),
            }
        },
        "order_items": {
            ids["target_item_id"]: {
                "variant_id": ordered_variant["variant_id"],
                "quantity": 1,
                "unit_price_minor": ordered_variant["price_minor"],
                "status": "unfulfilled",
            },
            ids["distractor_item_id"]: {
                "variant_id": distractor_variant["variant_id"],
                "quantity": 1,
                "unit_price_minor": distractor_variant["price_minor"],
                "status": "unfulfilled",
            },
        },
        "addresses": {ids["address_id"]: address},
        "shipping_address": dict(address),
        "inventory": {
            ordered_variant["variant_id"]: {"on_hand": 20, "reserved": 1},
            same_price_variant["variant_id"]: {
                "on_hand": 0 if replacement_out_of_stock else 8,
                "reserved": 0,
            },
            premium_variant["variant_id"]: {"on_hand": 8, "reserved": 0},
            distractor_variant["variant_id"]: {"on_hand": 20, "reserved": 1},
        },
        "allocations": {
            ids["target_item_id"]: {
                "variant_id": ordered_variant["variant_id"],
                "quantity": 1,
                "active": True,
            },
            ids["distractor_item_id"]: {
                "variant_id": distractor_variant["variant_id"],
                "quantity": 1,
                "active": True,
            },
        },
        "payment": {"captured_minor": captured, "refunded_minor": 0},
        "shipment": {
            "status": "handed_off" if handed_off else "pending",
            "carrier_handoff_at": (
                (as_of - timedelta(minutes=5)).isoformat() if handed_off else None
            ),
        },
        "refunds": {},
        "support_case": {"disposition": None, "outcome_code": None},
        "reply": None,
        "reply_count": 0,
    }


def _v2_required_assertions(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive terminal-state assertions from this row's own initial snapshot."""

    ids = row["fixture"]["ids"]
    snapshot = row["fixture"]["initial_snapshot"]
    assertions: list[dict[str, Any]] = [
        {"path": "support_case.disposition", "op": "eq", "value": row["expected_disposition"]},
        {
            "path": "support_case.outcome_code",
            "op": "eq",
            "value": row["expected_reply"]["outcome_code"],
        },
        {"path": "reply", "op": "eq", "value": row["expected_reply"]},
    ]
    if row["outcome_class"] != "execute":
        return assertions

    item = snapshot["order_items"][ids["target_item_id"]]
    ordered_variant_id = ids["ordered_variant_id"]
    if row["action_family"] == "cancel_item":
        refund_minor = item["quantity"] * item["unit_price_minor"]
        reserved_after = snapshot["inventory"][ordered_variant_id]["reserved"] - item["quantity"]
        assertions.extend(
            [
                {
                    "path": f"order_items.{ids['target_item_id']}.status",
                    "op": "eq",
                    "value": "cancelled",
                },
                {
                    "path": f"inventory.{ordered_variant_id}.reserved",
                    "op": "eq",
                    "value": reserved_after,
                },
                {
                    "path": f"refunds.{ids['target_item_id']}.amount_minor",
                    "op": "eq",
                    "value": refund_minor,
                },
                {"path": "payment.refunded_minor", "op": "eq", "value": refund_minor},
                {
                    "path": f"orders.{ids['order_number']}.status",
                    "op": "eq",
                    "value": "partially_cancelled",
                },
            ]
        )
    elif row["action_family"] == "change_address":
        assertions.append(
            {"path": "shipping_address", "op": "eq", "value": row["fixture"]["requested_address"]}
        )
    else:
        requested_variant_id = ids["requested_variant_id"]
        assertions.extend(
            [
                {
                    "path": f"order_items.{ids['target_item_id']}.variant_id",
                    "op": "eq",
                    "value": requested_variant_id,
                },
                {
                    "path": f"inventory.{requested_variant_id}.reserved",
                    "op": "eq",
                    "value": snapshot["inventory"][requested_variant_id]["reserved"]
                    + item["quantity"],
                },
                {
                    "path": f"inventory.{ordered_variant_id}.reserved",
                    "op": "eq",
                    "value": snapshot["inventory"][ordered_variant_id]["reserved"]
                    - item["quantity"],
                },
            ]
        )
    return assertions


def write_v2_data(data_dir: Path, *, force: bool = False) -> V2Data:
    """Write the v2 splits, refusing accidental replacement without force."""

    generated = generate_v2_data()
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = [data_dir / name for name in V2_DATA_FILES if (data_dir / name).exists()]
    if existing and not force:
        _check_v2_rows(data_dir, generated)
        return generated
    (data_dir / "train.jsonl").write_text(render_jsonl(generated.train), encoding="utf-8")
    (data_dir / "eval.jsonl").write_text(render_jsonl(generated.eval), encoding="utf-8")
    eval_hash_path = data_dir / EVAL_HASH_FILE
    if not eval_hash_path.exists():
        eval_hash_path.write_text(generated.hashes["eval.jsonl"] + "\n", encoding="utf-8")
    return generated


def check_v2_data(data_dir: Path, *, require_oracles: bool = True) -> dict[str, str]:
    """Prove byte stability, split separation, solvability, and oracle coverage."""

    generated = generate_v2_data()
    validate_v2_generated(generated)
    _check_v2_rows(data_dir, generated)
    frozen = (data_dir / EVAL_HASH_FILE).read_text(encoding="utf-8").strip()
    if frozen != generated.hashes["eval.jsonl"]:
        raise RuntimeError("generated v2 eval hash differs from the frozen pre-run hash")
    if require_oracles:
        traces = _load_oracle_traces(data_dir / "oracle_traces.jsonl")
        expected = {row["task_id"] for row in generated.train}
        if {trace["task_id"] for trace in traces} != expected:
            raise RuntimeError("v2 oracle traces do not cover exactly the training split")
        if any(trace["reward"] != 1.0 for trace in traces):
            raise RuntimeError("a v2 oracle trace failed the exact grader")
        generated.hashes["oracle_traces.jsonl"] = hashlib.sha256(
            (data_dir / "oracle_traces.jsonl").read_bytes()
        ).hexdigest()
    return generated.hashes


def _check_v2_rows(data_dir: Path, generated: V2Data) -> None:
    for name, rows in (("train.jsonl", generated.train), ("eval.jsonl", generated.eval)):
        path = data_dir / name
        if not path.exists():
            raise RuntimeError(f"missing generated v2 data file: {path}")
        if path.read_text(encoding="utf-8") != render_jsonl(rows):
            raise RuntimeError(f"generated v2 data drift: {path}")


def _load_oracle_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing compiled v2 oracle traces: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_v2_generated(generated: V2Data) -> None:
    """Prove balance, isolation, solvability, and one-fact clarification."""

    if len(generated.train) != spec.TRAIN_ROW_COUNT:
        raise RuntimeError(f"v2 train split is not {spec.TRAIN_ROW_COUNT} rows")
    if len(generated.eval) != spec.EVAL_ROW_COUNT:
        raise RuntimeError(f"v2 eval split is not {spec.EVAL_ROW_COUNT} rows")
    for rows, expected in (
        (generated.train, spec.TRAIN_ROWS_PER_CELL),
        (generated.eval, spec.EVAL_ROWS_PER_CELL),
    ):
        counts = Counter(row["cell"] for row in rows)
        if set(counts) != set(spec.CELLS) or set(counts.values()) != {expected}:
            raise RuntimeError("v2 task grid is incomplete or unbalanced")

    for key in ("scenario_family_id", "prompt_template_id"):
        if {row[key] for row in generated.train} & {row[key] for row in generated.eval}:
            raise RuntimeError(f"{key} overlaps the v2 splits")
    for path in ("target_product_id", "distractor_product_id", "customer_id"):
        train_values = {row["fixture"]["ids"][path] for row in generated.train}
        eval_values = {row["fixture"]["ids"][path] for row in generated.eval}
        if train_values & eval_values:
            raise RuntimeError(f"{path} overlaps the v2 splits")
    if _skeletons(generated.train) & _skeletons(generated.eval):
        raise RuntimeError("v2 prompt skeletons overlap the splits")
    if Counter(row["prompt_stratum"] for row in generated.train) != Counter(
        {stratum: spec.TRAIN_ROWS_PER_CELL // len(spec.PROMPT_STRATA) * len(spec.CELLS)
         for stratum in spec.PROMPT_STRATA}
    ):
        raise RuntimeError("v2 training strata are unbalanced")

    for row in (*generated.train, *generated.eval):
        _validate_v2_row(row)


def _validate_v2_row(row: dict[str, Any]) -> None:
    fixture = row["fixture"]
    ids = fixture["ids"]
    prompt = row["prompt_messages"][-1]["content"]
    lowered = prompt.lower()

    for marker in ("train", "eval", "order-resolution", "v2"):
        for value in (*ids.values(), prompt):
            if isinstance(value, str) and marker in value.lower():
                raise RuntimeError(f"task {row['task_id']} leaks {marker!r} into visible state")

    target_name = fixture["target_product_name"]
    distractor_name = fixture["distractor_product_name"]
    if target_name == distractor_name:
        raise RuntimeError(f"task {row['task_id']} has two identical order lines")
    if set(target_name.split()) & set(distractor_name.split()):
        raise RuntimeError(f"task {row['task_id']} order lines share a visible word")

    named = target_name.lower() in lowered
    if row["action_family"] == "change_address":
        # Address requests are order-level; naming a line would be noise.
        if named:
            raise RuntimeError(f"task {row['task_id']} names an item in an order-level request")
    elif row["outcome_class"] == "clarify" and row["action_family"] == "cancel_item":
        if named:
            raise RuntimeError(f"task {row['task_id']} clarification names its target product")
    elif not named:
        raise RuntimeError(f"task {row['task_id']} does not name its unique target product")
    if distractor_name.lower() in lowered:
        raise RuntimeError(f"task {row['task_id']} names the distractor product")

    address = fixture["requested_address"]
    if row["action_family"] == "change_address":
        omitted = ("postal_code",) if row["outcome_class"] == "clarify" else ()
        for field, value in address.items():
            if field == "line2" or value is None:
                continue
            present = str(value).lower() in lowered
            if field in omitted and present:
                raise RuntimeError(f"task {row['task_id']} states an omitted address fact")
            if field not in omitted and not present:
                raise RuntimeError(f"task {row['task_id']} omits required address fact {field}")

    if row["action_family"] == "replace_variant":
        size = fixture["requested_size"]
        if row["outcome_class"] == "clarify":
            if size is not None or any(
                option in lowered for option in spec.CATALOG_SIZES
            ):
                raise RuntimeError(f"task {row['task_id']} clarification states a size")
        elif size is None or size not in lowered:
            raise RuntimeError(f"task {row['task_id']} does not state its requested size")

    validate_reply(
        disposition=row["expected_reply"]["disposition"],
        outcome_code=row["expected_reply"]["outcome_code"],
        order_item_id=row["expected_reply"]["order_item_id"],
        missing_fields=row["expected_reply"]["missing_fields"],
    )
    as_of = datetime.fromisoformat(row["as_of"])
    order = fixture["initial_snapshot"]["orders"][ids["order_number"]]
    if datetime.fromisoformat(order["created_at"]) >= as_of:
        raise RuntimeError(f"task {row['task_id']} has an invalid order timestamp")
    if datetime.fromisoformat(fixture["expires_at"]) <= as_of:
        raise RuntimeError(f"task {row['task_id']} has an invalid expiration timestamp")
    handoff = fixture["initial_snapshot"]["shipment"]["carrier_handoff_at"]
    if handoff is not None and datetime.fromisoformat(handoff) > as_of:
        raise RuntimeError(f"task {row['task_id']} has a future carrier handoff")


def prompt_skeleton(text: str) -> str:
    """Normalize a rendered prompt to its wording skeleton."""

    words = [
        word
        for word in re.sub(r"[^a-z\s]", " ", text.lower()).split()
        if word and not word.isdigit()
    ]
    return hashlib.sha256(" ".join(words).encode()).hexdigest()


def _skeletons(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {prompt_skeleton(row["prompt_messages"][-1]["content"]) for row in rows}


async def sync_catalog_namespace(connection: AsyncConnection, catalog: V2Catalog) -> str:
    """Insert the v2 namespace, or require exact equality with what is stored.

    All-or-exact by design: a partially present namespace is a blocking stale
    catalog error rather than a silent merge, and a changed catalog necessarily
    receives a new namespace prefix.
    """

    prefix = catalog.id_prefix
    stored_products = (
        (
            await connection.execute(
                sa.select(products.c.product_id, products.c.name, products.c.category)
                .where(products.c.product_id.like(f"p{prefix}-%"))
                .order_by(products.c.product_id)
            )
        )
        .mappings()
        .all()
    )
    stored_variants = (
        (
            await connection.execute(
                sa.select(
                    product_variants.c.variant_id,
                    product_variants.c.product_id,
                    product_variants.c.sku,
                    product_variants.c.price_minor,
                    product_variants.c.currency,
                    product_variants.c.options,
                    product_variants.c.active,
                )
                .where(product_variants.c.variant_id.like(f"v{prefix}-%"))
                .order_by(product_variants.c.variant_id)
            )
        )
        .mappings()
        .all()
    )
    expected_products = [
        {
            "product_id": product["product_id"],
            "name": product["name"],
            "category": product["category"],
        }
        for product in catalog.products
    ]
    expected_variants = [
        {
            "variant_id": variant["variant_id"],
            "product_id": variant["product_id"],
            "sku": variant["sku"],
            "price_minor": variant["price_minor"],
            "currency": variant["currency"],
            "options": variant["options"],
            "active": variant["active"],
        }
        for variant in catalog.variants()
    ]
    if not stored_products and not stored_variants:
        await connection.execute(sa.insert(products).values(expected_products))
        await connection.execute(sa.insert(product_variants).values(expected_variants))
        await connection.execute(
            pg_insert(warehouses)
            .values(warehouse_id="warehouse-main", name="Main warehouse")
            .on_conflict_do_nothing()
        )
        return "inserted"

    actual = ([dict(row) for row in stored_products], [dict(row) for row in stored_variants])
    if actual != (expected_products, expected_variants):
        raise StaleCatalogError(
            f"catalog namespace {prefix} exists with {len(stored_products)} products and "
            f"{len(stored_variants)} variants that do not match the frozen v2 catalog"
        )
    return "verified"


__all__ = [
    "DEFAULT_SEED",
    "GeneratedData",
    "StaleCatalogError",
    "V2Catalog",
    "V2Data",
    "build_catalog",
    "build_v2_catalog",
    "check_data",
    "check_v2_data",
    "generate_data",
    "generate_v2_data",
    "initial_snapshot",
    "oracle_after_snapshot",
    "prompt_skeleton",
    "read_olist_calibration",
    "render_jsonl",
    "seed_immutable_catalog",
    "sync_catalog_namespace",
    "validate_v2_generated",
    "write_data",
    "write_v2_data",
]
