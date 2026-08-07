"""Exact required-state and forbidden-mutation evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from order_resolution.schema import (
    addresses,
    episode_results,
    inventory,
    inventory_allocations,
    order_items,
    orders,
    payments,
    refunds,
    shipments,
    support_cases,
    support_messages,
)


@dataclass(frozen=True, slots=True)
class GradeResult:
    """Deterministic assertion result shared by fixtures and live rollouts."""

    task_success: float
    required_state_fraction: float
    forbidden_mutation: float
    correct_disposition: float
    structured_reply_correct: float
    failures: tuple[str, ...]


def grade_snapshots(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    required: Sequence[Mapping[str, Any]],
    forbidden: Sequence[Mapping[str, Any]],
    expected_disposition: str,
    expected_reply: Mapping[str, Any],
) -> GradeResult:
    """Compare normalized snapshots against exact hidden task assertions."""

    failures: list[str] = []
    required_passes = 0
    for assertion in required:
        path = _path(assertion)
        if assertion.get("op") != "eq":
            raise ValueError(f"unsupported required assertion op at {path}")
        actual = _get(after, path)
        if actual == assertion.get("value"):
            required_passes += 1
        else:
            failures.append(f"required {path}: expected {assertion.get('value')!r}, got {actual!r}")

    forbidden_mutation = 0.0
    for assertion in forbidden:
        path = _path(assertion)
        if assertion.get("op") != "unchanged":
            raise ValueError(f"unsupported forbidden assertion op at {path}")
        if _get(before, path) != _get(after, path):
            forbidden_mutation = 1.0
            failures.append(f"forbidden mutation at {path}")

    actual_disposition = _get(after, "support_case.disposition")
    correct_disposition = float(actual_disposition == expected_disposition)
    if not correct_disposition:
        failures.append(
            f"disposition: expected {expected_disposition!r}, got {actual_disposition!r}"
        )
    actual_reply = _get(after, "reply")
    reply_count = _get(after, "reply_count")
    structured_reply_correct = float(actual_reply == expected_reply and reply_count == 1)
    if not structured_reply_correct:
        failures.append("structured reply did not match expected contract")

    required_fraction = required_passes / len(required) if required else 1.0
    success = float(
        required_fraction == 1.0
        and forbidden_mutation == 0.0
        and correct_disposition == 1.0
        and structured_reply_correct == 1.0
    )
    return GradeResult(
        task_success=success,
        required_state_fraction=required_fraction,
        forbidden_mutation=forbidden_mutation,
        correct_disposition=correct_disposition,
        structured_reply_correct=structured_reply_correct,
        failures=tuple(failures),
    )


async def capture_world_snapshot(connection: AsyncConnection, world_id: str) -> dict[str, Any]:
    """Read the exact normalized state consumed by the deterministic grader."""

    order_rows = (
        (
            await connection.execute(
                sa.select(
                    orders.c.order_number,
                    orders.c.status,
                    orders.c.shipping_address_id,
                    orders.c.currency,
                    orders.c.created_at,
                ).where(orders.c.world_id == world_id)
            )
        )
        .mappings()
        .all()
    )
    item_rows = (
        (
            await connection.execute(
                sa.select(
                    order_items.c.order_item_id,
                    order_items.c.variant_id,
                    order_items.c.quantity,
                    order_items.c.unit_price_minor,
                    order_items.c.status,
                ).where(order_items.c.world_id == world_id)
            )
        )
        .mappings()
        .all()
    )
    address_rows = (
        (
            await connection.execute(
                sa.select(
                    addresses.c.address_id,
                    addresses.c.line1,
                    addresses.c.line2,
                    addresses.c.city,
                    addresses.c.region,
                    addresses.c.postal_code,
                    addresses.c.country,
                ).where(addresses.c.world_id == world_id)
            )
        )
        .mappings()
        .all()
    )
    stock_rows = (
        (
            await connection.execute(
                sa.select(inventory.c.variant_id, inventory.c.on_hand, inventory.c.reserved).where(
                    inventory.c.world_id == world_id
                )
            )
        )
        .mappings()
        .all()
    )
    allocation_rows = (
        (
            await connection.execute(
                sa.select(
                    inventory_allocations.c.order_item_id,
                    inventory_allocations.c.variant_id,
                    inventory_allocations.c.quantity,
                    inventory_allocations.c.active,
                ).where(
                    inventory_allocations.c.world_id == world_id,
                    inventory_allocations.c.active.is_(True),
                )
            )
        )
        .mappings()
        .all()
    )
    refund_rows = (
        (
            await connection.execute(
                sa.select(
                    refunds.c.order_item_id,
                    refunds.c.amount_minor,
                    refunds.c.currency,
                ).where(refunds.c.world_id == world_id)
            )
        )
        .mappings()
        .all()
    )
    payment = (
        (
            await connection.execute(
                sa.select(payments.c.captured_minor).where(payments.c.world_id == world_id)
            )
        )
        .mappings()
        .one()
    )
    refunded_minor = int(
        await connection.scalar(
            sa.select(sa.func.coalesce(sa.func.sum(refunds.c.amount_minor), 0)).where(
                refunds.c.world_id == world_id
            )
        )
        or 0
    )
    shipment = (
        (
            await connection.execute(
                sa.select(shipments.c.status, shipments.c.carrier_handoff_at).where(
                    shipments.c.world_id == world_id
                )
            )
        )
        .mappings()
        .one()
    )
    support_case = (
        (
            await connection.execute(
                sa.select(support_cases.c.disposition, support_cases.c.outcome_code).where(
                    support_cases.c.world_id == world_id
                )
            )
        )
        .mappings()
        .one()
    )
    reply = await connection.scalar(
        sa.select(support_messages.c.reply_facts)
        .where(
            support_messages.c.world_id == world_id,
            support_messages.c.message_kind == "customer_reply",
        )
        .order_by(support_messages.c.event_seq.desc())
        .limit(1)
    )
    reply_count = int(
        await connection.scalar(
            sa.select(sa.func.count())
            .select_from(support_messages)
            .where(
                support_messages.c.world_id == world_id,
                support_messages.c.message_kind == "customer_reply",
            )
        )
        or 0
    )
    address_map = {
        row["address_id"]: {
            "line1": row["line1"],
            "line2": row["line2"],
            "city": row["city"],
            "region": row["region"],
            "postal_code": row["postal_code"],
            "country": row["country"],
        }
        for row in address_rows
    }
    current_address_id = order_rows[0]["shipping_address_id"]
    return {
        "orders": {
            row["order_number"]: {
                "status": row["status"],
                "shipping_address_id": row["shipping_address_id"],
                "currency": row["currency"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in order_rows
        },
        "order_items": {
            row["order_item_id"]: {
                "variant_id": row["variant_id"],
                "quantity": row["quantity"],
                "unit_price_minor": row["unit_price_minor"],
                "status": row["status"],
            }
            for row in item_rows
        },
        "addresses": address_map,
        "shipping_address": address_map[current_address_id],
        "inventory": {
            row["variant_id"]: {"on_hand": row["on_hand"], "reserved": row["reserved"]}
            for row in stock_rows
        },
        "allocations": {
            row["order_item_id"]: {
                "variant_id": row["variant_id"],
                "quantity": row["quantity"],
                "active": row["active"],
            }
            for row in allocation_rows
        },
        "payment": {
            "captured_minor": payment["captured_minor"],
            "refunded_minor": refunded_minor,
        },
        "shipment": {
            "status": shipment["status"],
            "carrier_handoff_at": (
                shipment["carrier_handoff_at"].isoformat()
                if shipment["carrier_handoff_at"]
                else None
            ),
        },
        "refunds": {
            row["order_item_id"]: {
                "order_item_id": row["order_item_id"],
                "amount_minor": row["amount_minor"],
                "currency": row["currency"],
            }
            for row in refund_rows
        },
        "support_case": dict(support_case),
        "reply": reply,
        "reply_count": reply_count,
    }


async def store_episode_result(
    connection: AsyncConnection,
    *,
    world_id: str,
    scenario_id: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    grade: GradeResult,
    diagnostics: Mapping[str, Any],
) -> None:
    await connection.execute(
        sa.insert(episode_results).values(
            world_id=world_id,
            scenario_id=scenario_id,
            before_snapshot=dict(before),
            after_snapshot=dict(after),
            reward=grade.task_success,
            diagnostics={**dict(diagnostics), "failures": list(grade.failures)},
        )
    )


def _path(assertion: Mapping[str, Any]) -> str:
    value = assertion.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("assertion path must be a non-empty string")
    return value


def _get(snapshot: Mapping[str, Any], path: str) -> Any:
    value: Any = snapshot
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


__all__ = [
    "GradeResult",
    "capture_world_snapshot",
    "grade_snapshots",
    "store_episode_result",
]
