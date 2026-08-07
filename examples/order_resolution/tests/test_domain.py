"""Atomic business-command and persistence invariant tests."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from order_resolution.command_codes import CommandCode, EnvelopeCode
from order_resolution.database import (
    AmbiguousCommitError,
    Database,
    DatabaseConfigurationError,
    validate_database_url,
)
from order_resolution.domain import (
    DomainInvariantError,
    OrderResolutionService,
    assert_world_invariants,
)
from order_resolution.schema import (
    addresses,
    audit_events,
    command_receipts,
    inventory,
    inventory_allocations,
    order_items,
    orders,
    refunds,
    support_cases,
    support_messages,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.conftest import seed_world


@pytest.mark.asyncio
async def test_cancel_is_atomic_idempotent_and_releases_inventory(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    service = OrderResolutionService(database)
    first = await service.cancel_order_item(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        reason="customer request",
    )
    second = await service.cancel_order_item(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        reason="customer request",
    )
    assert first == second
    assert first["code"] == "ITEM_CANCELLED"

    async with engine.connect() as connection:
        item_status = await connection.scalar(
            sa.select(order_items.c.status).where(
                order_items.c.world_id == ids["world_id"],
                order_items.c.order_item_id == ids["order_item_id"],
            )
        )
        order_status = await connection.scalar(
            sa.select(orders.c.status).where(
                orders.c.world_id == ids["world_id"], orders.c.order_id == ids["order_id"]
            )
        )
        reserved = await connection.scalar(
            sa.select(inventory.c.reserved).where(
                inventory.c.world_id == ids["world_id"],
                inventory.c.variant_id == ids["old_variant_id"],
            )
        )
        assert item_status == "cancelled"
        assert order_status == "cancelled"
        assert reserved == 0
        assert await _count(connection, refunds, ids["world_id"]) == 1
        assert await _count(connection, audit_events, ids["world_id"]) == 1
        assert await _count(connection, command_receipts, ids["world_id"]) == 1
        await assert_world_invariants(connection, ids["world_id"])


@pytest.mark.asyncio
async def test_policy_denials_leave_state_unchanged(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine, handed_off=True)
    service = OrderResolutionService(database)
    cancel = await service.cancel_order_item(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        reason="too late",
    )
    address = await service.change_shipping_address(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        address=_new_address(),
    )
    replace = await service.replace_order_item_variant(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        new_variant_id=ids["new_variant_id"],
        reason="different size",
    )
    assert {cancel["code"], address["code"], replace["code"]} == {"ALREADY_HANDED_TO_CARRIER"}
    async with engine.connect() as connection:
        assert await _count(connection, audit_events, ids["world_id"]) == 0
        assert await _count(connection, refunds, ids["world_id"]) == 0
        await assert_world_invariants(connection, ids["world_id"])


@pytest.mark.asyncio
async def test_address_change_creates_immutable_snapshot_and_replays(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    service = OrderResolutionService(database)
    first = await service.change_shipping_address(
        world_id=ids["world_id"], order_number=ids["order_number"], address=_new_address()
    )
    second = await service.change_shipping_address(
        world_id=ids["world_id"], order_number=ids["order_number"], address=_new_address()
    )
    assert first == second
    assert first["code"] == "SHIPPING_ADDRESS_CHANGED"
    async with engine.connect() as connection:
        address_ids = (
            await connection.scalars(
                sa.select(addresses.c.address_id)
                .where(addresses.c.world_id == ids["world_id"])
                .order_by(addresses.c.address_id)
            )
        ).all()
        current = await connection.scalar(
            sa.select(orders.c.shipping_address_id).where(
                orders.c.world_id == ids["world_id"], orders.c.order_id == ids["order_id"]
            )
        )
        assert len(address_ids) == 2
        assert ids["address_id"] in address_ids
        assert current == first["address_id"]
        await assert_world_invariants(connection, ids["world_id"])


@pytest.mark.asyncio
async def test_replacement_moves_exact_reservation_and_replays(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    service = OrderResolutionService(database)
    first = await service.replace_order_item_variant(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        new_variant_id=ids["new_variant_id"],
        reason="needs medium",
    )
    second = await service.replace_order_item_variant(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        new_variant_id=ids["new_variant_id"],
        reason="needs medium",
    )
    assert first == second
    assert first["code"] == "ITEM_VARIANT_REPLACED"
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                sa.select(inventory.c.variant_id, inventory.c.reserved).where(
                    inventory.c.world_id == ids["world_id"],
                    inventory.c.variant_id.in_([ids["old_variant_id"], ids["new_variant_id"]]),
                )
            )
        ).all()
        assert dict(rows) == {ids["old_variant_id"]: 0, ids["new_variant_id"]: 1}
        active = (
            await connection.execute(
                sa.select(
                    inventory_allocations.c.variant_id, inventory_allocations.c.quantity
                ).where(
                    inventory_allocations.c.world_id == ids["world_id"],
                    inventory_allocations.c.order_item_id == ids["order_item_id"],
                    inventory_allocations.c.active.is_(True),
                )
            )
        ).one()
        assert active == (ids["new_variant_id"], 1)
        await assert_world_invariants(connection, ids["world_id"])


@pytest.mark.asyncio
async def test_replacement_rejects_stock_and_price_without_partial_write(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    service = OrderResolutionService(database)
    out = await service.replace_order_item_variant(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        new_variant_id=ids["out_variant_id"],
        reason="large",
    )
    expensive = await service.replace_order_item_variant(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        new_variant_id=ids["expensive_variant_id"],
        reason="premium",
    )
    assert out["code"] == "VARIANT_OUT_OF_STOCK"
    assert expensive["code"] == "PRICE_OR_PRODUCT_MISMATCH"
    async with engine.connect() as connection:
        variant_id = await connection.scalar(
            sa.select(order_items.c.variant_id).where(
                order_items.c.world_id == ids["world_id"],
                order_items.c.order_item_id == ids["order_item_id"],
            )
        )
        assert variant_id == ids["old_variant_id"]
        assert await _count(connection, audit_events, ids["world_id"]) == 0
        await assert_world_invariants(connection, ids["world_id"])


@pytest.mark.asyncio
async def test_invariant_failure_rolls_back_without_receipt(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine, captured_minor=500)
    service = OrderResolutionService(database)
    with pytest.raises(DomainInvariantError, match="refund would exceed"):
        await service.cancel_order_item(
            world_id=ids["world_id"],
            order_number=ids["order_number"],
            order_item_id=ids["order_item_id"],
            reason="must roll back",
        )
    async with engine.connect() as connection:
        assert (
            await connection.scalar(
                sa.select(order_items.c.status).where(
                    order_items.c.world_id == ids["world_id"],
                    order_items.c.order_item_id == ids["order_item_id"],
                )
            )
            == "unfulfilled"
        )
        assert await _count(connection, refunds, ids["world_id"]) == 0
        assert await _count(connection, command_receipts, ids["world_id"]) == 0


@pytest.mark.asyncio
async def test_disconnect_after_commit_reconciles_receipt(engine: AsyncEngine) -> None:
    ids = await seed_world(engine)
    probes = 0

    async def disconnect_once(_request_id: str) -> None:
        nonlocal probes
        probes += 1
        if probes == 1:
            raise AmbiguousCommitError("simulated lost commit acknowledgement")

    database = Database(
        os_database_url(), pool_size=2, max_concurrency=4, after_commit=disconnect_once
    )
    try:
        result = await OrderResolutionService(database).cancel_order_item(
            world_id=ids["world_id"],
            order_number=ids["order_number"],
            order_item_id=ids["order_item_id"],
            reason="ambiguous commit",
        )
    finally:
        await database.aclose()
    assert result["code"] == "ITEM_CANCELLED"
    async with engine.connect() as connection:
        assert await _count(connection, refunds, ids["world_id"]) == 1
        assert await _count(connection, command_receipts, ids["world_id"]) == 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_command_has_one_effect(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    service = OrderResolutionService(database)

    async def invoke() -> dict[str, object]:
        return await service.cancel_order_item(
            world_id=ids["world_id"],
            order_number=ids["order_number"],
            order_item_id=ids["order_item_id"],
            reason="same request",
        )

    left, right = await asyncio.gather(invoke(), invoke())
    assert left == right
    async with engine.connect() as connection:
        assert await _count(connection, refunds, ids["world_id"]) == 1
        assert await _count(connection, command_receipts, ids["world_id"]) == 1


@pytest.mark.asyncio
async def test_database_constraints_reject_invalid_inventory(
    engine: AsyncEngine,
) -> None:
    ids = await seed_world(engine)
    with pytest.raises(IntegrityError):
        async with engine.begin() as connection:
            await connection.execute(
                sa.update(inventory)
                .where(
                    inventory.c.world_id == ids["world_id"],
                    inventory.c.variant_id == ids["old_variant_id"],
                )
                .values(reserved=999)
            )


def test_database_url_enforces_neon_direct_and_pooler_roles() -> None:
    direct = "postgresql://role:secret@ep-example.us-west-2.aws.neon.tech/db"
    pooled = "postgresql://role:secret@ep-example-pooler.us-west-2.aws.neon.tech/db"
    assert validate_database_url(direct, purpose="admin") == direct
    assert validate_database_url(pooled, purpose="runtime") == pooled
    with pytest.raises(DatabaseConfigurationError, match="direct"):
        validate_database_url(pooled, purpose="admin")
    with pytest.raises(DatabaseConfigurationError, match="pooled"):
        validate_database_url(direct, purpose="runtime")
    with pytest.raises(DatabaseConfigurationError, match="session options"):
        validate_database_url(f"{pooled}?options=-csearch_path%3Dpublic", purpose="runtime")


@pytest.mark.asyncio
async def test_commands_return_typed_command_codes(engine: AsyncEngine, database: Database) -> None:
    """Every emitted code is an enum member, so the vocabulary has one owner."""

    ids = await seed_world(engine, handed_off=True)
    service = OrderResolutionService(database)
    denied = await service.cancel_order_item(
        world_id=ids["world_id"],
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        reason="too late",
    )
    missing = await service.get_order(world_id=ids["world_id"], order_number="OR-NOPE")
    assert denied["code"] is CommandCode.ALREADY_HANDED_TO_CARRIER
    assert missing["code"] is CommandCode.ORDER_NOT_FOUND
    # StrEnum members serialize to the plain code the model must copy verbatim.
    assert json.loads(json.dumps(denied))["code"] == "ALREADY_HANDED_TO_CARRIER"


@pytest.mark.asyncio
async def test_get_order_exposes_customer_visible_product_identity(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    order = await OrderResolutionService(database).get_order(
        world_id=ids["world_id"], order_number=ids["order_number"]
    )
    assert order["ok"] is True
    item = order["items"][0]
    assert item["product_name"] == "Everyday shirt"
    assert item["sku"] == "SHIRT-BLUE-S"
    assert item["options"] == {"color": "blue", "size": "S"}


@pytest.mark.asyncio
async def test_invalid_reply_combination_is_non_terminal_and_keeps_the_case_open(
    engine: AsyncEngine, database: Database
) -> None:
    """A contract error must not close the case or consume the one valid reply."""

    ids = await seed_world(engine)
    service = OrderResolutionService(database)
    case_id = await _open_case(engine, ids)

    rejected = await service.reply_to_customer(
        world_id=ids["world_id"],
        case_id=case_id,
        disposition="completed",
        outcome_code="NEEDS_ORDER_ITEM",
        order_number=ids["order_number"],
        order_item_id=None,
        missing_fields=["order_item_id"],
    )
    assert rejected == {
        "ok": False,
        "code": EnvelopeCode.INVALID_REPLY_CONTRACT,
        "message": rejected["message"],
    }
    assert "requires disposition needs_information" in rejected["message"]

    async with engine.connect() as connection:
        assert await _case_status(connection, ids["world_id"], case_id) == ("open", None, None)
        assert await _count(connection, support_messages, ids["world_id"]) == 0
        assert await _count(connection, audit_events, ids["world_id"]) == 0

    accepted = await service.reply_to_customer(
        world_id=ids["world_id"],
        case_id=case_id,
        disposition="needs_information",
        outcome_code="NEEDS_ORDER_ITEM",
        order_number=ids["order_number"],
        order_item_id=None,
        missing_fields=["order_item_id"],
    )
    assert accepted["ok"] is True and accepted["terminal"] is True
    async with engine.connect() as connection:
        assert await _case_status(connection, ids["world_id"], case_id) == (
            "closed",
            "needs_information",
            "NEEDS_ORDER_ITEM",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome_code", ["ORDER_ITEM_NOT_FOUND", "ALREADY_CANCELLED", "REPLY_ALREADY_SENT"]
)
async def test_reply_rejects_retry_and_protocol_codes(
    engine: AsyncEngine, database: Database, outcome_code: str
) -> None:
    ids = await seed_world(engine)
    case_id = await _open_case(engine, ids)
    result = await OrderResolutionService(database).reply_to_customer(
        world_id=ids["world_id"],
        case_id=case_id,
        disposition="cannot_complete",
        outcome_code=outcome_code,
        order_number=ids["order_number"],
        order_item_id=None,
        missing_fields=[],
    )
    assert result["code"] is EnvelopeCode.INVALID_REPLY_CONTRACT
    async with engine.connect() as connection:
        assert await _case_status(connection, ids["world_id"], case_id) == ("open", None, None)


@pytest.mark.asyncio
async def test_second_reply_returns_the_protocol_code(
    engine: AsyncEngine, database: Database
) -> None:
    ids = await seed_world(engine)
    case_id = await _open_case(engine, ids)
    service = OrderResolutionService(database)
    reply = dict(
        world_id=ids["world_id"],
        case_id=case_id,
        disposition="cannot_complete",
        outcome_code="ALREADY_HANDED_TO_CARRIER",
        order_number=ids["order_number"],
        order_item_id=ids["order_item_id"],
        missing_fields=[],
    )
    assert (await service.reply_to_customer(**reply))["ok"] is True
    second = await service.reply_to_customer(**{**reply, "order_item_id": None})
    assert second["code"] is CommandCode.REPLY_ALREADY_SENT


async def _open_case(engine: AsyncEngine, ids: dict) -> str:
    case_id = f"case-{ids['world_id'][-12:]}"
    async with engine.begin() as connection:
        await connection.execute(
            sa.insert(support_cases).values(
                world_id=ids["world_id"],
                case_id=case_id,
                customer_id=ids["customer_id"],
                order_id=ids["order_id"],
                status="open",
                disposition=None,
                outcome_code=None,
                created_at=datetime(2026, 8, 5, 12, tzinfo=UTC),
            )
        )
    return case_id


async def _case_status(connection, world_id: str, case_id: str) -> tuple:
    row = (
        await connection.execute(
            sa.select(
                support_cases.c.status,
                support_cases.c.disposition,
                support_cases.c.outcome_code,
            ).where(support_cases.c.world_id == world_id, support_cases.c.case_id == case_id)
        )
    ).one()
    return tuple(row)


async def _count(connection, table: sa.Table, world_id: str) -> int:
    return int(
        await connection.scalar(
            sa.select(sa.func.count()).select_from(table).where(table.c.world_id == world_id)
        )
        or 0
    )


def _new_address() -> dict[str, str]:
    return {
        "line1": "99 New St",
        "city": "Berkeley",
        "region": "CA",
        "postal_code": "94704",
        "country": "us",
    }


def os_database_url() -> str:
    import os

    value = os.environ.get("ORDER_RESOLUTION_TEST_DATABASE_URL")
    assert value
    return value
