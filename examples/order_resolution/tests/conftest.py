"""Local PostgreSQL fixtures for order-resolution tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy as sa
from order_resolution.database import Database
from order_resolution.schema import (
    addresses,
    customers,
    inventory,
    inventory_allocations,
    order_items,
    orders,
    payments,
    product_variants,
    products,
    shipments,
    warehouses,
    worlds,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "integration: live external integration test")


def _database_url() -> str:
    value = os.environ.get("ORDER_RESOLUTION_TEST_DATABASE_URL")
    if not value:
        pytest.skip("ORDER_RESOLUTION_TEST_DATABASE_URL is required")
    return value


def _async_url(value: str) -> str:
    return value.replace("postgresql://", "postgresql+psycopg://", 1)


@pytest_asyncio.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    test_engine = create_async_engine(_async_url(_database_url()))
    try:
        yield test_engine
    finally:
        await test_engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def database() -> AsyncIterator[Database]:
    value = Database(_database_url(), max_concurrency=16, pool_size=8)
    try:
        yield value
    finally:
        await value.aclose()


async def seed_world(
    engine: AsyncEngine,
    *,
    handed_off: bool = False,
    captured_minor: int = 1_000,
    new_variant_stock: int = 5,
) -> dict[str, Any]:
    world_id = f"world-{uuid.uuid4()}"
    suffix = world_id[-8:]
    as_of = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    ids = {
        "world_id": world_id,
        "customer_id": f"customer-{suffix}",
        "address_id": f"address-{suffix}",
        "order_id": f"order-{suffix}",
        "order_number": f"OR-{suffix.upper()}",
        "order_item_id": f"item-{suffix}",
        "payment_id": f"payment-{suffix}",
        "shipment_id": f"shipment-{suffix}",
        "allocation_id": f"allocation-{suffix}",
        "old_variant_id": "variant-blue-small",
        "new_variant_id": "variant-blue-medium",
        "out_variant_id": "variant-blue-large",
        "expensive_variant_id": "variant-blue-premium",
        "warehouse_id": "warehouse-west",
    }
    async with engine.begin() as connection:
        await connection.execute(
            pg_insert(products)
            .values(product_id="product-shirt", name="Everyday shirt", category="apparel")
            .on_conflict_do_nothing()
        )
        for variant_id, sku, price_minor, size in (
            (ids["old_variant_id"], "SHIRT-BLUE-S", 1_000, "S"),
            (ids["new_variant_id"], "SHIRT-BLUE-M", 1_000, "M"),
            (ids["out_variant_id"], "SHIRT-BLUE-L", 1_000, "L"),
            (ids["expensive_variant_id"], "SHIRT-BLUE-P", 1_200, "P"),
        ):
            await connection.execute(
                pg_insert(product_variants)
                .values(
                    variant_id=variant_id,
                    product_id="product-shirt",
                    sku=sku,
                    price_minor=price_minor,
                    currency="USD",
                    options={"color": "blue", "size": size},
                    active=True,
                )
                .on_conflict_do_nothing()
            )
        await connection.execute(
            pg_insert(warehouses)
            .values(warehouse_id=ids["warehouse_id"], name="West warehouse")
            .on_conflict_do_nothing()
        )
        await connection.execute(
            sa.insert(worlds).values(
                world_id=world_id,
                scenario_id=f"scenario-{suffix}",
                as_of=as_of,
                next_event_seq=0,
                retain_operational_state=False,
                expires_at=as_of + timedelta(days=1),
            )
        )
        await connection.execute(
            sa.insert(customers).values(
                world_id=world_id,
                customer_id=ids["customer_id"],
                email=f"customer-{suffix}@example.test",
                name="Test customer",
            )
        )
        await connection.execute(
            sa.insert(addresses).values(
                world_id=world_id,
                address_id=ids["address_id"],
                customer_id=ids["customer_id"],
                line1="1 Original Ave",
                line2=None,
                city="Oakland",
                region="CA",
                postal_code="94612",
                country="US",
                created_event_seq=0,
            )
        )
        await connection.execute(
            sa.insert(orders).values(
                world_id=world_id,
                order_id=ids["order_id"],
                order_number=ids["order_number"],
                customer_id=ids["customer_id"],
                shipping_address_id=ids["address_id"],
                status="processing",
                currency="USD",
                created_at=as_of - timedelta(days=1),
            )
        )
        await connection.execute(
            sa.insert(order_items).values(
                world_id=world_id,
                order_item_id=ids["order_item_id"],
                order_id=ids["order_id"],
                variant_id=ids["old_variant_id"],
                quantity=1,
                unit_price_minor=1_000,
                status="unfulfilled",
            )
        )
        for variant_id, on_hand, reserved in (
            (ids["old_variant_id"], 10, 1),
            (ids["new_variant_id"], new_variant_stock, 0),
            (ids["out_variant_id"], 0, 0),
            (ids["expensive_variant_id"], 5, 0),
        ):
            await connection.execute(
                sa.insert(inventory).values(
                    world_id=world_id,
                    warehouse_id=ids["warehouse_id"],
                    variant_id=variant_id,
                    on_hand=on_hand,
                    reserved=reserved,
                )
            )
        await connection.execute(
            sa.insert(inventory_allocations).values(
                world_id=world_id,
                allocation_id=ids["allocation_id"],
                order_item_id=ids["order_item_id"],
                warehouse_id=ids["warehouse_id"],
                variant_id=ids["old_variant_id"],
                quantity=1,
                active=True,
            )
        )
        await connection.execute(
            sa.insert(payments).values(
                world_id=world_id,
                payment_id=ids["payment_id"],
                order_id=ids["order_id"],
                status="captured",
                captured_minor=captured_minor,
                currency="USD",
            )
        )
        await connection.execute(
            sa.insert(shipments).values(
                world_id=world_id,
                shipment_id=ids["shipment_id"],
                order_id=ids["order_id"],
                status="handed_off" if handed_off else "pending",
                carrier_handoff_at=as_of - timedelta(minutes=1) if handed_off else None,
                delivered_at=None,
            )
        )
    return ids
