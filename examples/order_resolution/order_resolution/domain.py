"""Atomic order-resolution policies and state transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from order_resolution.command_codes import CommandCode, EnvelopeCode
from order_resolution.database import Database
from order_resolution.policy import ReplyContractError, validate_reply
from order_resolution.schema import (
    addresses,
    audit_events,
    customers,
    inventory,
    inventory_allocations,
    order_items,
    orders,
    payments,
    product_variants,
    products,
    refunds,
    shipments,
    support_cases,
    support_messages,
    worlds,
)

HANDED_OFF_SHIPMENT_STATUSES = {"handed_off", "in_transit", "delivered"}
IMMUTABLE_ITEM_STATUSES = {"shipped", "delivered", "cancelled"}
ADDRESS_FIELDS = ("line1", "city", "region", "postal_code", "country")


class DomainInvariantError(RuntimeError):
    """Stored state violates a domain invariant and must roll back."""


class OrderResolutionService:
    """Deep business-command interface over the order-resolution database."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def lookup_orders(
        self, *, world_id: str, customer_email: str, order_number: str | None = None
    ) -> dict[str, Any]:
        email = _required_text(customer_email, "customer_email").lower()

        async def read(connection: AsyncConnection) -> dict[str, Any]:
            query = (
                sa.select(orders.c.order_number, orders.c.status, orders.c.created_at)
                .select_from(
                    customers.join(
                        orders,
                        sa.and_(
                            orders.c.world_id == customers.c.world_id,
                            orders.c.customer_id == customers.c.customer_id,
                        ),
                    )
                )
                .where(customers.c.world_id == world_id, sa.func.lower(customers.c.email) == email)
                .order_by(orders.c.created_at.desc())
            )
            if order_number is not None:
                query = query.where(orders.c.order_number == order_number)
            rows = (await connection.execute(query)).mappings().all()
            return {
                "ok": True,
                "orders": [
                    {
                        "order_number": row["order_number"],
                        "status": row["status"],
                        "created_at": row["created_at"].isoformat(),
                    }
                    for row in rows
                ],
            }

        return await self._database.read(read)

    async def get_order(self, *, world_id: str, order_number: str) -> dict[str, Any]:
        async def read(connection: AsyncConnection) -> dict[str, Any]:
            order = (
                (
                    await connection.execute(
                        sa.select(orders).where(
                            orders.c.world_id == world_id,
                            orders.c.order_number == order_number,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if order is None:
                return _failure(CommandCode.ORDER_NOT_FOUND)
            item_rows = (
                await connection.execute(
                    sa.select(
                        order_items.c.order_item_id,
                        order_items.c.variant_id,
                        order_items.c.quantity,
                        order_items.c.unit_price_minor,
                        order_items.c.status,
                        products.c.name.label("product_name"),
                        product_variants.c.sku,
                        product_variants.c.options,
                    )
                    .select_from(
                        order_items.join(
                            product_variants,
                            product_variants.c.variant_id == order_items.c.variant_id,
                        ).join(
                            products,
                            products.c.product_id == product_variants.c.product_id,
                        )
                    )
                    .where(
                        order_items.c.world_id == world_id,
                        order_items.c.order_id == order["order_id"],
                    )
                    .order_by(order_items.c.order_item_id)
                )
            ).mappings()
            shipment = (
                (
                    await connection.execute(
                        sa.select(shipments.c.status, shipments.c.carrier_handoff_at).where(
                            shipments.c.world_id == world_id,
                            shipments.c.order_id == order["order_id"],
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            return {
                "ok": True,
                "order_number": order_number,
                "status": order["status"],
                "shipping_address_id": order["shipping_address_id"],
                "items": [dict(row) for row in item_rows],
                "shipment": (
                    {
                        "status": shipment["status"],
                        "carrier_handoff_at": (
                            shipment["carrier_handoff_at"].isoformat()
                            if shipment["carrier_handoff_at"]
                            else None
                        ),
                    }
                    if shipment
                    else None
                ),
            }

        return await self._database.read(read)

    async def check_variant_availability(
        self,
        *,
        world_id: str,
        order_number: str,
        order_item_id: str,
        requested_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        options = {
            _required_text(key, "requested option name"): _required_text(value, str(key))
            for key, value in requested_options.items()
        }

        async def read(connection: AsyncConnection) -> dict[str, Any]:
            record = await _locked_item(
                connection,
                world_id=world_id,
                order_number=order_number,
                order_item_id=order_item_id,
            )
            if record is None:
                return _failure(CommandCode.ORDER_ITEM_NOT_FOUND)
            current = (
                await connection.execute(
                    sa.select(product_variants.c.product_id).where(
                        product_variants.c.variant_id == record["variant_id"]
                    )
                )
            ).scalar_one()
            rows = (
                await connection.execute(
                    sa.select(
                        product_variants.c.variant_id,
                        product_variants.c.options,
                        product_variants.c.price_minor,
                        inventory.c.on_hand,
                        inventory.c.reserved,
                    )
                    .select_from(
                        product_variants.join(
                            inventory,
                            inventory.c.variant_id == product_variants.c.variant_id,
                        )
                    )
                    .where(
                        product_variants.c.product_id == current,
                        product_variants.c.active.is_(True),
                        inventory.c.world_id == world_id,
                    )
                    .order_by(product_variants.c.variant_id)
                )
            ).mappings()
            candidates = [
                {
                    "variant_id": row["variant_id"],
                    "options": row["options"],
                    "same_price": row["price_minor"] == record["unit_price_minor"],
                    "available": row["on_hand"] - row["reserved"],
                }
                for row in rows
                if all(str(row["options"].get(key)) == value for key, value in options.items())
            ]
            return {"ok": True, "candidates": candidates}

        return await self._database.read(read)

    async def reply_to_customer(
        self,
        *,
        world_id: str,
        case_id: str,
        disposition: str,
        outcome_code: str,
        order_number: str,
        order_item_id: str | None,
        missing_fields: Sequence[str],
    ) -> dict[str, Any]:
        facts = {
            "disposition": disposition,
            "outcome_code": _required_text(outcome_code, "outcome_code"),
            "order_number": _required_text(order_number, "order_number"),
            "order_item_id": order_item_id,
            "missing_fields": sorted(
                {_required_text(value, "missing_field") for value in missing_fields}
            ),
        }
        # Reject impossible combinations before the transaction so a contract
        # error stays visible, non-terminal, and does not consume the one reply.
        try:
            validate_reply(
                disposition=facts["disposition"],
                outcome_code=facts["outcome_code"],
                order_item_id=facts["order_item_id"],
                missing_fields=facts["missing_fields"],
            )
        except ReplyContractError as error:
            return _failure(EnvelopeCode.INVALID_REPLY_CONTRACT, str(error))
        payload = {"world_id": world_id, "case_id": case_id, **facts}

        async def work(connection: AsyncConnection, request_id: str) -> dict[str, Any]:
            support_case = (
                (
                    await connection.execute(
                        sa.select(support_cases)
                        .where(
                            support_cases.c.world_id == world_id,
                            support_cases.c.case_id == case_id,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if support_case is None:
                raise DomainInvariantError("support case is missing")
            prior_reply = await connection.scalar(
                sa.select(sa.func.count())
                .select_from(support_messages)
                .where(
                    support_messages.c.world_id == world_id,
                    support_messages.c.case_id == case_id,
                    support_messages.c.message_kind == "customer_reply",
                )
            )
            if prior_reply:
                return _failure(CommandCode.REPLY_ALREADY_SENT)
            event_seq, occurred_at = await _next_event(connection, world_id)
            await connection.execute(
                sa.update(support_cases)
                .where(
                    support_cases.c.world_id == world_id,
                    support_cases.c.case_id == case_id,
                )
                .values(
                    status="closed",
                    disposition=disposition,
                    outcome_code=facts["outcome_code"],
                )
            )
            rendered = _render_reply(facts)
            await connection.execute(
                sa.insert(support_messages).values(
                    world_id=world_id,
                    message_id=_derived_id("message", request_id),
                    case_id=case_id,
                    role="assistant",
                    message_kind="customer_reply",
                    content=rendered,
                    reply_facts=facts,
                    event_seq=event_seq,
                    created_at=occurred_at,
                )
            )
            await _audit(
                connection,
                world_id=world_id,
                request_id=request_id,
                event_seq=event_seq,
                occurred_at=occurred_at,
                action="reply_to_customer",
                entity_type="support_case",
                entity_id=case_id,
                before={
                    "status": support_case["status"],
                    "disposition": support_case["disposition"],
                    "outcome_code": support_case["outcome_code"],
                },
                after={
                    "status": "closed",
                    "disposition": disposition,
                    "outcome_code": facts["outcome_code"],
                },
            )
            return {"ok": True, "terminal": True, "reply": facts, "rendered": rendered}

        return await self._database.execute_command(
            world_id=world_id,
            command_name="reply_to_customer",
            payload=payload,
            work=work,
        )

    async def cancel_order_item(
        self,
        *,
        world_id: str,
        order_number: str,
        order_item_id: str,
        reason: str,
    ) -> dict[str, Any]:
        payload = {
            "world_id": world_id,
            "order_number": order_number,
            "order_item_id": order_item_id,
            "reason": _required_text(reason, "reason"),
        }

        async def work(connection: AsyncConnection, request_id: str) -> dict[str, Any]:
            record = await _locked_item(
                connection,
                world_id=world_id,
                order_number=order_number,
                order_item_id=order_item_id,
            )
            if record is None:
                return _failure(CommandCode.ORDER_ITEM_NOT_FOUND)
            if record["item_status"] == "cancelled":
                return _failure(CommandCode.ALREADY_CANCELLED)
            if record["item_status"] in IMMUTABLE_ITEM_STATUSES or await _handed_off(
                connection, world_id=world_id, order_id=record["order_id"]
            ):
                return _failure(CommandCode.ALREADY_HANDED_TO_CARRIER)

            allocation = await _active_allocation(
                connection, world_id=world_id, order_item_id=order_item_id
            )
            if allocation is None or allocation["quantity"] != record["quantity"]:
                raise DomainInvariantError("active allocation does not match item quantity")
            stock = await _locked_inventory(
                connection,
                world_id=world_id,
                warehouse_id=allocation["warehouse_id"],
                variant_id=allocation["variant_id"],
            )
            if stock is None or stock["reserved"] < allocation["quantity"]:
                raise DomainInvariantError("inventory reservation is missing or undersized")

            payment = await _locked_payment(
                connection, world_id=world_id, order_id=record["order_id"]
            )
            if payment is None or payment["status"] != "captured":
                raise DomainInvariantError("cancellation requires one captured payment")
            refunded = await connection.scalar(
                sa.select(sa.func.coalesce(sa.func.sum(refunds.c.amount_minor), 0)).where(
                    refunds.c.world_id == world_id,
                    refunds.c.payment_id == payment["payment_id"],
                )
            )
            refund_minor = record["quantity"] * record["unit_price_minor"]
            if int(refunded or 0) + refund_minor > payment["captured_minor"]:
                raise DomainInvariantError("refund would exceed captured payment")

            changed = await connection.execute(
                sa.update(inventory)
                .where(
                    inventory.c.world_id == world_id,
                    inventory.c.warehouse_id == allocation["warehouse_id"],
                    inventory.c.variant_id == allocation["variant_id"],
                    inventory.c.reserved >= allocation["quantity"],
                )
                .values(reserved=inventory.c.reserved - allocation["quantity"])
            )
            _require_one(changed.rowcount, "inventory reservation release")
            await connection.execute(
                sa.update(inventory_allocations)
                .where(
                    inventory_allocations.c.world_id == world_id,
                    inventory_allocations.c.allocation_id == allocation["allocation_id"],
                    inventory_allocations.c.active.is_(True),
                )
                .values(active=False)
            )
            await connection.execute(
                sa.update(order_items)
                .where(
                    order_items.c.world_id == world_id,
                    order_items.c.order_item_id == order_item_id,
                )
                .values(status="cancelled")
            )
            event_seq, occurred_at = await _next_event(connection, world_id)
            refund_id = _derived_id("refund", request_id)
            await connection.execute(
                sa.insert(refunds).values(
                    world_id=world_id,
                    refund_id=refund_id,
                    payment_id=payment["payment_id"],
                    order_item_id=order_item_id,
                    amount_minor=refund_minor,
                    currency=record["currency"],
                    reason=payload["reason"],
                    request_id=request_id,
                    created_at=occurred_at,
                )
            )
            order_status = await _recompute_order_status(
                connection, world_id=world_id, order_id=record["order_id"]
            )
            await _audit(
                connection,
                world_id=world_id,
                request_id=request_id,
                event_seq=event_seq,
                occurred_at=occurred_at,
                action="cancel_order_item",
                entity_type="order_item",
                entity_id=order_item_id,
                before={
                    "status": record["item_status"],
                    "variant_id": record["variant_id"],
                    "reserved": stock["reserved"],
                },
                after={
                    "status": "cancelled",
                    "variant_id": record["variant_id"],
                    "reserved": stock["reserved"] - allocation["quantity"],
                },
            )
            return {
                "ok": True,
                "code": CommandCode.ITEM_CANCELLED,
                "order_number": order_number,
                "order_item_id": order_item_id,
                "refund_id": refund_id,
                "refund_minor": refund_minor,
                "currency": record["currency"],
                "order_status": order_status,
            }

        return await self._database.execute_command(
            world_id=world_id,
            command_name="cancel_order_item",
            payload=payload,
            work=work,
        )

    async def change_shipping_address(
        self,
        *,
        world_id: str,
        order_number: str,
        address: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = _normalized_address(address)
        payload = {
            "world_id": world_id,
            "order_number": order_number,
            "address": normalized,
        }

        async def work(connection: AsyncConnection, request_id: str) -> dict[str, Any]:
            order = await _locked_order(connection, world_id=world_id, order_number=order_number)
            if order is None:
                return _failure(CommandCode.ORDER_NOT_FOUND)
            if await _handed_off(connection, world_id=world_id, order_id=order["order_id"]):
                return _failure(CommandCode.ALREADY_HANDED_TO_CARRIER)
            old_address = (
                (
                    await connection.execute(
                        sa.select(addresses).where(
                            addresses.c.world_id == world_id,
                            addresses.c.address_id == order["shipping_address_id"],
                        )
                    )
                )
                .mappings()
                .one()
            )

            event_seq, occurred_at = await _next_event(connection, world_id)
            address_id = _derived_id("address", request_id)
            await connection.execute(
                sa.insert(addresses).values(
                    world_id=world_id,
                    address_id=address_id,
                    customer_id=order["customer_id"],
                    created_event_seq=event_seq,
                    **normalized,
                )
            )
            await connection.execute(
                sa.update(orders)
                .where(orders.c.world_id == world_id, orders.c.order_id == order["order_id"])
                .values(shipping_address_id=address_id)
            )
            await _audit(
                connection,
                world_id=world_id,
                request_id=request_id,
                event_seq=event_seq,
                occurred_at=occurred_at,
                action="change_shipping_address",
                entity_type="order",
                entity_id=order["order_id"],
                before={
                    "shipping_address_id": order["shipping_address_id"],
                    "address": _address_json(old_address),
                },
                after={"shipping_address_id": address_id, "address": normalized},
            )
            return {
                "ok": True,
                "code": CommandCode.SHIPPING_ADDRESS_CHANGED,
                "order_number": order_number,
                "address_id": address_id,
            }

        return await self._database.execute_command(
            world_id=world_id,
            command_name="change_shipping_address",
            payload=payload,
            work=work,
        )

    async def replace_order_item_variant(
        self,
        *,
        world_id: str,
        order_number: str,
        order_item_id: str,
        new_variant_id: str,
        reason: str,
    ) -> dict[str, Any]:
        payload = {
            "world_id": world_id,
            "order_number": order_number,
            "order_item_id": order_item_id,
            "new_variant_id": _required_text(new_variant_id, "new_variant_id"),
            "reason": _required_text(reason, "reason"),
        }

        async def work(connection: AsyncConnection, request_id: str) -> dict[str, Any]:
            record = await _locked_item(
                connection,
                world_id=world_id,
                order_number=order_number,
                order_item_id=order_item_id,
            )
            if record is None:
                return _failure(CommandCode.ORDER_ITEM_NOT_FOUND)
            if record["variant_id"] == new_variant_id:
                return _failure(CommandCode.ALREADY_REQUESTED_VARIANT)
            if record["item_status"] in IMMUTABLE_ITEM_STATUSES or await _handed_off(
                connection, world_id=world_id, order_id=record["order_id"]
            ):
                return _failure(CommandCode.ALREADY_HANDED_TO_CARRIER)

            variants = (
                (
                    await connection.execute(
                        sa.select(product_variants).where(
                            product_variants.c.variant_id.in_(
                                [record["variant_id"], new_variant_id]
                            )
                        )
                    )
                )
                .mappings()
                .all()
            )
            by_id = {variant["variant_id"]: variant for variant in variants}
            old_variant = by_id.get(record["variant_id"])
            new_variant = by_id.get(new_variant_id)
            if new_variant is None or not new_variant["active"]:
                return _failure(CommandCode.VARIANT_NOT_FOUND)
            if old_variant is None:
                raise DomainInvariantError("current variant is missing from the catalog")
            if (
                new_variant["product_id"] != old_variant["product_id"]
                or new_variant["price_minor"] != record["unit_price_minor"]
                or new_variant["currency"] != record["currency"]
            ):
                return _failure(CommandCode.PRICE_OR_PRODUCT_MISMATCH)

            allocation = await _active_allocation(
                connection, world_id=world_id, order_item_id=order_item_id
            )
            if allocation is None or allocation["variant_id"] != record["variant_id"]:
                raise DomainInvariantError("active allocation does not match current variant")
            stocks = await _locked_inventory_pair(
                connection,
                world_id=world_id,
                warehouse_id=allocation["warehouse_id"],
                variant_ids=[record["variant_id"], new_variant_id],
            )
            old_stock = stocks.get(record["variant_id"])
            new_stock = stocks.get(new_variant_id)
            if old_stock is None or old_stock["reserved"] < record["quantity"]:
                raise DomainInvariantError("current reservation is missing or undersized")
            if (
                new_stock is None
                or new_stock["on_hand"] - new_stock["reserved"] < record["quantity"]
            ):
                return _failure(CommandCode.VARIANT_OUT_OF_STOCK)

            release = await connection.execute(
                sa.update(inventory)
                .where(
                    inventory.c.world_id == world_id,
                    inventory.c.warehouse_id == allocation["warehouse_id"],
                    inventory.c.variant_id == record["variant_id"],
                    inventory.c.reserved >= record["quantity"],
                )
                .values(reserved=inventory.c.reserved - record["quantity"])
            )
            reserve = await connection.execute(
                sa.update(inventory)
                .where(
                    inventory.c.world_id == world_id,
                    inventory.c.warehouse_id == allocation["warehouse_id"],
                    inventory.c.variant_id == new_variant_id,
                    inventory.c.on_hand - inventory.c.reserved >= record["quantity"],
                )
                .values(reserved=inventory.c.reserved + record["quantity"])
            )
            _require_one(release.rowcount, "old inventory reservation release")
            _require_one(reserve.rowcount, "new inventory reservation")
            await connection.execute(
                sa.update(inventory_allocations)
                .where(
                    inventory_allocations.c.world_id == world_id,
                    inventory_allocations.c.allocation_id == allocation["allocation_id"],
                    inventory_allocations.c.active.is_(True),
                )
                .values(active=False)
            )
            new_allocation_id = _derived_id("allocation", request_id)
            await connection.execute(
                sa.insert(inventory_allocations).values(
                    world_id=world_id,
                    allocation_id=new_allocation_id,
                    order_item_id=order_item_id,
                    warehouse_id=allocation["warehouse_id"],
                    variant_id=new_variant_id,
                    quantity=record["quantity"],
                    active=True,
                )
            )
            await connection.execute(
                sa.update(order_items)
                .where(
                    order_items.c.world_id == world_id,
                    order_items.c.order_item_id == order_item_id,
                )
                .values(variant_id=new_variant_id)
            )
            event_seq, occurred_at = await _next_event(connection, world_id)
            await _audit(
                connection,
                world_id=world_id,
                request_id=request_id,
                event_seq=event_seq,
                occurred_at=occurred_at,
                action="replace_order_item_variant",
                entity_type="order_item",
                entity_id=order_item_id,
                before={
                    "variant_id": record["variant_id"],
                    "allocation_id": allocation["allocation_id"],
                },
                after={
                    "variant_id": new_variant_id,
                    "allocation_id": new_allocation_id,
                },
            )
            return {
                "ok": True,
                "code": CommandCode.ITEM_VARIANT_REPLACED,
                "order_number": order_number,
                "order_item_id": order_item_id,
                "old_variant_id": record["variant_id"],
                "new_variant_id": new_variant_id,
            }

        return await self._database.execute_command(
            world_id=world_id,
            command_name="replace_order_item_variant",
            payload=payload,
            work=work,
        )


async def assert_world_invariants(connection: AsyncConnection, world_id: str) -> None:
    """Raise when relational state violates an invariant not owned by a DB check."""

    allocation_rows = (
        await connection.execute(
            sa.select(
                order_items.c.order_item_id,
                order_items.c.variant_id.label("item_variant_id"),
                order_items.c.quantity.label("item_quantity"),
                order_items.c.status,
                inventory_allocations.c.variant_id.label("allocation_variant_id"),
                inventory_allocations.c.quantity.label("allocation_quantity"),
            )
            .select_from(
                order_items.outerjoin(
                    inventory_allocations,
                    sa.and_(
                        inventory_allocations.c.world_id == order_items.c.world_id,
                        inventory_allocations.c.order_item_id == order_items.c.order_item_id,
                        inventory_allocations.c.active.is_(True),
                    ),
                )
            )
            .where(order_items.c.world_id == world_id)
        )
    ).mappings()
    for row in allocation_rows:
        if row["status"] == "unfulfilled" and (
            row["allocation_variant_id"] != row["item_variant_id"]
            or row["allocation_quantity"] != row["item_quantity"]
        ):
            raise DomainInvariantError("unfulfilled item lacks one matching active allocation")
        if row["status"] == "cancelled" and row["allocation_variant_id"] is not None:
            raise DomainInvariantError("cancelled item retains an active allocation")

    refund_rows = (
        await connection.execute(
            sa.select(
                payments.c.payment_id,
                payments.c.captured_minor,
                sa.func.coalesce(sa.func.sum(refunds.c.amount_minor), 0).label("refunded_minor"),
            )
            .select_from(
                payments.outerjoin(
                    refunds,
                    sa.and_(
                        refunds.c.world_id == payments.c.world_id,
                        refunds.c.payment_id == payments.c.payment_id,
                    ),
                )
            )
            .where(payments.c.world_id == world_id)
            .group_by(payments.c.payment_id, payments.c.captured_minor)
        )
    ).mappings()
    if any(row["refunded_minor"] > row["captured_minor"] for row in refund_rows):
        raise DomainInvariantError("refunds exceed captured payment")

    events = (
        await connection.execute(
            sa.select(audit_events.c.event_seq, audit_events.c.occurred_at)
            .where(audit_events.c.world_id == world_id)
            .order_by(audit_events.c.event_seq)
        )
    ).all()
    if any(
        right[0] <= left[0] or right[1] <= left[1]
        for left, right in zip(events, events[1:], strict=False)
    ):
        raise DomainInvariantError("audit event history is not monotonic")


async def _locked_order(
    connection: AsyncConnection, *, world_id: str, order_number: str
) -> Mapping[str, Any] | None:
    return (
        (
            await connection.execute(
                sa.select(orders)
                .where(orders.c.world_id == world_id, orders.c.order_number == order_number)
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )


async def _locked_item(
    connection: AsyncConnection,
    *,
    world_id: str,
    order_number: str,
    order_item_id: str,
) -> Mapping[str, Any] | None:
    return (
        (
            await connection.execute(
                sa.select(
                    orders.c.order_id,
                    orders.c.currency,
                    orders.c.status.label("order_status"),
                    order_items.c.order_item_id,
                    order_items.c.variant_id,
                    order_items.c.quantity,
                    order_items.c.unit_price_minor,
                    order_items.c.status.label("item_status"),
                )
                .select_from(
                    orders.join(
                        order_items,
                        sa.and_(
                            order_items.c.world_id == orders.c.world_id,
                            order_items.c.order_id == orders.c.order_id,
                        ),
                    )
                )
                .where(
                    orders.c.world_id == world_id,
                    orders.c.order_number == order_number,
                    order_items.c.order_item_id == order_item_id,
                )
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )


async def _handed_off(connection: AsyncConnection, *, world_id: str, order_id: str) -> bool:
    as_of = await connection.scalar(sa.select(worlds.c.as_of).where(worlds.c.world_id == world_id))
    if as_of is None:
        raise DomainInvariantError("world is missing")
    shipment = (
        (
            await connection.execute(
                sa.select(shipments.c.status, shipments.c.carrier_handoff_at)
                .where(shipments.c.world_id == world_id, shipments.c.order_id == order_id)
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )
    return bool(
        shipment
        and (
            shipment["status"] in HANDED_OFF_SHIPMENT_STATUSES
            or (
                shipment["carrier_handoff_at"] is not None
                and shipment["carrier_handoff_at"] <= as_of
            )
        )
    )


async def _active_allocation(
    connection: AsyncConnection, *, world_id: str, order_item_id: str
) -> Mapping[str, Any] | None:
    return (
        (
            await connection.execute(
                sa.select(inventory_allocations)
                .where(
                    inventory_allocations.c.world_id == world_id,
                    inventory_allocations.c.order_item_id == order_item_id,
                    inventory_allocations.c.active.is_(True),
                )
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )


async def _locked_inventory(
    connection: AsyncConnection, *, world_id: str, warehouse_id: str, variant_id: str
) -> Mapping[str, Any] | None:
    return (
        (
            await connection.execute(
                sa.select(inventory)
                .where(
                    inventory.c.world_id == world_id,
                    inventory.c.warehouse_id == warehouse_id,
                    inventory.c.variant_id == variant_id,
                )
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )


async def _locked_inventory_pair(
    connection: AsyncConnection,
    *,
    world_id: str,
    warehouse_id: str,
    variant_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    rows = (
        await connection.execute(
            sa.select(inventory)
            .where(
                inventory.c.world_id == world_id,
                inventory.c.warehouse_id == warehouse_id,
                inventory.c.variant_id.in_(sorted(set(variant_ids))),
            )
            .order_by(inventory.c.variant_id)
            .with_for_update()
        )
    ).mappings()
    return {row["variant_id"]: row for row in rows}


async def _locked_payment(
    connection: AsyncConnection, *, world_id: str, order_id: str
) -> Mapping[str, Any] | None:
    return (
        (
            await connection.execute(
                sa.select(payments)
                .where(payments.c.world_id == world_id, payments.c.order_id == order_id)
                .with_for_update()
            )
        )
        .mappings()
        .one_or_none()
    )


async def _next_event(connection: AsyncConnection, world_id: str) -> tuple[int, datetime]:
    row = (
        await connection.execute(
            sa.update(worlds)
            .where(worlds.c.world_id == world_id)
            .values(next_event_seq=worlds.c.next_event_seq + 1)
            .returning(worlds.c.next_event_seq, worlds.c.as_of)
        )
    ).one_or_none()
    if row is None:
        raise DomainInvariantError("world is missing")
    event_seq = int(row.next_event_seq)
    return event_seq, row.as_of + timedelta(microseconds=event_seq)


async def _recompute_order_status(
    connection: AsyncConnection, *, world_id: str, order_id: str
) -> str:
    statuses = list(
        (
            await connection.scalars(
                sa.select(order_items.c.status).where(
                    order_items.c.world_id == world_id,
                    order_items.c.order_id == order_id,
                )
            )
        ).all()
    )
    if not statuses:
        raise DomainInvariantError("order has no items")
    if all(status == "cancelled" for status in statuses):
        status = "cancelled"
    elif any(status == "cancelled" for status in statuses):
        status = "partially_cancelled"
    elif all(item_status in {"shipped", "delivered"} for item_status in statuses):
        status = "fulfilled"
    else:
        status = "processing"
    await connection.execute(
        sa.update(orders)
        .where(orders.c.world_id == world_id, orders.c.order_id == order_id)
        .values(status=status)
    )
    return status


async def _audit(
    connection: AsyncConnection,
    *,
    world_id: str,
    request_id: str,
    event_seq: int,
    occurred_at: datetime,
    action: str,
    entity_type: str,
    entity_id: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    await connection.execute(
        sa.insert(audit_events).values(
            world_id=world_id,
            event_id=_derived_id("event", request_id),
            event_seq=event_seq,
            actor="support_agent",
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            before_state=dict(before),
            after_state=dict(after),
            occurred_at=occurred_at,
            request_id=request_id,
        )
    )


def _failure(code: str, message: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "code": code}
    if message is not None:
        result["message"] = message
    return result


def _render_reply(facts: Mapping[str, Any]) -> str:
    order_number = facts["order_number"]
    outcome = str(facts["outcome_code"]).lower().replace("_", " ")
    if facts["disposition"] == "completed":
        return f"we completed {outcome} for order {order_number}."
    if facts["disposition"] == "needs_information":
        fields = ", ".join(facts["missing_fields"])
        return f"we need {fields} before we can update order {order_number}."
    return f"we cannot complete {outcome} for order {order_number}."


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalized_address(address: Mapping[str, Any]) -> dict[str, str | None]:
    missing = [field for field in ADDRESS_FIELDS if not str(address.get(field, "")).strip()]
    if missing:
        raise ValueError(f"address is missing required fields: {', '.join(missing)}")
    return {
        "line1": str(address["line1"]).strip(),
        "line2": str(address.get("line2") or "").strip() or None,
        "city": str(address["city"]).strip(),
        "region": str(address["region"]).strip(),
        "postal_code": str(address["postal_code"]).strip(),
        "country": str(address["country"]).strip().upper(),
    }


def _address_json(address: Mapping[str, Any]) -> dict[str, str | None]:
    return {field: address[field] for field in (*ADDRESS_FIELDS, "line2")}


def _derived_id(kind: str, request_id: str) -> str:
    digest = hashlib.sha256(f"{kind}:{request_id}".encode()).hexdigest()[:24]
    return f"{kind}_{digest}"


def _require_one(rowcount: int, operation: str) -> None:
    if rowcount != 1:
        raise DomainInvariantError(f"{operation} affected {rowcount} rows")


def stable_json(value: Mapping[str, Any]) -> str:
    """Expose the command canonicalization shape for deterministic fixtures/tests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = ["DomainInvariantError", "OrderResolutionService", "assert_world_invariants"]
