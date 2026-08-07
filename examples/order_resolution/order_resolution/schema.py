"""SQLAlchemy Core ownership of the order-resolution database schema."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

CATALOG_SCHEMA = "catalog"
COMMERCE_SCHEMA = "commerce"
BENCH_SCHEMA = "bench"
SCHEMAS = (CATALOG_SCHEMA, COMMERCE_SCHEMA, BENCH_SCHEMA)

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)


products = sa.Table(
    "products",
    metadata,
    sa.Column("product_id", sa.Text, primary_key=True),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("category", sa.Text, nullable=False),
    schema=CATALOG_SCHEMA,
)

product_variants = sa.Table(
    "product_variants",
    metadata,
    sa.Column("variant_id", sa.Text, primary_key=True),
    sa.Column(
        "product_id",
        sa.Text,
        sa.ForeignKey(f"{CATALOG_SCHEMA}.products.product_id", ondelete="RESTRICT"),
        nullable=False,
    ),
    sa.Column("sku", sa.Text, nullable=False, unique=True),
    sa.Column("price_minor", sa.Integer, nullable=False),
    sa.Column("currency", sa.Text, nullable=False),
    sa.Column("options", JSONB, nullable=False),
    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.CheckConstraint("price_minor >= 0", name="price_nonnegative"),
    schema=CATALOG_SCHEMA,
)

warehouses = sa.Table(
    "warehouses",
    metadata,
    sa.Column("warehouse_id", sa.Text, primary_key=True),
    sa.Column("name", sa.Text, nullable=False),
    schema=CATALOG_SCHEMA,
)

worlds = sa.Table(
    "worlds",
    metadata,
    sa.Column("world_id", sa.Text, primary_key=True),
    sa.Column("scenario_id", sa.Text, nullable=False),
    sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
    sa.Column("next_event_seq", sa.BigInteger, nullable=False, server_default="0"),
    sa.Column("retain_operational_state", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("next_event_seq >= 0", name="event_seq_nonnegative"),
    schema=BENCH_SCHEMA,
)


def _world_id() -> sa.Column[str]:
    return sa.Column(
        "world_id",
        sa.Text,
        sa.ForeignKey(f"{BENCH_SCHEMA}.worlds.world_id", ondelete="CASCADE"),
        primary_key=True,
    )


customers = sa.Table(
    "customers",
    metadata,
    _world_id(),
    sa.Column("customer_id", sa.Text, primary_key=True),
    sa.Column("email", sa.Text, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.UniqueConstraint("world_id", "email"),
    schema=COMMERCE_SCHEMA,
)

addresses = sa.Table(
    "addresses",
    metadata,
    _world_id(),
    sa.Column("address_id", sa.Text, primary_key=True),
    sa.Column("customer_id", sa.Text, nullable=False),
    sa.Column("line1", sa.Text, nullable=False),
    sa.Column("line2", sa.Text),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("region", sa.Text, nullable=False),
    sa.Column("postal_code", sa.Text, nullable=False),
    sa.Column("country", sa.Text, nullable=False),
    sa.Column("created_event_seq", sa.BigInteger, nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "customer_id"],
        [f"{COMMERCE_SCHEMA}.customers.world_id", f"{COMMERCE_SCHEMA}.customers.customer_id"],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("created_event_seq >= 0", name="created_event_seq_nonnegative"),
    schema=COMMERCE_SCHEMA,
)

orders = sa.Table(
    "orders",
    metadata,
    _world_id(),
    sa.Column("order_id", sa.Text, primary_key=True),
    sa.Column("order_number", sa.Text, nullable=False),
    sa.Column("customer_id", sa.Text, nullable=False),
    sa.Column("shipping_address_id", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("currency", sa.Text, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "customer_id"],
        [f"{COMMERCE_SCHEMA}.customers.world_id", f"{COMMERCE_SCHEMA}.customers.customer_id"],
        ondelete="RESTRICT",
    ),
    sa.ForeignKeyConstraint(
        ["world_id", "shipping_address_id"],
        [f"{COMMERCE_SCHEMA}.addresses.world_id", f"{COMMERCE_SCHEMA}.addresses.address_id"],
        ondelete="RESTRICT",
    ),
    sa.UniqueConstraint("world_id", "order_number"),
    schema=COMMERCE_SCHEMA,
)

inventory = sa.Table(
    "inventory",
    metadata,
    _world_id(),
    sa.Column(
        "warehouse_id",
        sa.Text,
        sa.ForeignKey(f"{CATALOG_SCHEMA}.warehouses.warehouse_id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Column(
        "variant_id",
        sa.Text,
        sa.ForeignKey(f"{CATALOG_SCHEMA}.product_variants.variant_id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Column("on_hand", sa.Integer, nullable=False),
    sa.Column("reserved", sa.Integer, nullable=False),
    sa.CheckConstraint("on_hand >= 0", name="on_hand_nonnegative"),
    sa.CheckConstraint("reserved >= 0", name="reserved_nonnegative"),
    sa.CheckConstraint("reserved <= on_hand", name="reserved_not_over_on_hand"),
    schema=COMMERCE_SCHEMA,
)

order_items = sa.Table(
    "order_items",
    metadata,
    _world_id(),
    sa.Column("order_item_id", sa.Text, primary_key=True),
    sa.Column("order_id", sa.Text, nullable=False),
    sa.Column(
        "variant_id",
        sa.Text,
        sa.ForeignKey(f"{CATALOG_SCHEMA}.product_variants.variant_id", ondelete="RESTRICT"),
        nullable=False,
    ),
    sa.Column("quantity", sa.Integer, nullable=False),
    sa.Column("unit_price_minor", sa.Integer, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "order_id"],
        [f"{COMMERCE_SCHEMA}.orders.world_id", f"{COMMERCE_SCHEMA}.orders.order_id"],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("quantity > 0", name="quantity_positive"),
    sa.CheckConstraint("unit_price_minor >= 0", name="unit_price_nonnegative"),
    schema=COMMERCE_SCHEMA,
)

inventory_allocations = sa.Table(
    "inventory_allocations",
    metadata,
    _world_id(),
    sa.Column("allocation_id", sa.Text, primary_key=True),
    sa.Column("order_item_id", sa.Text, nullable=False),
    sa.Column("warehouse_id", sa.Text, nullable=False),
    sa.Column("variant_id", sa.Text, nullable=False),
    sa.Column("quantity", sa.Integer, nullable=False),
    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.ForeignKeyConstraint(
        ["world_id", "order_item_id"],
        [
            f"{COMMERCE_SCHEMA}.order_items.world_id",
            f"{COMMERCE_SCHEMA}.order_items.order_item_id",
        ],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["world_id", "warehouse_id", "variant_id"],
        [
            f"{COMMERCE_SCHEMA}.inventory.world_id",
            f"{COMMERCE_SCHEMA}.inventory.warehouse_id",
            f"{COMMERCE_SCHEMA}.inventory.variant_id",
        ],
        ondelete="RESTRICT",
    ),
    sa.CheckConstraint("quantity > 0", name="quantity_positive"),
    schema=COMMERCE_SCHEMA,
)
sa.Index(
    "uq_inventory_allocations_active_item",
    inventory_allocations.c.world_id,
    inventory_allocations.c.order_item_id,
    unique=True,
    postgresql_where=inventory_allocations.c.active.is_(True),
)

payments = sa.Table(
    "payments",
    metadata,
    _world_id(),
    sa.Column("payment_id", sa.Text, primary_key=True),
    sa.Column("order_id", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("captured_minor", sa.Integer, nullable=False),
    sa.Column("currency", sa.Text, nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "order_id"],
        [f"{COMMERCE_SCHEMA}.orders.world_id", f"{COMMERCE_SCHEMA}.orders.order_id"],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("captured_minor >= 0", name="captured_nonnegative"),
    schema=COMMERCE_SCHEMA,
)

refunds = sa.Table(
    "refunds",
    metadata,
    _world_id(),
    sa.Column("refund_id", sa.Text, primary_key=True),
    sa.Column("payment_id", sa.Text, nullable=False),
    sa.Column("order_item_id", sa.Text, nullable=False),
    sa.Column("amount_minor", sa.Integer, nullable=False),
    sa.Column("currency", sa.Text, nullable=False),
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("request_id", sa.Text, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "payment_id"],
        [f"{COMMERCE_SCHEMA}.payments.world_id", f"{COMMERCE_SCHEMA}.payments.payment_id"],
        ondelete="RESTRICT",
    ),
    sa.ForeignKeyConstraint(
        ["world_id", "order_item_id"],
        [
            f"{COMMERCE_SCHEMA}.order_items.world_id",
            f"{COMMERCE_SCHEMA}.order_items.order_item_id",
        ],
        ondelete="RESTRICT",
    ),
    sa.UniqueConstraint("world_id", "request_id"),
    sa.CheckConstraint("amount_minor > 0", name="amount_positive"),
    schema=COMMERCE_SCHEMA,
)

shipments = sa.Table(
    "shipments",
    metadata,
    _world_id(),
    sa.Column("shipment_id", sa.Text, primary_key=True),
    sa.Column("order_id", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("carrier_handoff_at", sa.DateTime(timezone=True)),
    sa.Column("delivered_at", sa.DateTime(timezone=True)),
    sa.ForeignKeyConstraint(
        ["world_id", "order_id"],
        [f"{COMMERCE_SCHEMA}.orders.world_id", f"{COMMERCE_SCHEMA}.orders.order_id"],
        ondelete="CASCADE",
    ),
    schema=COMMERCE_SCHEMA,
)

support_cases = sa.Table(
    "support_cases",
    metadata,
    _world_id(),
    sa.Column("case_id", sa.Text, primary_key=True),
    sa.Column("customer_id", sa.Text, nullable=False),
    sa.Column("order_id", sa.Text),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("disposition", sa.Text),
    sa.Column("outcome_code", sa.Text),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "customer_id"],
        [f"{COMMERCE_SCHEMA}.customers.world_id", f"{COMMERCE_SCHEMA}.customers.customer_id"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["world_id", "order_id"],
        [f"{COMMERCE_SCHEMA}.orders.world_id", f"{COMMERCE_SCHEMA}.orders.order_id"],
        ondelete="RESTRICT",
    ),
    schema=COMMERCE_SCHEMA,
)

support_messages = sa.Table(
    "support_messages",
    metadata,
    _world_id(),
    sa.Column("message_id", sa.Text, primary_key=True),
    sa.Column("case_id", sa.Text, nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("message_kind", sa.Text, nullable=False),
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("reply_facts", JSONB),
    sa.Column("event_seq", sa.BigInteger, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(
        ["world_id", "case_id"],
        [f"{COMMERCE_SCHEMA}.support_cases.world_id", f"{COMMERCE_SCHEMA}.support_cases.case_id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("world_id", "case_id", "event_seq"),
    schema=COMMERCE_SCHEMA,
)

audit_events = sa.Table(
    "audit_events",
    metadata,
    _world_id(),
    sa.Column("event_id", sa.Text, primary_key=True),
    sa.Column("event_seq", sa.BigInteger, nullable=False),
    sa.Column("actor", sa.Text, nullable=False),
    sa.Column("action", sa.Text, nullable=False),
    sa.Column("entity_type", sa.Text, nullable=False),
    sa.Column("entity_id", sa.Text, nullable=False),
    sa.Column("before_state", JSONB, nullable=False),
    sa.Column("after_state", JSONB, nullable=False),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("request_id", sa.Text, nullable=False),
    sa.UniqueConstraint("world_id", "event_seq"),
    sa.UniqueConstraint("world_id", "request_id"),
    schema=COMMERCE_SCHEMA,
)

command_receipts = sa.Table(
    "command_receipts",
    metadata,
    _world_id(),
    sa.Column("receipt_id", sa.Text, primary_key=True),
    sa.Column("command_name", sa.Text, nullable=False),
    sa.Column("request_hash", sa.Text, nullable=False),
    sa.Column("result", JSONB, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.UniqueConstraint("world_id", "receipt_id"),
    sa.UniqueConstraint("world_id", "command_name", "request_hash"),
    schema=BENCH_SCHEMA,
)

episode_results = sa.Table(
    "episode_results",
    metadata,
    _world_id(),
    sa.Column("scenario_id", sa.Text, nullable=False),
    sa.Column("before_snapshot", JSONB, nullable=False),
    sa.Column("after_snapshot", JSONB, nullable=False),
    sa.Column("reward", sa.Float, nullable=False),
    sa.Column("diagnostics", JSONB, nullable=False),
    sa.Column(
        "recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    schema=BENCH_SCHEMA,
)

ALL_TABLES = tuple(metadata.sorted_tables)
