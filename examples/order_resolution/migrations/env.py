"""Alembic environment for generated order-resolution migrations."""

from __future__ import annotations

import os
from logging.config import fileConfig
from urllib.parse import urlsplit, urlunsplit

from alembic import context
from order_resolution.database import validate_database_url
from order_resolution.schema import metadata
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = metadata


def _migration_url() -> str:
    raw_url = os.environ.get("ORDER_RESOLUTION_ADMIN_DATABASE_URL")
    if not raw_url:
        raise RuntimeError("ORDER_RESOLUTION_ADMIN_DATABASE_URL is required")
    validate_database_url(raw_url, purpose="admin")
    parsed = urlsplit(raw_url)
    return urlunsplit(("postgresql+psycopg", parsed.netloc, parsed.path, parsed.query, ""))


def _configure(connection=None, *, url: str | None = None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        include_schemas=True,
        compare_type=True,
        literal_binds=url is not None,
        dialect_opts={"paramstyle": "named"},
    )


def run_migrations_offline() -> None:
    _configure(url=_migration_url())
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _migration_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _configure(connection)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
