"""Live Neon privilege, isolation, expiration, load, and teardown gate."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from order_resolution.branching import (
    NeonApi,
    read_project_manifest,
    resolve_neon_api_key,
    retarget_runtime_url,
)
from order_resolution.database import DatabaseConfigurationError, validate_database_url
from order_resolution.fixtures import build_v2_catalog
from order_resolution.smoke import run_load_smoke

EXAMPLE_ROOT = Path(__file__).parents[1]
WORKTREE_ROOT = Path(__file__).parents[5]
DATA_DIR = EXAMPLE_ROOT / "data"
MANIFEST_PATH = EXAMPLE_ROOT / "artifacts" / "neon.json"
NEON_ENV_PATH = WORKTREE_ROOT / ".neon.env"


def _cannot_connect(database_url: str) -> None:
    with pytest.raises(psycopg.OperationalError):
        psycopg.connect(database_url, connect_timeout=5)


@pytest.mark.integration
def test_live_neon_child_is_least_privilege_isolated_and_disposable() -> None:
    manifest = read_project_manifest(MANIFEST_PATH)
    api_key = resolve_neon_api_key(NEON_ENV_PATH)
    primary = None
    sibling = None
    primary_deleted = False
    with NeonApi(api_key) as api:
        try:
            primary = api.create_runtime_branch(manifest, purpose="integration-primary")
            sibling = api.create_runtime_branch(manifest, purpose="integration-sibling")
            expires_at = datetime.fromisoformat(primary.expires_at.replace("Z", "+00:00"))
            assert datetime.now(UTC) < expires_at <= datetime.now(UTC) + timedelta(hours=24)
            assert (
                validate_database_url(primary.admin_database_url, purpose="admin")
                == primary.admin_database_url
            )
            assert (
                validate_database_url(primary.runtime_database_url, purpose="runtime")
                == primary.runtime_database_url
            )
            with pytest.raises(DatabaseConfigurationError, match="direct Neon endpoint"):
                validate_database_url(primary.runtime_database_url, purpose="admin")
            assert "postgresql://" not in repr(primary)

            with psycopg.connect(
                primary.runtime_database_url,
                autocommit=True,
                connect_timeout=30,
            ) as connection:
                with connection.cursor() as cursor:
                    # Both namespaces coexist: v1's rows are never modified, and
                    # v2 lives under its own content-addressed prefix.
                    catalog = build_v2_catalog()
                    cursor.execute(
                        "SELECT count(*) FROM catalog.products WHERE product_id LIKE %s",
                        (f"p{catalog.id_prefix}-%",),
                    )
                    assert cursor.fetchone() == (250,)
                    cursor.execute(
                        "SELECT count(*) FROM catalog.product_variants WHERE variant_id LIKE %s",
                        (f"v{catalog.id_prefix}-%",),
                    )
                    assert cursor.fetchone() == (750,)
                    cursor.execute(
                        "SELECT count(*) FROM catalog.products WHERE product_id LIKE 'product-%'"
                    )
                    assert cursor.fetchone() == (250,)
                    cursor.execute("SELECT pg_has_role(current_user, 'neon_superuser', 'member')")
                    assert cursor.fetchone() == (False,)
                    with pytest.raises(psycopg.errors.InsufficientPrivilege):
                        cursor.execute("UPDATE catalog.products SET name = name WHERE false")
                    with pytest.raises(psycopg.errors.InsufficientPrivilege):
                        cursor.execute("CREATE TABLE commerce.runtime_forbidden (id integer)")

            smoke = asyncio.run(
                run_load_smoke(
                    primary.runtime_database_url,
                    DATA_DIR,
                    concurrent_groups=16,
                    group_size=8,
                )
            )
            assert smoke == {"concurrent_groups": 16, "group_size": 8, "rollouts": 128}

            parent_admin_url = api.connection_uri(
                manifest,
                branch_id=manifest.parent_branch_id,
                role_name=manifest.admin_role_name,
                pooled=False,
            )
            _cannot_connect(retarget_runtime_url(primary.runtime_database_url, parent_admin_url))
            _cannot_connect(
                retarget_runtime_url(primary.runtime_database_url, sibling.admin_database_url)
            )

            old_runtime_url = primary.runtime_database_url
            api.delete_branch(manifest.project_id, primary.branch_id)
            primary_deleted = True
            _cannot_connect(old_runtime_url)
        finally:
            if primary is not None and not primary_deleted:
                api.delete_branch(manifest.project_id, primary.branch_id)
            if sibling is not None:
                api.delete_branch(manifest.project_id, sibling.branch_id)
