"""Secret-boundary and URL tests for Neon lifecycle helpers."""

from __future__ import annotations

from order_resolution.branching import (
    ProjectManifest,
    RuntimeBranch,
    read_env_file,
    retarget_runtime_url,
    write_project_manifest,
)
from order_resolution.hosting import (
    CANONICAL_RUNTIME_DEPENDENCIES,
    build_environment_bundle,
    inspect_environment_bundle,
)


def test_env_file_is_parsed_without_execution(tmp_path) -> None:
    env_file = tmp_path / ".neon.env"
    env_file.write_text(
        'NEON_API_KEY="secret-value"\nexport NEON_PROJECT_ID=test-project\n',
        encoding="utf-8",
    )

    assert read_env_file(env_file) == {
        "NEON_API_KEY": "secret-value",
        "NEON_PROJECT_ID": "test-project",
    }


def test_manifest_contains_identifiers_but_no_credentials(tmp_path) -> None:
    manifest = ProjectManifest(
        project_id="test-project",
        project_name="order-resolution-neon-mvp",
        org_id="test-org",
        region_id="aws-us-west-2",
        parent_branch_id="br-parent",
        parent_branch_name="main",
        parent_endpoint_id="ep-parent",
        database_name="order_resolution",
        admin_role_name="order_resolution_owner",
        lifecycle_owner="Angel",
        created_at="2026-08-05T12:00:00Z",
        delete_after="2026-08-12T12:00:00Z",
        cost_cap_usd=1.0,
    )
    path = tmp_path / "neon.json"

    write_project_manifest(path, manifest)

    payload = path.read_text(encoding="utf-8")
    assert "test-project" in payload
    assert "postgresql://" not in payload
    assert "NEON_API_KEY" not in payload


def test_runtime_branch_repr_hides_database_urls() -> None:
    branch = RuntimeBranch(
        project_id="test-project",
        parent_branch_id="br-parent",
        branch_id="br-child",
        branch_name="validation",
        endpoint_id="ep-child",
        expires_at="2026-08-06T12:00:00Z",
        database_name="order_resolution",
        runtime_role_name="order_runtime_test",
        admin_database_url="postgresql://admin:secret@ep-child.us.neon.tech/db",
        runtime_database_url=("postgresql://runtime:secret@ep-child-pooler.us.neon.tech/db"),
    )

    assert "postgresql://" not in repr(branch)
    assert "secret" not in repr(branch)


def test_runtime_credentials_can_be_retargeted_without_changing_database() -> None:
    runtime = (
        "postgresql://order_runtime:runtime-secret@"
        "ep-child-pooler.us-west-2.aws.neon.tech/order_resolution?sslmode=require"
    )
    target = (
        "postgresql://admin:admin-secret@"
        "ep-parent.us-west-2.aws.neon.tech/order_resolution?sslmode=require"
    )

    retargeted = retarget_runtime_url(runtime, target)

    assert retargeted == (
        "postgresql://order_runtime:runtime-secret@"
        "ep-parent.us-west-2.aws.neon.tech/order_resolution?sslmode=require"
    )


def test_bundle_contains_only_the_disposable_runtime_secret() -> None:
    runtime_url = (
        "postgresql://order_runtime:runtime-secret@"
        "ep-child-pooler.us-west-2.aws.neon.tech/order_resolution?sslmode=require"
    )
    bundle = build_environment_bundle(runtime_url)

    inspection = inspect_environment_bundle(
        bundle,
        runtime_database_url=runtime_url,
        forbidden_secrets={
            "admin_url": (
                "postgresql://admin:admin-secret@ep-child.us-west-2.aws.neon.tech/order_resolution"
            ),
            "api_key": "neon-api-secret",
        },
    )

    assert inspection["pip_dependencies"] == list(CANONICAL_RUNTIME_DEPENDENCIES)
    assert inspection["python_version"] == "3.12"
    assert inspection["benchmax_version"] == "0.2.3"
    assert inspection["secret_boundary"] == "ok"
    assert len(inspection["digest"]) == 64
