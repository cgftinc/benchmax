"""Secret-safe Neon project and disposable branch lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shlex
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx
import sqlalchemy as sa
from psycopg import sql
from sqlalchemy.ext.asyncio import create_async_engine

from order_resolution.database import validate_database_url
from order_resolution.fixtures import seed_immutable_catalog

NEON_API_BASE = "https://console.neon.tech/api/v2"
PROJECT_NAME = "order-resolution-neon-mvp"
PROJECT_ORG_ID = "org-tiny-art-93053842"
PROJECT_REGION = "aws-us-west-2"
PROJECT_DATABASE = "order_resolution"
PROJECT_ADMIN_ROLE = "order_resolution_owner"
PROJECT_PG_VERSION = 17
PROJECT_LIFETIME = timedelta(days=7)
CHILD_LIFETIME = timedelta(hours=24)
EXPERIMENT_COST_CAP_USD = 1.0
RUNTIME_ROLE_PREFIX = "order_runtime_"
_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")


class NeonApiError(RuntimeError):
    """A redacted Neon control-plane request failure."""

    def __init__(self, method: str, path: str, status_code: int | None, message: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        super().__init__(
            f"neon api {method} {path} failed"
            + (f" ({status_code})" if status_code is not None else "")
            + f": {message}"
        )


@dataclass(frozen=True, slots=True)
class ProjectManifest:
    """Non-secret identifiers and teardown ownership for the dedicated project."""

    project_id: str
    project_name: str
    org_id: str
    region_id: str
    parent_branch_id: str
    parent_branch_name: str
    parent_endpoint_id: str
    database_name: str
    admin_role_name: str
    lifecycle_owner: str
    created_at: str
    delete_after: str
    cost_cap_usd: float


@dataclass(frozen=True, slots=True)
class RuntimeBranch:
    """A child branch whose secret URLs exist only in process memory."""

    project_id: str
    parent_branch_id: str
    branch_id: str
    branch_name: str
    endpoint_id: str
    expires_at: str
    database_name: str
    runtime_role_name: str
    admin_database_url: str = field(repr=False)
    runtime_database_url: str = field(repr=False)


class NeonApi:
    """Narrow synchronous Neon API client with redacted failures."""

    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        if not api_key.strip():
            raise ValueError("NEON_API_KEY is required")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(60),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> NeonApi:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        method = method.upper()
        for attempt in range(2):
            try:
                response = self._client.request(
                    method,
                    f"{NEON_API_BASE}{path}",
                    params=params,
                    json=body,
                )
            except httpx.RequestError as error:
                if method == "GET" and attempt == 0:
                    continue
                raise NeonApiError(method, path, None, type(error).__name__) from error
            if response.status_code in {423, 503} and attempt == 0:
                time.sleep(0.5)
                continue
            if response.status_code >= 400:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                message = str(payload.get("message") or payload.get("code") or "request rejected")
                raise NeonApiError(method, path, response.status_code, message)
            if response.status_code == 204 or not response.content:
                return {}
            payload = response.json()
            if not isinstance(payload, dict):
                raise NeonApiError(method, path, response.status_code, "non-object response")
            return payload
        raise AssertionError("unreachable Neon request retry state")

    def ensure_project(self, *, owner: str, now: datetime | None = None) -> ProjectManifest:
        projects = self.request(
            "GET", "/projects", params={"search": PROJECT_NAME, "limit": 100}
        ).get("projects", [])
        exact = [project for project in projects if project.get("name") == PROJECT_NAME]
        if len(exact) > 1:
            raise RuntimeError(f"multiple Neon projects are named {PROJECT_NAME!r}")
        if exact:
            project = exact[0]
            self._validate_project(project)
            return self._manifest_for_project(project, owner=owner)

        created_at = now or datetime.now(UTC)
        body = {
            "project": {
                "name": PROJECT_NAME,
                "org_id": PROJECT_ORG_ID,
                "region_id": PROJECT_REGION,
                "pg_version": PROJECT_PG_VERSION,
                "history_retention_seconds": 86_400,
                "store_passwords": True,
                "default_endpoint_settings": {
                    "autoscaling_limit_min_cu": 0.25,
                    "autoscaling_limit_max_cu": 1,
                    "suspend_timeout_seconds": 300,
                },
                "branch": {
                    "name": "main",
                    "database_name": PROJECT_DATABASE,
                    "role_name": PROJECT_ADMIN_ROLE,
                    "annotations": {
                        "owner": owner,
                        "purpose": PROJECT_NAME,
                        "delete_after": _timestamp(created_at + PROJECT_LIFETIME),
                    },
                },
            }
        }
        try:
            response = self.request("POST", "/projects", body=body)
            self.wait_operations(response.get("operations", []))
            project = response["project"]
        except NeonApiError as error:
            if error.status_code is not None:
                raise
            reconciled = self.request(
                "GET", "/projects", params={"search": PROJECT_NAME, "limit": 100}
            ).get("projects", [])
            matching = [item for item in reconciled if item.get("name") == PROJECT_NAME]
            if len(matching) != 1:
                raise
            project = matching[0]
        self._validate_project(project)
        return self._manifest_for_project(project, owner=owner)

    def create_runtime_branch(
        self,
        manifest: ProjectManifest,
        *,
        purpose: str,
        now: datetime | None = None,
    ) -> RuntimeBranch:
        created_at = now or datetime.now(UTC)
        expires_at = _timestamp(created_at + CHILD_LIFETIME)
        branch_name = f"{purpose}-{created_at:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        path = f"/projects/{manifest.project_id}/branches"
        body = {
            "branch": {
                "parent_id": manifest.parent_branch_id,
                "name": branch_name,
                "expires_at": expires_at,
                "protected": False,
            },
            "endpoints": [
                {
                    "type": "read_write",
                    "autoscaling_limit_min_cu": 0.25,
                    "autoscaling_limit_max_cu": 1,
                    "suspend_timeout_seconds": 300,
                }
            ],
        }
        try:
            response = self.request("POST", path, body=body)
        except NeonApiError as error:
            if error.status_code is not None:
                raise
            matching = self.request("GET", path, params={"search": branch_name, "limit": 100}).get(
                "branches", []
            )
            exact = [branch for branch in matching if branch.get("name") == branch_name]
            if len(exact) != 1:
                raise
            response = {"branch": exact[0]}
        self.wait_operations(response.get("operations", []))
        branch = response["branch"]
        branch = self.wait_branch_ready(manifest.project_id, branch["id"])
        endpoints = response.get("endpoints", [])
        if not endpoints:
            endpoints = self.request(
                "GET",
                f"/projects/{manifest.project_id}/branches/{branch['id']}/endpoints",
            ).get("endpoints", [])
        endpoint = next(item for item in endpoints if item.get("type") == "read_write")
        admin_url = self.connection_uri(
            manifest,
            branch_id=branch["id"],
            role_name=manifest.admin_role_name,
            pooled=False,
        )
        pooled_admin_url = self.connection_uri(
            manifest,
            branch_id=branch["id"],
            role_name=manifest.admin_role_name,
            pooled=True,
        )
        try:
            role_name, password = create_runtime_role(admin_url)
        except BaseException:
            self.delete_branch(manifest.project_id, branch["id"])
            raise
        return RuntimeBranch(
            project_id=manifest.project_id,
            parent_branch_id=manifest.parent_branch_id,
            branch_id=branch["id"],
            branch_name=branch["name"],
            endpoint_id=endpoint["id"],
            expires_at=str(branch.get("expires_at") or expires_at),
            database_name=manifest.database_name,
            runtime_role_name=role_name,
            admin_database_url=admin_url,
            runtime_database_url=_role_url(pooled_admin_url, role_name, password, pooled=True),
        )

    def connection_uri(
        self,
        manifest: ProjectManifest,
        *,
        branch_id: str,
        role_name: str,
        pooled: bool,
    ) -> str:
        response = self.request(
            "GET",
            f"/projects/{manifest.project_id}/connection_uri",
            params={
                "branch_id": branch_id,
                "database_name": manifest.database_name,
                "role_name": role_name,
                "pooled": str(pooled).lower(),
            },
        )
        uri = response.get("uri")
        if not isinstance(uri, str):
            raise RuntimeError("Neon connection URI response is missing uri")
        return uri

    def delete_branch(self, project_id: str, branch_id: str) -> None:
        path = f"/projects/{project_id}/branches/{branch_id}"
        try:
            response = self.request(
                "DELETE",
                path,
                params={"hard_delete": "true"},
            )
        except NeonApiError as error:
            if error.status_code is not None:
                raise
            response = {}
        self.wait_operations(response.get("operations", []))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                self.request("GET", path)
            except NeonApiError as error:
                if error.status_code == 404:
                    return
                raise
            time.sleep(0.5)
        raise TimeoutError(f"Neon branch {branch_id} was not deleted within 120 seconds")

    def wait_operations(self, operations: list[dict[str, Any]]) -> None:
        for operation in operations:
            operation_id = operation.get("id")
            project_id = operation.get("project_id")
            if not operation_id or not project_id:
                continue
            current = operation
            deadline = time.monotonic() + 120
            while current.get("status") not in {"finished", "failed", "cancelled"}:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Neon operation {operation_id} exceeded 120 seconds")
                time.sleep(0.5)
                current = self.request(
                    "GET", f"/projects/{project_id}/operations/{operation_id}"
                ).get("operation", {})
            if current.get("status") != "finished":
                raise RuntimeError(
                    f"Neon operation {operation_id} ended as {current.get('status', 'unknown')}"
                )

    def wait_branch_ready(self, project_id: str, branch_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 120
        while True:
            branch = self.request("GET", f"/projects/{project_id}/branches/{branch_id}").get(
                "branch", {}
            )
            state = branch.get("current_state")
            if state == "ready":
                return branch
            if state in {"error", "deleting"}:
                raise RuntimeError(f"Neon branch {branch_id} entered {state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Neon branch {branch_id} was not ready within 120 seconds")
            time.sleep(0.5)

    def _manifest_for_project(
        self,
        project: dict[str, Any],
        *,
        owner: str,
    ) -> ProjectManifest:
        project_id = str(project["id"])
        branches = self.request("GET", f"/projects/{project_id}/branches").get("branches", [])
        parent = next(branch for branch in branches if branch.get("default"))
        endpoints = self.request("GET", f"/projects/{project_id}/endpoints").get("endpoints", [])
        endpoint = next(item for item in endpoints if item.get("branch_id") == parent["id"])
        databases = self.request(
            "GET", f"/projects/{project_id}/branches/{parent['id']}/databases"
        ).get("databases", [])
        database = next(item for item in databases if item.get("name") == PROJECT_DATABASE)
        created_at = datetime.fromisoformat(str(project["created_at"]).replace("Z", "+00:00"))
        delete_after = created_at + PROJECT_LIFETIME
        return ProjectManifest(
            project_id=project_id,
            project_name=str(project["name"]),
            org_id=str(project["org_id"]),
            region_id=str(project["region_id"]),
            parent_branch_id=str(parent["id"]),
            parent_branch_name=str(parent["name"]),
            parent_endpoint_id=str(endpoint["id"]),
            database_name=str(database["name"]),
            admin_role_name=str(database["owner_name"]),
            lifecycle_owner=owner,
            created_at=_timestamp(created_at),
            delete_after=_timestamp(delete_after),
            cost_cap_usd=EXPERIMENT_COST_CAP_USD,
        )

    @staticmethod
    def _validate_project(project: dict[str, Any]) -> None:
        expected = {
            "name": PROJECT_NAME,
            "org_id": PROJECT_ORG_ID,
            "region_id": PROJECT_REGION,
            "pg_version": PROJECT_PG_VERSION,
        }
        mismatches = [key for key, value in expected.items() if project.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "existing Neon project does not match approved configuration: "
                + ", ".join(sorted(mismatches))
            )


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a strict shell-style env file without executing it."""

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, raw_value = line.partition("=")
        if not separator or not _ENV_KEY.fullmatch(key):
            raise RuntimeError(f"invalid entry in {path.name} on line {line_number}")
        parsed = shlex.split(raw_value, posix=True)
        if len(parsed) != 1:
            raise RuntimeError(f"invalid value in {path.name} on line {line_number}")
        values[key] = parsed[0]
    return values


def resolve_neon_api_key(path: Path) -> str:
    value = os.environ.get("NEON_API_KEY", "").strip()
    if not value and path.is_file():
        value = read_env_file(path).get("NEON_API_KEY", "").strip()
    if not value:
        raise RuntimeError(f"NEON_API_KEY is required in the environment or {path}")
    return value


def write_project_manifest(path: Path, manifest: ProjectManifest) -> None:
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    if "postgresql://" in payload or "NEON_API_KEY" in payload:
        raise RuntimeError("refusing to write a secret-bearing Neon manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def read_project_manifest(path: Path) -> ProjectManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProjectManifest(**payload)


async def seed_parent_catalog(admin_database_url: str) -> None:
    validate_database_url(admin_database_url, purpose="admin")
    engine = create_async_engine(
        _async_url(admin_database_url),
        poolclass=sa.pool.NullPool,
        connect_args={"prepare_threshold": None},
    )
    try:
        async with engine.begin() as connection:
            await seed_immutable_catalog(connection)
            await connection.execute(
                sa.text("REVOKE CONNECT, TEMPORARY ON DATABASE order_resolution FROM PUBLIC")
            )
            await connection.execute(sa.text("REVOKE CREATE ON SCHEMA public FROM PUBLIC"))
    finally:
        await engine.dispose()


def create_runtime_role(admin_database_url: str) -> tuple[str, str]:
    """Create one child-only least-privilege login and return its ephemeral password."""

    import psycopg

    validate_database_url(admin_database_url, purpose="admin")
    role_name = f"{RUNTIME_ROLE_PREFIX}{secrets.token_hex(6)}"
    password = secrets.token_urlsafe(32)
    database_name = unquote(urlsplit(admin_database_url).path.lstrip("/"))
    with psycopg.connect(admin_database_url, autocommit=True, connect_timeout=60) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "CREATE ROLE {role} WITH LOGIN PASSWORD {password} "
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT"
                ).format(role=sql.Identifier(role_name), password=sql.Literal(password))
            )
            cursor.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {database} TO {role}").format(
                    database=sql.Identifier(database_name), role=sql.Identifier(role_name)
                )
            )
            for schema_name in ("catalog", "commerce", "bench"):
                cursor.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA {schema} TO {role}").format(
                        schema=sql.Identifier(schema_name), role=sql.Identifier(role_name)
                    )
                )
            cursor.execute(
                sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA catalog TO {role}").format(
                    role=sql.Identifier(role_name)
                )
            )
            for schema_name in ("commerce", "bench"):
                cursor.execute(
                    sql.SQL(
                        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                        "IN SCHEMA {schema} TO {role}"
                    ).format(schema=sql.Identifier(schema_name), role=sql.Identifier(role_name))
                )
    return role_name, password


def retarget_runtime_url(runtime_database_url: str, target_admin_url: str) -> str:
    """Apply child-only runtime credentials to another branch's direct endpoint."""

    runtime = urlsplit(runtime_database_url)
    target = urlsplit(target_admin_url)
    if runtime.username is None or runtime.password is None or target.hostname is None:
        raise ValueError("database URL is missing credentials or hostname")
    netloc = _netloc(runtime.username, runtime.password, target.hostname, target.port)
    return urlunsplit(("postgresql", netloc, target.path, target.query, ""))


def _role_url(admin_url: str, role_name: str, password: str, *, pooled: bool) -> str:
    parsed = urlsplit(admin_url)
    if parsed.hostname is None:
        raise ValueError("admin database URL is missing a hostname")
    hostname = parsed.hostname
    if pooled and "-pooler." not in hostname:
        first, separator, rest = hostname.partition(".")
        if not separator:
            raise ValueError("Neon hostname is not in the expected form")
        hostname = f"{first}-pooler.{rest}"
    netloc = _netloc(role_name, password, hostname, parsed.port)
    result = urlunsplit(("postgresql", netloc, parsed.path, parsed.query, ""))
    validate_database_url(result, purpose="runtime" if pooled else "admin")
    return result


def _netloc(username: str, password: str, hostname: str, port: int | None) -> str:
    authority = f"{quote(username, safe='')}:{quote(password, safe='')}@{hostname}"
    return f"{authority}:{port}" if port is not None else authority


def _async_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    return urlunsplit(("postgresql+psycopg", parsed.netloc, parsed.path, parsed.query, ""))


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def setup_project(
    *,
    api_key: str,
    manifest_path: Path,
    owner: str,
) -> tuple[ProjectManifest, str]:
    """Create/reconcile the dedicated project and return its direct parent URL in memory."""

    with NeonApi(api_key) as api:
        manifest = api.ensure_project(owner=owner)
        admin_url = api.connection_uri(
            manifest,
            branch_id=manifest.parent_branch_id,
            role_name=manifest.admin_role_name,
            pooled=False,
        )
    write_project_manifest(manifest_path, manifest)
    return manifest, admin_url


def run_seed_parent(admin_database_url: str) -> None:
    asyncio.run(seed_parent_catalog(admin_database_url))


async def sync_parent_v2_catalog(admin_database_url: str) -> str:
    """Reconcile the content-addressed v2 namespace in one transaction."""

    from order_resolution.fixtures import build_v2_catalog, sync_catalog_namespace

    validate_database_url(admin_database_url, purpose="admin")
    engine = create_async_engine(
        _async_url(admin_database_url),
        poolclass=sa.pool.NullPool,
        connect_args={"prepare_threshold": None},
    )
    try:
        async with engine.begin() as connection:
            return await sync_catalog_namespace(connection, build_v2_catalog())
    finally:
        await engine.dispose()


def run_sync_parent_v2_catalog(admin_database_url: str) -> str:
    return asyncio.run(sync_parent_v2_catalog(admin_database_url))


__all__ = [
    "CHILD_LIFETIME",
    "NeonApi",
    "NeonApiError",
    "ProjectManifest",
    "RuntimeBranch",
    "create_runtime_role",
    "read_env_file",
    "read_project_manifest",
    "resolve_neon_api_key",
    "retarget_runtime_url",
    "run_seed_parent",
    "seed_parent_catalog",
    "setup_project",
    "write_project_manifest",
]
