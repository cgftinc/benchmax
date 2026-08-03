"""Idempotent provisioning for the example's Neon Lakebase database.

Self-serve from a Neon API key and a project id — no console clicks. Running this
twice is safe: it never creates a second project, role, schema, or extension.

What it does, in order (see the runbook, ``README.md`` in this package):

1. **Preload libraries** — enable ``lakebase_vector`` + ``lakebase_text`` via the
   project-settings ``preload_libraries.enabled_libraries`` API. This REPLACES the
   list, so the existing/default libraries are read first and preserved, then the
   two lakebase libraries appended. (The raw ``shared_preload_libraries`` /
   ``neon.lakebase_mode`` GUCs are rejected by the API — the project setting is the
   supported path.)
2. **Apply** — restart the endpoint; an idle/suspended compute cannot be restarted
   (``endpoint is not active`` is expected, not an error) and simply picks up the
   config on its next wake, which the admin connect below triggers.
3. **Extensions** — the admin role (Neon ``neon_superuser``, e.g. ``neondb_owner``)
   runs ``CREATE EXTENSION ... CASCADE``. This is the ONLY step that installs
   extensions; it is kept split from the writer's DDL because a Neon API-created
   role has ``neon_superuser`` but SQL-created roles do NOT.
4. **Roles + schema + grants** — create the data-preparation role (owns the
   schema and every version table; does DDL + ingest) and the read-only role (schema ``USAGE`` +
   ``SELECT`` on current + future reader objects via the writer's default
   privileges). Passwords are reused from any database URLs already in the generated
   environment file so a
   re-run does not rotate live credentials.

The two provider surfaces are ``NEON_DATA_PREPARATION_DATABASE_URL`` and
``NEON_SEARCH_DATABASE_URL``. The setup script writes both to ``.env.neon`` and
never prints them, the API key, or the temporary admin connection.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass

from neon_backend.config import CORPUS_SCHEMA, RO_ROLE, WRITER_ROLE

NEON_API = "https://console.neon.tech/api/v2"

# The lakebase libraries this provider needs preloaded. lakebase_vector supplies
# the lakebase_ann access method + pgvector types; lakebase_text the lakebase_bm25
# access method + to_bm25query/<@>.
LAKEBASE_PRELOAD_LIBS = ("lakebase_vector", "lakebase_text")

# Extensions the admin installs (CASCADE pulls pgvector for lakebase_vector).
REQUIRED_EXTENSIONS = ("lakebase_vector", "lakebase_text")


@dataclass(frozen=True)
class ProvisionResult:
    """Outcome of a provisioning run.

    Args:
        project_id: The Neon project the sample DB lives in.
        data_preparation_database_url: Database URL used to build the corpus.
        search_database_url: Read-only database URL used by rollout search.
    """

    project_id: str
    data_preparation_database_url: str
    search_database_url: str


def _api(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    """Call the Neon REST API and return the decoded JSON (raises on HTTP error)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{NEON_API}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read() or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"neon api {method} {path} -> {exc.code}: {detail}") from exc


def ensure_preload_libraries(api_key: str, project_id: str) -> None:
    """Enable the lakebase preload libraries, preserving the existing set (idempotent).

    Reads the current ``enabled_libraries`` (falling back to the project's default
    set when unset), and PATCHes only if a lakebase library is missing — so a
    re-run is a no-op.
    """
    project = _api("GET", f"/projects/{project_id}", api_key)["project"]
    preload = project.get("settings", {}).get("preload_libraries", {})
    if preload.get("enabled_libraries"):
        enabled = list(preload["enabled_libraries"])
    else:
        available = _api("GET", f"/projects/{project_id}/available_preload_libraries", api_key)
        libs = available if isinstance(available, list) else available.get("libraries", [])
        enabled = [lib["library_name"] for lib in libs if lib.get("is_default")]
    missing = [lib for lib in LAKEBASE_PRELOAD_LIBS if lib not in enabled]
    if not missing:
        print("preload libraries already enabled")
        return
    enabled.extend(missing)
    _api(
        "PATCH",
        f"/projects/{project_id}",
        api_key,
        {"project": {"settings": {"preload_libraries": {"enabled_libraries": enabled}}}},
    )
    print(f"enabled preload libraries: {enabled}")


def apply_endpoint_config(api_key: str, project_id: str) -> None:
    """Restart the endpoint to apply preload config; tolerate an idle compute.

    A suspended compute rejects the restart (``endpoint is not active``); that is
    expected — waking it (the admin connect that follows) applies the config.
    """
    endpoints = _api("GET", f"/projects/{project_id}/endpoints", api_key)["endpoints"]
    for ep in endpoints:
        if ep["type"] != "read_write":
            continue
        try:
            _api("POST", f"/projects/{project_id}/endpoints/{ep['id']}/restart", api_key)
            print(f"restarted endpoint {ep['id']}")
        except RuntimeError as exc:
            if "not active" in str(exc):
                print(f"endpoint {ep['id']} idle — config applies on next wake")
            else:
                raise


def fetch_admin_dsn(api_key: str, project_id: str) -> str:
    """Return the admin (neon_superuser) DSN, preferring ``NEON_ADMIN_DSN`` from env.

    Falls back to the project's default branch/database/owner via the
    ``connection_uri`` API. Never printed by this module.
    """
    env = os.environ.get("NEON_ADMIN_DSN")
    if env:
        return env
    branches = _api("GET", f"/projects/{project_id}/branches", api_key)["branches"]
    default = next(b for b in branches if b.get("default"))
    dbs = _api("GET", f"/projects/{project_id}/branches/{default['id']}/databases", api_key)[
        "databases"
    ]
    db = dbs[0]
    uri = _api(
        "GET",
        f"/projects/{project_id}/connection_uri?branch_id={default['id']}"
        f"&database_name={db['name']}&role_name={db['owner_name']}&pooled=false",
        api_key,
    )["uri"]
    return uri


def _dsn_for_role(admin_dsn: str, role: str, password: str) -> str:
    """Build a role DSN from the admin DSN's host/db, keeping its SSL query tail."""
    m = re.match(r"postgresql://[^:]+:[^@]+@([^/?]+)/([^?]+)(\?.*)?", admin_dsn)
    if not m:
        raise ValueError("admin DSN is not in the expected postgresql:// form")
    host, db, tail = m.group(1), m.group(2), m.group(3) or "?sslmode=require"
    return f"postgresql://{role}:{password}@{host}/{db}{tail}"


def _existing_password(database_url: str | None) -> str | None:
    """Extract a password from an existing generated database URL, if present."""
    if not database_url:
        return None
    m = re.match(r"postgresql://[^:]+:([^@]+)@", database_url)
    return m.group(1) if m else None


def ensure_extensions_roles_schema(admin_dsn: str, writer_pw: str, ro_pw: str) -> None:
    """Install extensions (admin) and create the writer/ro roles + schema + grants.

    Idempotent: roles/schema use existence guards; passwords are (re)set to the
    supplied values; grants are re-issued harmlessly. The RO role gets schema
    ``USAGE``, ``SELECT`` on every current table/view (``GRANT SELECT ON ALL
    TABLES``), and — via the writer's default privileges — ``SELECT`` on every
    future table/view the writer creates, so a new corpus version is readable
    without a manual grant.

    The provisioner grants the admin writer membership while configuring ownership
    and default privileges, then revokes it. Neon does not synchronously grant new
    role membership to ``neon_superuser``, and that role cannot bypass ownership
    checks. The read-only role is never granted to the admin.
    """
    import psycopg
    from psycopg import sql

    conn = psycopg.connect(admin_dsn, connect_timeout=60, autocommit=True)
    cur = conn.cursor()
    for ext in REQUIRED_EXTENSIONS:
        statement = sql.SQL("CREATE EXTENSION IF NOT EXISTS {} CASCADE").format(sql.Identifier(ext))
        cur.execute(statement)

    for role, pw in ((WRITER_ROLE, writer_pw), (RO_ROLE, ro_pw)):
        cur.execute(
            sql.SQL(
                "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = {name}) "
                "THEN CREATE ROLE {role} LOGIN; END IF; END $$"
            ).format(name=sql.Literal(role), role=sql.Identifier(role))
        )
        cur.execute(
            sql.SQL("ALTER ROLE {role} LOGIN PASSWORD {pw}").format(
                role=sql.Identifier(role), pw=sql.Literal(pw)
            )
        )

    schema = sql.Identifier(CORPUS_SCHEMA)
    writer = sql.Identifier(WRITER_ROLE)
    ro = sql.Identifier(RO_ROLE)

    # Temporary writer membership: required to CREATE SCHEMA ... AUTHORIZATION the
    # writer, ALTER its DEFAULT PRIVILEGES, and GRANT on its owned tables.
    cur.execute(sql.SQL("GRANT {writer} TO CURRENT_USER").format(writer=writer))
    try:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION {writer}").format(
                schema=schema, writer=writer
            )
        )
        # Resolve unqualified identifiers the client emits (view/table names)
        # against the corpus schema, with public for the extension types/operators.
        for role_id in (writer, ro):
            cur.execute(
                sql.SQL("ALTER ROLE {role} SET search_path = {schema}, public").format(
                    role=role_id, schema=schema
                )
            )
        cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {schema} TO {ro}").format(schema=schema, ro=ro))
        # Future writer-created tables AND views (relkind r/v are both "TABLES" for
        # default privileges) become RO-readable automatically.
        cur.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {writer} IN SCHEMA {schema} "
                "GRANT SELECT ON TABLES TO {ro}"
            ).format(writer=writer, schema=schema, ro=ro)
        )
        # Repair privileges on any tables/views that ALREADY exist (default
        # privileges only cover future objects); a no-op on a fresh schema.
        cur.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {ro}").format(
                schema=schema, ro=ro
            )
        )
    finally:
        cur.execute(sql.SQL("REVOKE {writer} FROM CURRENT_USER").format(writer=writer))
    conn.close()
    print(f"ensured extensions + roles ({WRITER_ROLE}, {RO_ROLE}) + schema {CORPUS_SCHEMA}")


def provision(
    api_key: str,
    project_id: str,
    *,
    existing_data_preparation_database_url: str | None = None,
    existing_search_database_url: str | None = None,
) -> ProvisionResult:
    """Run the idempotent provisioning flow and return restricted database URLs."""
    ensure_preload_libraries(api_key, project_id)
    apply_endpoint_config(api_key, project_id)
    admin_dsn = fetch_admin_dsn(api_key, project_id)
    writer_pw = _existing_password(existing_data_preparation_database_url) or secrets.token_hex(16)
    ro_pw = _existing_password(existing_search_database_url) or secrets.token_hex(16)
    ensure_extensions_roles_schema(admin_dsn, writer_pw, ro_pw)
    return ProvisionResult(
        project_id=project_id,
        data_preparation_database_url=_dsn_for_role(admin_dsn, WRITER_ROLE, writer_pw),
        search_database_url=_dsn_for_role(admin_dsn, RO_ROLE, ro_pw),
    )


def write_env_file(path: str, updates: dict[str, str]) -> None:
    """Merge *updates* into the KEY="VALUE" env file at *path*, enforcing mode 0600.

    Atomic (temp file + ``os.replace``) and secret-safe: the temp file is created
    with :func:`tempfile.mkstemp` in the DESTINATION directory (a fresh random name,
    ``O_EXCL`` + 0600 — so a symlink/pre-create at a predictable ``.tmp`` path can
    never redirect or truncate the write), forced 0600, flushed + ``fsync``ed, then
    ``os.replace``d over the target (atomic on the same filesystem). Values are
    double-quoted (a Neon DSN contains ``&``, which unquoted breaks ``source``).
    Existing keys are updated in place; unrelated lines are preserved. The
    function never logs the values it writes.
    """
    resolved = os.path.expanduser(path)
    lines: list[str] = []
    seen: set[str] = set()
    if os.path.exists(resolved):
        with open(resolved) as existing_file:
            existing_lines = existing_file.read().splitlines()
        for line in existing_lines:
            m = re.match(r"^([A-Z_]+)=", line)
            if m and m.group(1) in updates:
                seen.add(m.group(1))
                lines.append(f'{m.group(1)}="{updates[m.group(1)]}"')
            else:
                lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f'{key}="{value}"')
    dest_dir = os.path.dirname(resolved) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".neon-env-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)  # enforce 0600 on the fd regardless of umask
        with os.fdopen(fd, "w") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, resolved)
    except BaseException:
        # never leave a secret-bearing temp file behind on failure
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    os.chmod(resolved, 0o600)
