# neon lakebase corpus — provisioning runbook

recreate the sample database (vector + bm25 + filter search) on a neon project from
just a neon api key. all steps are idempotent; re-running never duplicates a
project, role, schema, or extension.

## prerequisites

- a neon **project id** and an **api key** scoped to it (a project-scoped key is
  enough — no account/org key needed).
- the `neon` extra installed: `uv sync --extra neon`.
- the lakebase extensions must be available to the project
  (`GET /projects/{id}/available_preload_libraries` lists `lakebase_vector` +
  `lakebase_text`). lakebase search is generally available on postgres 16+.

## 1. set the developer-local env

credentials live ONLY in `~/.config/neon-benchmax.env` (never committed). start
with:

```
NEON_API_KEY="<your project-scoped neon api key>"
NEON_PROJECT_ID="<your project id>"
```

quote values — a neon dsn contains `&`, which unquoted breaks `source`.

## 2. provision (preload libs + extensions + roles + schema + grants)

```
set -a; source ~/.config/neon-benchmax.env; set +a
uv run --extra neon python -m castform.rag.corpus.neon.provision
```

this: enables the `lakebase_vector`+`lakebase_text` preload libraries via the
project setting `preload_libraries.enabled_libraries` (read-then-merge — the
existing/default libraries are preserved); restarts/wakes the endpoint to apply it;
installs the extensions as the admin (`neon_superuser`) role; and creates the two
non-superuser roles + the `benchmax_corpus` schema with explicit grants:

- **admin** — installs extensions only.
- **writer** (`benchmax_writer`) — owns the schema + every version table; does DDL
  + ingest. surface env var `NEON_CORPUS_DSN_RW`.
- **read-only** (`benchmax_ro`) — schema `USAGE` + `SELECT` on current and future
  writer-created tables/views (via `ALTER DEFAULT PRIVILEGES`). surface env var
  `NEON_CORPUS_DSN_RO`.

it writes `NEON_PROJECT_ID` + `NEON_CORPUS_DSN_RW` + `NEON_CORPUS_DSN_RO` directly
into the env file (`NEON_BENCHMAX_ENV_FILE`, default `~/.config/neon-benchmax.env`)
at mode `0600`, quoted — the credential-bearing dsns are **never printed or
logged**; stdout is a secret-free confirmation only. re-source the file afterwards:

```
set -a; source ~/.config/neon-benchmax.env; set +a
```

re-running provision reuses the passwords already present in the dsns, so live
credentials are not rotated. (if you keep `NEON_ADMIN_DSN` in the file, provision
uses it; otherwise it fetches the admin dsn from the api each run.)

## 3. load the tiny smoke fixture + verify

the smoke fixture (`sample_fixture.py`) is a handful of rows with known relevance
ordering — built as the writer, then queried under the read-only role. the
integration test both loads it and asserts correctness:

```
set -a; source ~/.config/neon-benchmax.env; set +a
uv run --extra neon python -m pytest -m integration \
  packages/castform/tests/integration/rag/corpus/neon/test_live_smoke.py
```

it proves one vector query, one bm25 query (negative-score, ascending = best-first),
and one metadata-filtered query return correctly-ordered rows under
`NEON_CORPUS_DSN_RO`, and that the read-only role cannot write.

to load the fixture without the test (e.g. to inspect it by hand):

```python
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon import sample_fixture as sf
import os
sf.load_smoke_corpus(NeonClient(lambda: os.environ["NEON_CORPUS_DSN_RW"]))
```

## notes

- **scale-to-zero**: the first connection wakes a suspended compute (cold start,
  a few seconds); the client retries a dead connection once, and the test wraps
  the build in a bounded connect-retry.
- **proven DDL**: the frozen access methods / opclasses / operators / casts are in
  `CONTRACT.md` (§PROVEN ON LIVE NEON) and `schema.PROVEN_ANN_DDL`.
- a project-scoped key confines everything to that one project, so the sample DB
  lives in its `benchmax_corpus` schema there.
