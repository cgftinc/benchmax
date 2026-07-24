# neon corpus provider — frozen contracts (slice a)

design-lock for the neon lakebase (postgres + pgvector + bm25) corpus provider.
this slice ships typed stubs, this doc, and parametrized test skeletons only. all
real sql/client/filter/search logic lands in slices 1/2/4; live verification of
the ann access method + index EXPLAIN landed in slice 3 (see PROVEN ON LIVE NEON
below — the previously-provisional ann/bm25/gin DDL is now frozen from empirical
results against a live neon lakebase compute).

> convention note: the existing `corpus/` tests are class-grouped with fake
> injection and no `parametrize`/`xfail`. this slice introduces parametrized
> skeletons marked `xfail(raises=NotImplementedError, strict=True)` for the
> not-yet-built sql, so they xfail cleanly (no silent XPASS) and flip to failing
> the moment a stub is implemented incorrectly. flagged as a new local convention.

## 1. physical table + versioned-replace lifecycle

managed physical table (per version):

| column | type | notes |
|---|---|---|
| `id` | `text primary key` | chunk hash (sha256 hexdigest) |
| `content` | `text not null` | |
| `metadata` | `jsonb not null default '{}'` | |
| `embedding` | `vector(3072)` | PROVEN §PROVEN ON LIVE NEON; lakebase_ann indexable at 3072 |
| `source_file` | `text not null` | typed scan-order column (B6) |
| `chunk_index` | `integer not null` | typed scan-order column (B6) |
| `content_tsv` | `tsvector generated always as (to_tsvector(<config>::regconfig, content)) stored` | config allowlisted, baked per version |

- **logical vs physical**: readers address a stable *logical name* via an
  owner-rights view (`security_invoker = false`) with an **explicit column list**
  (never `SELECT *`). each ingest builds `<logical>__v<N>` with its own indexes.
- **per-version ledger** `neon_corpus_versions(logical_name, version, state,
  is_current, created_at, ready_at, activated_at, retired_at)`. `state` has a
  **DB CHECK domain** `building -> ready -> activated -> retired`
  (`VERSION_STATE_TRANSITIONS` frozen for Slice 2 enforcement).
- **current-pointer invariant**: `state='activated'` is *historical* (published at
  least once); `is_current` marks the currently-published version. a **partial
  unique index** `ON neon_corpus_versions (logical_name) WHERE is_current`
  (`CREATE_CURRENT_POINTER_INDEX`) enforces **at-most-one** current row per logical
  name (zero is legal — e.g. before the first publish). a SUCCESSFUL atomic
  activation establishes exactly-one for a published corpus; DB-enforced
  *existence* of a current row, if ever required, needs a separate current-pointer
  structure (out of scope now). rollback flips `is_current` between two
  `activated` versions without changing state.
- **lifecycle-SQL owner**: the per-version table/index/view/activation SQL is
  composed and executed in `NeonClient` (`client.py`), reusing the validated
  helpers + shared ledger DDL constants frozen in `schema.py`. activate/rollback
  are execute-and-check flows (`RETURNING`-guarded row-count checks that abort the
  transaction with `VersionStateError` before publishing), not static composable
  lists, so they live in the client rather than as `schema.py` builders.
- **atomic activate** (`NeonClient.activate(spec, grant)`): one transaction under
  `pg_advisory_xact_lock(logical)` clears the prior `is_current`, sets this
  version `activated`/`is_current` (guarded `state='ready'` + `RETURNING`),
  `create or replace view`, AND issues the RO grants from `grant` — commit or roll
  back together. **rollback** (`NeonClient.rollback(logical, version)`): re-point
  `is_current` to a prior `activated` version (row-locked `FOR UPDATE` first); old
  physical tables retained → O(1), non-destructive.
- **retention** (`RetentionPolicy`, self-validating): keep >= 2 activated
  (rollback always has a target) + >= 1 ready. the **pruning seam** and full
  concurrent allocation/prune race-safety are **deferred to Slice 2** (where the
  real DDL/transactions land); the invariant + locking contract are frozen now.
- **RO grants** (`ReadGrantSpec`): the vector + filter paths read purely through
  the owner-rights view (`security_invoker = false`), so `SELECT` on the view is
  all they need — never a physical-table grant. issued on FIRST view creation
  (`create or replace` preserves an existing ACL but a first create has none),
  which is why activation carries `grant`. **exception — bm25** (proven slice 3,
  see §PROVEN ON LIVE NEON): `to_bm25query` runs with the RO *invoker's* rights
  (not the view owner's) and reads the bm25 index's base-table stats, so RO ALSO
  needs `SELECT` on the version tables — granted narrowly via the writer's `ALTER
  DEFAULT PRIVILEGES` + `GRANT SELECT ON ALL TABLES` (see `provision.py`), NOT any
  write/DDL privilege. so "RO never touches physical tables" holds for the view
  reads; bm25 is the one read that needs the base-table `SELECT`.
- **view identifier policy** (B4): the reader-facing view name is
  `view_name(logical_name)` — validated printable-ASCII and length-fitted to 63
  bytes (a long logical name is hash-fitted, same as physical names); readers
  resolve it through `view_name`, never assume it equals the raw logical name.
- indexes per version: `ann` (`lakebase_ann`, PROVEN), `bm25` (`lakebase_bm25`,
  PROVEN lexical), `meta_gin` (`jsonb_path_ops`, serves `@>`, PROVEN), `scan`
  (btree `(source_file, chunk_index, id)`), `tsv_gin` (native fts fallback).
- **injection-safe DDL** (B4): all identifiers via `psycopg.sql.Identifier`;
  regconfig via allowlist + bound `sql.Literal`; version numbers validated
  (`validate_version`, positive int, rejects bool); logical names validated
  (`validate_logical_name`, printable ASCII); names **byte-safe** to 63 bytes with
  a 64-bit content-hash suffix (`_fit_identifier` — collision-resistant, not a
  uniqueness guarantee); the execute seam accepts `sql.Composable`, never `str`.

## 2. credential constructor signature

`dsn_provider: str | TokenProvider | None`, resolved lazily, mirroring
turbopuffer's `as_token_provider`/`env_token`. **separate read vs write
surfaces** — a single provider can't be both rw-ingest and ro-search:

- read: `resolve_read_dsn_provider(...)` -> default `env_token("NEON_CORPUS_DSN_RO")`, select-only.
- write: `resolve_write_dsn_provider(...)` -> default `env_token("NEON_CORPUS_DSN_RW")`, ddl+dml.

the sandbox rollout env only ever receives the ro provider. seam built in slice 4.

## 3. 9-op filter truth table (type-directed, indexable)

operators: `eq, ne, in, gt, gte, lt, lte, contains_any, contains_all` (shared six
+ `ne/gt/lt` in a neon-local superset; promotion to the shared enum deferred to
slice 4). metadata key + every value are **bound params** (`%(k)s`/`%(v)s`);
`psycopg.sql.Identifier` is never used on caller keys.

three safety properties are frozen (they drove the review):

- **type-directed, never-throwing**: `eq/ne/in` and `contains_*` emit JSONB
  **containment** (`metadata @> jsonb_build_object(...)`), type-aware, no cast. the
  range ops are the only cast path, and the cast lives **inside a CASE** (`CASE
  WHEN jsonb_typeof(...) = 'number' THEN (...)::numeric ... ELSE NULL END`) — NOT
  behind `AND`, because Postgres does not guarantee `AND` short-circuits, whereas
  a CASE only evaluates the matching branch, so `::numeric` never sees a
  non-number.
- **correct negation**: bare containment is two-valued (FALSE for missing/null/
  wrong-type), so `not(false)` would wrongly INCLUDE those rows. every op therefore
  also carries a **three-valued `negated_leaf_sql`** (`CASE WHEN NOT
  jsonb_exists(metadata, key) THEN NULL WHEN jsonb_typeof(...) <> '<type>' THEN
  NULL ELSE <containment> END`); `NotPredicate` wraps *that* in `not (...)`, so
  `not(null) = null` keeps missing/null/wrong-type excluded. key existence uses
  `jsonb_exists(metadata, %(k)s)` (function form; the `?` operator collides with
  psycopg placeholders).
- **indexable positives**: positive containment is served by the `meta_gin`
  `jsonb_path_ops` GIN (the `?|`/`?&` forms are rejected — a whole-doc GIN cannot
  serve them). negated CASE leaves and range CASEs are not GIN-eligible; neither is
  empty-array `contains_all` (`@> '[]'` has no scalar token), so its
  `empty_operand_indexable` is False (B3 caveat).

canonical POSITIVE sql (numeric value shown for eq/ne/in; text for contains; the
cast token follows the value type — `CONTAINS_ATOM_BY_TYPE` freezes text/number/
boolean shapes). the metadata **key** param is cast `%(k)s::text` inside every
`jsonb_build_object` (which is VARIADIC `"any"` and cannot infer a bound param's
type — an uncast key raises `IndeterminateDatatype` on live Postgres, proven slice-4
live; the range ops' `metadata -> %(k)s` resolves the param to text on its own):

| op | family | positive sql | indexable |
|---|---|---|---|
| eq | containment | `metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))` | yes |
| ne | negated containment | `(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))) IS NOT TRUE` | no |
| in | containment OR | `(metadata @> …%(v0)s…) OR (metadata @> …%(v1)s…)` | yes |
| gt/gte/lt/lte | range CASE | `CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' THEN (metadata ->> %(k)s)::numeric {op} %(v)s::numeric ELSE NULL END` | no |
| contains_any | containment OR | `(metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v0)s::text))) OR …%(v1)s…)` | yes |
| contains_all | array containment | `metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))` | yes (empty operand: no) |

**five distinct edge outcomes** of the POSITIVE leaf (negation inverts via the
three-valued `negated_leaf_sql`, so a negated leaf still excludes
missing/null/wrong-type):

| op | missing key | json null | wrong type | empty operand |
|---|---|---|---|---|
| eq | exclude | exclude | exclude | — |
| ne | **include** | **include** | **include** | — |
| in | exclude | exclude | exclude | exclude |
| gt/gte/lt/lte | exclude | exclude | exclude | — |
| contains_any | exclude | exclude | exclude | exclude |
| contains_all | exclude | exclude | exclude | **include** |

- **ne is null-safe** via `IS NOT TRUE` (null-INCLUSIVE); this differs from
  `NotPredicate(eq)`, which is null-EXCLUSIVE via the three-valued leaf.
- **`contains_all []`**: `@> '{"key": []}'` is true iff the field is a present
  array; when the field is **missing** it is **excluded** (not vacuously true).
  the empty-array case is NOT index-accelerated.
- **value validation** (raises `InvalidFilterError` in slice 4): range ops require
  a numeric value (int/float, **not** bool); `in`/`contains_*` require a
  homogeneous list (mixed json types and bool-as-number rejected); `eq/ne` take a
  single text/number/boolean scalar.

## 4. public score contract — one formula per mode (+ native score split)

uniform rank-based reciprocal rank, always higher-better:

```
surfaced_score(rank) = 1 / (SURFACED_RANK_K + rank)   # rank 0-based, K = 60
```

- monotonic decreasing in rank by construction => `search_related`
  relevance-descending holds regardless of raw scorer direction (bm25 `<@>`
  negative/lower-better, vector distance lower-better, rrf higher-better).
- **native score preserved separately (NB1)**: `QueryHit.native_score` and the
  `native_score` result key carry the raw backend number for diagnostics /
  calibration — never overloaded onto `max_score`. after dedup, `native_score`
  comes from the **same winning hit that supplied `max_score`** (the best-ranked
  occurrence), not averaged or retained per-query.
- **multi-query dedup + ordering** (in `NeonChunkSource.search_related`): a chunk
  hit by several queries keeps the **max** reciprocal rank as `max_score`;
  results sort by the FULL 3-tuple `(len(queries), not same_file, max_score)` all
  descending (mirrors `postgres/source.py`). result dicts key `{chunk, queries,
  same_file, max_score, native_score}`.
- exact numeric anchors: rank 0 -> `1/60 = 0.016666666666666666`, rank 1 ->
  `1/61`, rank 2 -> `1/62`. `<@>` polarity PROVEN on live neon (slice 3): scores
  are NEGATIVE, lower = more relevant, so candidate ordering is `ASC` (best-first)
  — smoke fixture `smoke-1` = -3.74 < `smoke-2` = -2.62 for query "quick brown
  fox". formula + monotonicity + dedup frozen here.

## 5. scan_chunks determinism

`NeonChunkSource.scan_chunks(batch_size=1000) -> Iterator[Chunk]` yields the full
corpus in a **total, pageable** order `(source_file, chunk_index, id)` over the
TYPED NOT NULL columns (not JSONB extraction), backed by the `scan` btree. this
avoids lexical `chunk_index` sorting (1,10,2), NULL-break keysets, and collation
ambiguity; the keyset cursor is `(source_file, chunk_index, id) > (%s, %s, %s)`.

## 6. eval jsonl schema

`NeonEvalRecord{capability, search_mode, query, filter_dsl (json dsl or null),
gold_chunk_hashes (exact — carried explicitly because `Chunk.to_dict` omits
`hash`), decoy_chunk_hashes}`. per-mode thresholds + lexical-ablation delta frozen:

| mode | hit@5 | mrr@5 | notes |
|---|---|---|---|
| lexical | 0.80 | 0.65 | fixed floor |
| vector | 0.85 | 0.70 | BM25 baseline + measured margin |
| hybrid | 0.80 | 0.50 | loose SMOKE floor (amended, see below) |

**hybrid smoke-floor amendment (reviewed, slice 7).** the frozen hybrid bar was
originally `0.90/0.75` as a fusion-necessity gate. that gate is **unbuildable on
this corpus**: the 31,665-chunk handbook is BOTH lexical-strong and vector-strong
(vector-only hit@5 ≈ 0.96), so the `lexical>5 AND vector>5` both-legs-miss
precondition holds 0/28 across blind candidates — isolating RRF fusion needs a
retrieval-CONFIG change (weaker/lower-dim embedder or starved top-k), not more
data. RRF is real and unit-tested in slice 4. hybrid therefore ships as a
**DEFERRED capability (Path X)** with a loose smoke floor `0.80/0.50`
(`eval_schema.DEFAULT_THRESHOLDS["hybrid"]`), NOT a fusion-beats-both-legs claim.
re-attempting a fusion-necessity gate must change the retrieval config first.

`LEXICAL_ABLATION_MIN_DELTA = 0.05`. measured provenance thresholds override the
fallbacks above at gate time.

## 7. embedding dim/metric + internal query interface

- `EMBEDDING_DIM = 3072`, `DISTANCE_METRIC = "cosine"`.
- `NeonQueryRequest(mode, top_k, text, vector, filter, hybrid)`. **filtering
  orthogonal** (3 modes x filtered/unfiltered), not a 4th mode. **hybrid rrf
  single-owned** by the query layer (`fuse_rrf`).

## PROVEN ON LIVE NEON (slice 3)

verified against a live neon lakebase compute (PG 18.4; `lakebase_vector`
1.0.0-dev, `lakebase_text` 0.1.0-dev, pgvector 0.8.1). the previously-provisional
DDL is now **frozen** in `schema.PROVEN_ANN_DDL` and `client.py`.

- **ann (B1)** — `CREATE INDEX ... USING lakebase_ann (embedding
  vector_cosine_ops)` on a full-precision `vector(3072)` column; query `embedding
  <=> %(vector)s::vector` `ORDER BY ... ASC`. `EXPLAIN` shows `Index Scan using
  ..._ann` (natural, not seq scan). frozen as one coherent unit — **type**
  `vector(3072)` + **opclass** `vector_cosine_ops` + **operator** `<=>` +
  **query-param cast** `::vector`. the cast is REQUIRED: a bound python list binds
  as `float8[]` and the cast-less operator errors (`operator does not exist:
  vector <=> double precision[]`). `lakebase_ann` (unlike pgvector `hnsw`, which
  rejects >2000 dims) indexes 3072 directly, so `halfvec` is NOT needed for
  correctness — `halfvec(3072)`+`halfvec_cosine_ops` also builds and is
  planner-used, kept in `schema.ANN_HALFVEC_ALTERNATIVE` as the storage-saving
  swap (halves bytes/vector at a small recall cost).
- **bm25 (B13)** — `CREATE INDEX ... USING lakebase_bm25 (content_tsv
  tsvector_bm25_ops) WITH (k1 = 1.2, b = 0.75)` built AFTER the load + `VACUUM`
  (corpus stats come from index metadata). query `content_tsv <@> to_bm25query(
  to_tsvector('pg_catalog.english'::regconfig, %(text)s), '<schema>.<index>'
  ::regclass)` `ORDER BY ... ASC`. `to_bm25query(tsvector, regclass) ->
  bm25query_tsvector`. scores NEGATIVE (lower = more relevant); `EXPLAIN` shows
  `Index Scan using ..._bm25` with top-K pushdown once the row count makes the
  index cheaper than a seq scan.
- **meta_gin (B3)** — `USING gin (metadata jsonb_path_ops)`; predicate `metadata
  @> %(f)s::jsonb`. `EXPLAIN` shows `Bitmap Index Scan on ..._gin` on a selective
  predicate at scale (tiny tables seq-scan, which is optimal and score-identical).

frozen operational caveats (from the slice-3 verify + design review):

- **regconfig lockstep** — the `content_tsv` generated column and every bm25 query
  MUST use the identical fully-qualified `'pg_catalog.english'::regconfig`; drift
  is silent recall loss, not an error.
- **k1/b baked in the index** — bm25 scores come from the index's stored
  `k1=1.2,b=0.75`; rebuilding with different values shifts magnitudes. frozen.
- **build-once versions** — bm25 idf is index-build-time; versions are
  build-once-immutable (versioned-replace), so idf never drifts under a live view.
  the writer owns the table and runs `VACUUM ANALYZE` in the ingest path (RO
  cannot vacuum).
- **NULL-drop** — `to_tsvector(NULL)`/NULL embedding silently drop a row from
  bm25/ann; `content` is `NOT NULL`, and ingest must supply a non-null embedding.
- **storage / projection** — a `vector(3072)` is ~12 KB and TOASTs out of line
  (uncompressible), plus `lakebase_ann` keeps its own copy — budget ~24 KB/row
  all-in; project `embedding` OUT of client-facing selects to avoid detoast over
  the wire. no pgvector ANN fallback exists at 3072 (>2000 cap), so the column is
  indexable ONLY by `lakebase_ann` — an availability dependency.
- **no PREPARE across index recreation** — the bm25 `::regclass` binds an OID; the
  client interpolates the schema/index per statement (re-parsed each call), and the
  client connects with `prepare_threshold=None` so psycopg never auto-prepares a
  plan that could reference a dropped/stale index after a version/index swap.
- **extension version pin (N3, follow-up)** — verified against dev builds
  `lakebase_vector` 1.0.0-dev / `lakebase_text` 0.1.0-dev; behavior matched the
  frozen contract, but these are pre-GA versions. a hard version-gate in
  `provision.py` (assert `extversion` against an allowlist before build) is a
  deferred follow-up, not implemented now.

## enablement (slice 3, how the sample DB is stood up)

the lakebase extensions load only when `lakebase_vector`+`lakebase_text` are in the
compute's preload libraries. on neon this is set via the **project setting**
`preload_libraries.enabled_libraries` (the raw `shared_preload_libraries` /
`neon.lakebase_mode` GUCs are rejected by the settings API), then the endpoint is
restarted/woken and the admin (`neon_superuser`) runs `CREATE EXTENSION ...
CASCADE`. see `provision.py` (idempotent) and `README.md` (runbook). three roles
with EXPLICIT grants (a SQL-created role does NOT inherit `neon_superuser`):
**admin** installs extensions only; **writer** (`benchmax_writer`) OWNS the schema
+ version tables (DDL + ingest) -> `NEON_CORPUS_DSN_RW`; **read-only**
(`benchmax_ro`) gets schema `USAGE` + the writer's `ALTER DEFAULT PRIVILEGES ...
GRANT SELECT ON TABLES` (covers current + future version tables/views) ->
`NEON_CORPUS_DSN_RO`. the RO SELECT on the base tables is what lets `to_bm25query`
(which runs with invoker rights, NOT the owner-rights view's) read the bm25 index
stats — verified end-to-end under the RO role.

## deferred to the implementing slice (contract frozen now, enforcement later)

- **B5 pruning seam + concurrent allocation/prune race-safety -> Slice 2.** the
  ledger schema, state CHECK domain, current-pointer unique index, advisory-lock
  activation contract, and retention policy are frozen here; the prune executor
  and the concurrent build-vs-prune race tests land in Slice 2 with the real DDL.
- **B7 fully-seeded behavioral tests -> the slice that implements each behavior
  (1/2/4).** round 2 ships correctly-structured strict xfails (fake-backed,
  non-vacuous, typed) that raise `NotImplementedError`; the real behavior
  assertions (filter SQL execution, paged scan rows, transaction rollback events)
  are filled when each behavior is implemented.

## deviations from the brief (surfaced, not silently re-planned)

1. **no in-repo sql template exists** — the `postgres` provider is an http client
   to the corpora api (client-side dict merge). neon is the first sql-emitting
   backend; the sql seam is frozen from first principles. turbopuffer remains the
   template for the credential seam + pickle-safe search client; score
   dedup/order mirrors `postgres/source.py`'s 3-tuple.
2. **new test convention** — parametrized `xfail(strict=True,
   raises=NotImplementedError)` skeletons (house style is class-grouped
   fake-injection). justified by the testable-skeleton requirement.
3. **shared operator enum untouched** — `ne/gt/lt` in a neon-local superset;
   promotion into `search_schema/search_types.py` deferred to slice 4.
4. **`?|`/`?&` -> `@>` containment** — the filter representation changed from the
   first draft's jsonb key-existence to indexable containment, reconciling B2
   (type safety) and B3 (indexability) in one form.
