# neon corpus provider — frozen contracts (slice a)

design-lock for the neon lakebase (postgres + pgvector + bm25) corpus provider.
this slice ships typed stubs, this doc, and parametrized test skeletons only. all
real sql/client/filter/search logic lands in slices 1/2/4; live verification of
the ann access method + index EXPLAIN lands in slice 3 (see PROVISIONAL below).

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
| `embedding` | PROVISIONAL vector type | see §PROVISIONAL — not frozen |
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
- **atomic activate** (`activate_version_sql(spec, grant)`): one transaction under
  `pg_advisory_xact_lock(logical)` clears the prior `is_current`, sets this
  version `activated`/`is_current`, `create or replace view`, AND issues the RO
  grants from `grant` — commit or roll back together (proven by
  `test_activation_rolls_back_atomically`). **rollback**: re-point `is_current` to
  a prior `activated` version; old physical tables retained → O(1),
  non-destructive.
- **retention** (`RetentionPolicy`, self-validating): keep >= 2 activated
  (rollback always has a target) + >= 1 ready. the **pruning seam** and full
  concurrent allocation/prune race-safety are **deferred to Slice 2** (where the
  real DDL/transactions land); the invariant + locking contract are frozen now.
- **RO grants** (`ReadGrantSpec`): owner-rights view (`security_invoker = false`)
  => RO gets schema `USAGE` + `SELECT` on the stable view only, never physical
  tables. issued on FIRST view creation (`create or replace` preserves an existing
  ACL but a first create has none), which is why activation carries `grant`.
- **view identifier policy** (B4): the reader-facing view name is
  `view_name(logical_name)` — validated printable-ASCII and length-fitted to 63
  bytes (a long logical name is hash-fitted, same as physical names); readers
  resolve it through `view_name`, never assume it equals the raw logical name.
- indexes per version: `ann` (PROVISIONAL), `bm25` (lexical), `meta_gin`
  (`jsonb_path_ops`, serves `@>`), `scan` (btree `(source_file, chunk_index,
  id)`), `tsv_gin` (native fts fallback).
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
boolean shapes):

| op | family | positive sql | indexable |
|---|---|---|---|
| eq | containment | `metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))` | yes |
| ne | negated containment | `(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))) IS NOT TRUE` | no |
| in | containment OR | `(metadata @> …%(v0)s…) OR (metadata @> …%(v1)s…)` | yes |
| gt/gte/lt/lte | range CASE | `CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' THEN (metadata ->> %(k)s)::numeric {op} %(v)s::numeric ELSE NULL END` | no |
| contains_any | containment OR | `(metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text))) OR …%(v1)s…)` | yes |
| contains_all | array containment | `metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))` | yes (empty operand: no) |

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
  `1/61`, rank 2 -> `1/62`. empirical `<@>` range validation deferred to the slice
  3 live smoke; formula + monotonicity + dedup frozen here.

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

| mode | hit@5 | mrr@5 |
|---|---|---|
| lexical | 0.80 | 0.65 |
| vector | 0.85 | 0.70 |
| hybrid | 0.90 | 0.75 |

`LEXICAL_ABLATION_MIN_DELTA = 0.05`. schema only; data built later.

## 7. embedding dim/metric + internal query interface

- `EMBEDDING_DIM = 3072`, `DISTANCE_METRIC = "cosine"`.
- `NeonQueryRequest(mode, top_k, text, vector, filter, hybrid)`. **filtering
  orthogonal** (3 modes x filtered/unfiltered), not a 4th mode. **hybrid rrf
  single-owned** by the query layer (`fuse_rrf`).

## PROVISIONAL — must verify on live neon (slice 3)

the following are **not frozen**; slice 3 must verify them against the live
lakebase DB and then freeze only the proven form (see
`schema.PROVISIONAL_ANN_OPTIONS`):

- **ann access method + vector type + opclass** (B1). the stub emits vanilla
  `hnsw`/`halfvec` as a *fallback*; the intended primary is native
  `lakebase_ann` on `vector(3072)`. the 2000-dim `vector` hnsw limit and the
  `halfvec` workaround are vanilla-pgvector facts, UNPROVEN for `lakebase_ann`.
  slice 3 must: attempt `lakebase_ann` on `vector(3072)` AND the halfvec
  alternative, verify index creation, cosine queries, matching query casts, and
  `EXPLAIN` index use — then freeze the proven DDL.
- **meta_gin index EXPLAIN** (B3). the `@>`/`jsonb_path_ops` pairing is standard
  postgres and frozen, but slice 3 should confirm via `EXPLAIN` that the
  containment predicates actually hit the GIN on live data.

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
