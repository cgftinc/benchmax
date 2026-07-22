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
  created_at, ready_at, activated_at)` with state `building -> ready ->
  activated -> retired`. this replaces the single-row pointer so concurrent
  ingest / enumerate / prune / build-vs-ready are all well-defined.
- **atomic activate**: one transaction under `pg_advisory_xact_lock(logical)`
  upserts the ledger active row AND `create or replace view` — they commit or
  roll back together (proven by `test_activation_rolls_back_atomically`).
  **rollback**: re-point to any prior `ready`/`activated` version; old physical
  tables retained until pruned → O(1), non-destructive.
- **retention/pruning** (`RetentionPolicy`): keep >= 2 activated (rollback always
  has a target) + >= 1 ready; older versions retired then pruned.
- **RO grants** (`ReadGrantSpec`): the RO role gets schema `USAGE` + `SELECT` on
  the stable view only, re-issued on FIRST view creation (`create or replace`
  preserves an existing ACL but the first create has none). owner-rights view =>
  RO never touches physical tables.
- indexes per version: `ann` (PROVISIONAL), `bm25` (lexical), `meta_gin`
  (`jsonb_path_ops`, serves `@>`), `scan` (btree `(source_file, chunk_index,
  id)`), `tsv_gin` (native fts fallback).
- **injection-safe DDL** (B4): all identifiers via `psycopg.sql.Identifier`;
  regconfig + options via allowlist + bound `sql.Literal`; version numbers
  validated (`validate_version`, positive int, rejects bool); names length-safe
  to 63 bytes via content-hash suffix (`_fit_identifier`); the execute seam
  accepts `sql.Composable`, never `str`.

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

two safety properties are frozen (they drove the review REVISE):

- **type-directed, never-throwing**: `eq/ne/in` and `contains_*` emit JSONB
  **containment** (`metadata @> jsonb_build_object(...)`), which is type-aware and
  needs no cast — a heterogeneous stored value can never abort the query. only
  `gt/gte/lt/lte` cast, and they are **guarded** by `jsonb_typeof(metadata ->
  key) = 'number'`.
- **indexable**: containment is served by the `meta_gin` `jsonb_path_ops` GIN.
  the old `?|`/`?&` forms are rejected — a whole-doc GIN cannot serve them (B3).
  range predicates are not GIN-indexable (per-key expression btree is an
  operational add-on).

canonical value-present sql (numeric value shown for eq/ne/in; text for contains):

| op | family | canonical sql | indexable |
|---|---|---|---|
| eq | containment | `metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))` | yes |
| ne | negated containment | `(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))) IS NOT TRUE` | no |
| in | containment OR | `(metadata @> …%(v0)s…) OR (metadata @> …%(v1)s…)` | yes |
| gt/gte/lt/lte | range | `jsonb_typeof(metadata -> %(k)s) = 'number' AND (metadata ->> %(k)s)::numeric {op} %(v)s::numeric` | no |
| contains_any | containment OR | `(metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text))) OR …%(v1)s…)` | yes |
| contains_all | array containment | `metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))` | yes |

**five distinct edge outcomes** (include / exclude), per op:

| op | missing key | json null | wrong type | empty operand | negated |
|---|---|---|---|---|---|
| eq | exclude | exclude | exclude | — | `not (…)` 3-valued |
| ne | **include** | **include** | **include** | — | `not (…)` |
| in | exclude | exclude | exclude | exclude | `not (…)` |
| gt/gte/lt/lte | exclude | exclude | exclude | — | `not (…)` |
| contains_any | exclude | exclude | exclude | exclude | `not (…)` |
| contains_all | exclude | exclude | exclude | **include** | `not (…)` |

- **ne is null-safe** via `IS NOT TRUE`: missing/null/wrong-type => included; only
  an equal stored value is excluded.
- **`contains_all []`**: `@> '{"key": []}'` is true iff the field is a present
  array (empty array is contained in any array); when the field is **missing** it
  is **excluded** (not vacuously true).
- **negation**: `NotPredicate` -> `not (<inner>)`, inheriting 3-valued logic
  (`not(null)` is null → a negated leaf over a missing key still excludes).
  null-inclusive negation would need `(<inner>) is not true`; the frozen contract
  is plain `not (…)`.
- **value validation** (raises `InvalidFilterError` in slice 4): range ops
  require a numeric value (int/float, **not** bool); `in`/`contains_*` require a
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
  calibration — never overloaded onto `max_score`.
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
