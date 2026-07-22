# neon corpus provider — frozen contracts (slice a)

design-lock for the neon lakebase (postgres + pgvector + bm25) corpus provider.
this slice ships typed stubs, this doc, and parametrized test skeletons only. all
real sql/client/filter/search logic lands in slices 1/2/4; the live raw-score
smoke lands in slice 3. the seven contracts below are frozen here.

> convention note: the existing `corpus/` tests are class-grouped with fake
> injection and assert on native filter dicts — they do not use
> `pytest.mark.parametrize`, `xfail`, or expected-sql assertions. this slice
> deliberately introduces parametrized `xfail` skeletons for the not-yet-built
> sql, because the brief mandates testable truth-table/score skeletons that
> slices 1/2/4 fill in. flagged so reviewers know it is a new local convention.

## 1. physical table + versioned-replace model

managed physical table (per version):

| column | type | notes |
|---|---|---|
| `id` | `text primary key` | chunk hash (sha256 hexdigest) |
| `content` | `text not null` | |
| `metadata` | `jsonb not null default '{}'` | |
| `embedding` | `vector(3072)` | stored full precision |
| `content_tsv` | `tsvector generated always as (to_tsvector(<config>::regconfig, content)) stored` | config **configurable**, baked per version |

- **logical vs physical**: readers address a stable *logical name*; each ingest
  builds `\<logical\>__v\<N\>` with its own indexes. an *active-version pointer* =
  a `neon_corpus_versions` registry row + a `create or replace view \<logical\>`.
- **atomic activate**: one transaction upserts the registry pointer and
  re-points the view. **rollback**: re-point to a prior version; old physical
  tables are retained until pruned, so rollback is O(1) and non-destructive.
- chunk identity changes with content/metadata, so versioned *replace* (not
  in-place upsert) is the correct shape.
- indexes per version: `hnsw` (ann), `bm25` (lexical), `meta_gin` (jsonb
  containment), `tsv_gin` (native fts fallback).
- **pgvector dim gotcha (frozen decision)**: `vector` hnsw/ivfflat index only up
  to 2000 dims, so 3072-dim ann is indexed on a `halfvec(3072)` cast expression
  with `halfvec_cosine_ops` (halfvec supports hnsw up to 4000). storage stays
  full-precision `vector(3072)`.

artifacts: `schema.py` (`NeonTableSpec`, ddl templates, `physical_table_name`,
`index_names`, activate/rollback stubs).

## 2. credential constructor signature

`dsn_provider: str | TokenProvider | None`, resolved lazily, mirroring
turbopuffer's `as_token_provider`/`env_token`. **separate read vs write
surfaces** — a single provider can't be both rw-ingest and ro-search:

- read: `resolve_read_dsn_provider(...)` -> default `env_token("NEON_CORPUS_DSN_RO")`, select-only grant.
- write: `resolve_write_dsn_provider(...)` -> default `env_token("NEON_CORPUS_DSN_RW")`, ddl+dml grant.

the sandbox rollout env only ever receives the ro provider. artifacts:
`credentials.py` (signatures), consumed by `search.py`/`source.py`/`client.py`.
seam implemented in slice 4.

## 3. 9-op filter truth table

operators: `eq, ne, in, gt, gte, lt, lte, contains_any, contains_all`. the
shared enum has six today; neon adds `ne, gt, lt` (promotion into the shared
enum deferred to slice 4 — a local `NeonFieldOperator` superset holds the
contract meanwhile). metadata is accessed via **bound json paths**
(`metadata ->> %(k)s`), never `psycopg.sql.Identifier` on a caller key. cast is
`::numeric` iff the dsl value is int/float, else text.

value-present emitted sql (numeric variant shown; text variant drops `::numeric`):

| op | emitted sql | cast | null/missing key | empty-array operand | negation |
|---|---|---|---|---|---|
| eq | `(metadata ->> %(k)s) = %(v)s` | text | exclude | — | `not (...)` |
| ne | `(metadata ->> %(k)s) is distinct from %(v)s` | text | **include** | — | `not (...)` |
| in | `(metadata ->> %(k)s) = any(%(v)s)` | text | exclude | exclude | `not (...)` |
| gt | `(metadata ->> %(k)s)::numeric > %(v)s` | numeric | exclude | — | `not (...)` |
| gte | `(metadata ->> %(k)s)::numeric >= %(v)s` | numeric | exclude | — | `not (...)` |
| lt | `(metadata ->> %(k)s)::numeric < %(v)s` | numeric | exclude | — | `not (...)` |
| lte | `(metadata ->> %(k)s)::numeric <= %(v)s` | numeric | exclude | — | `not (...)` |
| contains_any | `(metadata -> %(k)s) ?| %(v)s` | jsonb_array | exclude | exclude | `not (...)` |
| contains_all | `(metadata -> %(k)s) ?& %(v)s` | jsonb_array | exclude | **include** (vacuous) | `not (...)` |

edge-condition rules (frozen):
- **null / missing key**: `->>` yields sql null; under 3-valued logic every op
  except `ne` drops the row. `ne` uses `is distinct from`, so a null/missing key
  is "distinct from" the value and the row is **included** (null-safe ne).
- **empty-array operand**: `in []` and `contains_any []` match nothing
  (exclude); `contains_all []` is vacuously true (**include**) — the one
  exception.
- **negation**: `NotPredicate` -> `not (<inner>)`, inheriting 3-valued logic, so
  `not (null)` is null and negating a leaf over a missing key still excludes.
  null-inclusive negation would need `(<inner>) is not true`; the frozen
  contract is plain `not (...)`.
- **contains_any/all representation**: primary form is jsonb key-existence
  `?|` / `?&` over an array-of-text field. the typed-array `&&` / `@>` form is
  the deferred fallback for numeric/typed arrays (noted per row in
  `filter_mapper.py`).

artifacts: `filter_mapper.py` (`FILTER_TRUTH_TABLE`, `NEON_FIELD_OPERATORS`,
`NEGATION_TEMPLATE`, `predicate_to_sql` stub); skeleton
`tests/.../neon/test_filter_truth_table.py`.

## 4. public score contract — one formula per mode

native scorers disagree on direction: bm25 `<@>` is negative/lower-better,
vector cosine distance is lower-better, rrf is higher-better. rather than surface
three scales, the **public score is a uniform rank-based reciprocal rank**, always
higher-better:

```
surfaced_score(rank) = 1 / (SURFACED_RANK_K + rank)   # rank 0-based, K = 60
```

- monotonic decreasing in rank by construction => `search_related`
  relevance-descending holds regardless of the raw scorer range.
- `rank` = 0-based position in a single query's result list ordered better-first
  by that mode's native scorer (bm25 asc `<@>`, vector asc distance, hybrid the
  fused ordering).
- **multi-query dedup** mirrors `postgres/source.py`: a chunk hit by several
  queries keeps the **max** reciprocal rank as `max_score`; results sort by
  `(len(queries), not same_file, max_score)` all descending.
- raw native scores retained internally, not surfaced. empirical `<@>` range
  validation deferred to the slice 3 live smoke; formula + monotonicity + dedup
  frozen here.

exact numeric anchors (in the test skeleton): rank 0 -> `1/60 =
0.016666666666666666`, rank 1 -> `1/61 = 0.01639344262295082`, rank 2 -> `1/62 =
0.016129032258064516`. a chunk at rank 0 in query a and rank 2 in query b
dedups to `max_score = 1/60`.

artifacts: `search.py` (`surfaced_score`, `SURFACED_RANK_K`, `fuse_rrf` stub);
skeleton `tests/.../neon/test_surfaced_score.py`.

## 5. scan_chunks determinism

`NeonChunkSource.scan_chunks(batch_size=1000) -> Iterator[Chunk]` yields the full
corpus in a **stable order** `(file, index, id)` (`SCAN_ORDER_BY`) via keyset
pagination, so qa-gen full-corpus materialization is reproducible run to run
(not just count/sample reads). `file`/`index` come from chunk metadata
(`source_file`, `chunk_index`); `id` (hash) is the final tiebreak. neon extension
to the surface; promotion onto the shared `ChunkSource` protocol deferred.

artifacts: `source.py` (`scan_chunks` stub, `SCAN_ORDER_BY`); skeleton
`tests/.../neon/test_scan_chunks_determinism.py`.

## 6. eval jsonl schema

fields: `capability`, `search_mode`, `query`, `filter_dsl` (json dsl form or
null), `gold_chunk_hashes` (exact — carried explicitly because `Chunk.to_dict`
omits `hash`), `decoy_chunk_hashes`. per-mode thresholds + lexical-ablation delta
are frozen:

| mode | hit@5 | mrr@5 |
|---|---|---|
| lexical | 0.80 | 0.65 |
| vector | 0.85 | 0.70 |
| hybrid | 0.90 | 0.75 |

`LEXICAL_ABLATION_MIN_DELTA = 0.05` — hybrid must beat lexical-only hit@5 by at
least this. schema only; data built later. artifacts: `eval_schema.py`
(`NeonEvalRecord`, `NeonEvalThresholds`, `DEFAULT_THRESHOLDS`).

## 7. embedding dim/metric + internal query interface

- `EMBEDDING_DIM = 3072`, `DISTANCE_METRIC = "cosine"` (`schema.py`).
- internal request `NeonQueryRequest(mode, top_k, text, vector, filter, hybrid)`
  (`search.py`). **filtering is orthogonal** (3 modes x filtered/unfiltered), not
  a fourth mode. **hybrid rrf has a single owner** = the internal query layer
  (`fuse_rrf`); no other component blends lexical+vector.

## deviations from the brief (escalated, not silently re-planned)

1. **no in-repo sql template exists.** the `postgres` provider is an http client
   to the corpora api (client-side dict merge for `search_related`), and no
   provider imports psycopg. neon is the first sql-emitting backend, so the sql
   seam (bound json paths, `<@>`, halfvec index) is frozen here from first
   principles rather than mirrored. turbopuffer is still the template for the
   credential seam and the pickle-safe search client shape.
2. **new test convention.** parametrized `xfail` skeletons are introduced (see
   note at top); the house style is class-grouped fake-injection. justified by
   the brief's testable-skeleton requirement.
3. **shared operator enum untouched.** `ne/gt/lt` live in a neon-local
   `NeonFieldOperator` superset; promoting them into
   `search_schema/search_types.py` is a cross-cutting edit deferred to slice 4 to
   keep this slice inside the `neon/` package.
