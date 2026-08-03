# Neon example backend contract

This provider gives Benchmax environments a versioned Neon Lakebase corpus with
vector, BM25, and hybrid retrieval. These invariants are part of its public
contract.

## credentials

- `NeonChunkSource` receives the data-preparation database URL used to build and activate corpus versions.
- `NeonSearch` receives the read-only search database URL and cannot mutate the corpus. This URL is intentionally included in the serialized environment so hosted rollouts can search Neon.
- Live database connections are never serialized. Async retrieval opens a fresh connection per operation and retries a dead connection once.

## Versioned replacement

Each logical corpus has immutable physical tables named by version and one stable
reader view. Ingestion creates and fills a new table, builds its indexes, records
it as ready, then atomically swaps the view while holding an advisory lock. Query
transactions take the corresponding shared lock, so all legs of a hybrid query
read the same version. Prior versions remain available for rollback and may be
pruned by the configured retention policy.

The physical schema contains:

- a deterministic chunk hash primary key;
- content and JSON metadata;
- typed `source_file` and `chunk_index` columns for stable scans;
- `vector(3072)` embeddings using cosine distance;
- a stored `tsvector` generated with the version's fixed text-search config.

The Lakebase ANN and BM25 indexes are built after loading the rows. The embedding
model must therefore return exactly 3,072 values.

## Query behavior

- Modes are `lexical`, `vector`, and `hybrid`; `auto` chooses hybrid when an
  embedder exists and lexical otherwise.
- Hybrid retrieval oversamples each leg and combines their rankings once with
  reciprocal-rank fusion.
- Results always expose a higher-is-better rank score, `1 / (60 + rank)`. Native
  BM25, cosine-distance, or RRF scores remain available internally for diagnostics.
- Metadata filters are parameterized and pushed into both hybrid legs. Public
  filter capabilities match Benchmax's shared operators: `eq`, `in`, `gte`,
  `lte`, `contains_any`, and `contains_all`, plus `and`, `or`, and `not`.
- Candidate SQL binds every caller value; validated identifiers and allowlisted
  text-search configs are the only values interpolated as SQL structure.

## Read-only grants

The rollout role receives schema usage and select privileges only. BM25's
`to_bm25query` runs with invoker rights and needs select access to the physical
version tables in addition to the reader view, so the writer's default privileges
grant select on future reader objects. Neither grant includes insert, update,
delete, schema changes, or version activation.
