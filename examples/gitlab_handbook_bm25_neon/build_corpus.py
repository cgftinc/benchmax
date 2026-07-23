"""Ingest the deterministic handbook collection into the Neon lakebase corpus.

The corpus-build half of the ported loader: sparse-checkout the pinned handbook,
re-chunk it with filterable metadata (:mod:`handbook_corpus`), then publish it as a
new Neon corpus version at full 3072-dim. The build streams (embed + insert one
batch at a time, committing each) and is crash-resumable, rather than the
provider's ``populate_from_chunks`` which embeds the whole corpus up front (~3 GB
of float lists) and OOMs a memory-bounded host at full-corpus scale.

The embeddings host is pinned to ``llm.<base_domain>`` (default ``castform.dev``)
so vectors are computed against the platform ``text-embedding-3-large`` endpoint,
never leaking to ``castform.com`` (NB2). DSNs come only from the Neon seam
(``NEON_CORPUS_DSN_RW`` / ``NEON_CORPUS_DSN_RO``); a bare ``DATABASE_URL`` never
satisfies it (NB5).

Run (from the benchmax workspace root, after sourcing the 0600 creds file)::

    set -a; source ~/.config/neon-benchmax.env; set +a
    uv run --extra neon python examples/gitlab_handbook_bm25_neon/build_corpus.py \\
        --work-dir /tmp/handbook_repo --provenance-out datasets/corpus_build.json

This spends embedding credits over the full corpus; it is a one-shot build, not a
CI step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from castform.rag.corpus.embed import DEFAULT_EMBED_MODEL, platform_embed_fn
from castform.rag.corpus.neon.client import INSERT_COLUMNS, NeonClient
from castform.rag.corpus.neon.credentials import resolve_write_dsn_provider
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA, RO_ROLE
from castform.rag.corpus.neon.schema import (
    DEFAULT_TEXT_SEARCH_CONFIG,
    NeonTableSpec,
    ReadGrantSpec,
    index_names,
    physical_table_name,
    view_name,
)
from castform.rag.corpus.neon.source import NeonIngestError
from castform.rag.qa_generation.neon_entrypoint import neon_llm_url

from handbook_corpus import (
    EXPECTED_CHUNK_COUNT,
    HANDBOOK_COMMIT,
    HANDBOOK_REPO_URL,
    HANDBOOK_SUBDIR,
    LOGICAL_NAME,
    ChunkerParams,
    build_collection,
    git_tracked_docs,
    sparse_checkout,
)


def _resolve_build_version(client: NeonClient, logical_name: str) -> tuple[int, str]:
    """Return ``(version, mode)`` for the next build.

    ``mode`` is one of:

    * ``"finalized"`` — a prior run built and indexed a version (state ``ready``)
      but died before activating it; reuse and publish it, never re-embed.
    * ``"resume"`` — a prior run died mid-load (state ``building``); reuse its
      version and table and embed only the still-missing chunks.
    * ``"fresh"`` — no in-flight version; allocate one past the current maximum.

    ``finalized`` outranks ``resume`` because a ``ready`` version is complete and
    only needs publishing, whereas a ``building`` one still needs embedding.
    """
    rows = client.execute(client.read_ledger_sql(), {"logical": logical_name})
    ready = [
        int(v) for v, state, is_current in rows if state == "ready" and not is_current
    ]
    building = [int(v) for v, state, _cur in rows if state == "building"]
    if ready:
        return max(ready), "finalized"
    if building:
        return max(building), "resume"
    return max((int(v) for v, _state, _cur in rows), default=0) + 1, "fresh"


def _present_ids(client: NeonClient, spec: NeonTableSpec) -> set[str]:
    """Return the chunk ids already committed to a version's table (for resume)."""
    from psycopg import sql

    table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
    rows = client.execute(sql.SQL("SELECT id FROM {}").format(table))
    return {r[0] for r in rows}


def _table_exists(client: NeonClient, spec: NeonTableSpec) -> bool:
    """Whether a version's physical table has been created yet.

    A ``building`` ledger row can outlive a crash between allocation and table
    creation, leaving the row without its table; resuming such a version must
    create the table before inserting.
    """
    from psycopg import sql

    rows = client.execute(
        sql.SQL(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %(schema)s AND table_name = %(table)s"
        ),
        {
            "schema": CORPUS_SCHEMA,
            "table": physical_table_name(spec.logical_name, spec.version),
        },
    )
    return bool(rows)


def _idempotent_insert_sql(spec: NeonTableSpec):
    """``INSERT ... ON CONFLICT (id) DO NOTHING`` so a re-inserted batch is a no-op."""
    from psycopg import sql

    table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in INSERT_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS)
    return sql.SQL(
        "INSERT INTO {table} ({cols}) VALUES ({vals}) ON CONFLICT (id) DO NOTHING"
    ).format(table=table, cols=cols, vals=placeholders)


def _stream_insert(
    client: NeonClient,
    spec: NeonTableSpec,
    chunks: list,
    embed_fn,
    batch_size: int,
) -> None:
    """Embed + insert the *missing* chunks one batch at a time, committing each.

    Resumable and idempotent: chunks whose id is already committed are skipped (so
    a restart never re-embeds paid-for rows), each batch commits immediately (so a
    death loses at most the in-flight batch), and the insert is ``ON CONFLICT DO
    NOTHING`` (so a partially-committed batch replays cleanly). Emits a per-batch
    heartbeat so progress is visible in the detached log.
    """
    from psycopg.types.json import Jsonb

    present = _present_ids(client, spec)
    remaining = [c for c in chunks if c.hash not in present]
    insert_sql = _idempotent_insert_sql(spec)
    conn = client._live_conn()
    total = len(remaining)
    n_batches = (total + batch_size - 1) // batch_size
    start_time = time.time()
    committed = len(present)
    for b, start in enumerate(range(0, total, batch_size), start=1):
        batch = remaining[start : start + batch_size]
        vectors = embed_fn([c.content for c in batch])
        if len(vectors) != len(batch):
            raise NeonIngestError(
                f"embed_fn returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        rows = []
        for chunk, vector in zip(batch, vectors, strict=True):
            md = chunk.metadata_dict
            rows.append(
                (
                    chunk.hash,
                    chunk.content,
                    Jsonb(md),
                    list(vector),
                    str(md.get("file", "")),
                    int(md.get("index", 0) or 0),
                )
            )
        with conn.cursor() as cur:
            cur.executemany(insert_sql, rows)
        conn.commit()
        committed += len(rows)
        print(
            f"batch {b}/{n_batches} committed, rows={committed}, "
            f"elapsed={time.time() - start_time:.0f}s",
            flush=True,
        )
        del rows, vectors


def _finalize_version(client: NeonClient, spec: NeonTableSpec, expected: int) -> None:
    """Index, vacuum, and mark a version ready — only once it holds every chunk.

    Guards the versioned-replace contract: a version that is short of ``expected``
    rows is never finalized (and therefore never activated), so the reader view can
    never point at a partial corpus. Index creation tolerates a duplicate from a
    prior partly-finalized run.
    """
    import psycopg

    count = len(_present_ids(client, spec))
    if count != expected:
        raise NeonIngestError(
            f"refusing to finalize {spec.logical_name} v{spec.version}: "
            f"{count} rows present, expected {expected}"
        )

    existing = _existing_indexes(client, spec)
    names = index_names(spec.logical_name, spec.version)
    conn = client._live_conn()
    previous = conn.autocommit
    conn.autocommit = True  # isolate each CREATE INDEX so one dup can't poison the rest
    try:
        for name, stmt in [
            (names["ann"], client.create_ann_index_sql(spec)),
            (names["bm25"], client.create_bm25_index_sql(spec)),
        ]:
            if name not in existing:
                conn.execute(stmt)
        if names["meta_gin"] not in existing:
            for stmt in client.create_aux_indexes_sql(spec):
                try:
                    conn.execute(stmt)
                except psycopg.errors.DuplicateTable:
                    pass
    finally:
        conn.autocommit = previous

    client.vacuum(spec)
    client.execute(
        client.mark_ready_sql(spec),
        {"logical": spec.logical_name, "version": spec.version},
    )


def _existing_indexes(client: NeonClient, spec: NeonTableSpec) -> set[str]:
    from psycopg import sql

    rows = client.execute(
        sql.SQL("SELECT indexname FROM pg_indexes WHERE schemaname = %(schema)s"),
        {"schema": CORPUS_SCHEMA},
    )
    return {r[0] for r in rows}


def _build_and_activate(
    client: NeonClient,
    logical_name: str,
    chunks: list,
    embed_fn,
    batch_size: int,
    *,
    text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
) -> tuple[int, int]:
    """Resolve/resume a version, stream-ingest the missing chunks, then activate it.

    The crash-resumable orchestration shared by the CLI build and the unit tests,
    kept free of the sparse-checkout so it is driveable with a fake client:

    1. decide the target version — publish a lingering ``ready`` one (finalized),
       resume a lingering ``building`` one, or allocate a fresh one;
    2. ensure the extensions, pgvector adapters, and ledger exist (all idempotent);
       create the physical table on a fresh version, or on a resumed ``building``
       version whose table never got created;
    3. stream the still-missing chunks in per-batch-committed batches;
    4. finalize (index + vacuum + mark ready) only at the full expected row count,
       then activate — so a partial corpus is never published.

    Returns ``(version, rows_present)`` where ``rows_present`` is the table's row
    count after the build (carried-over rows included).
    """
    version, mode = _resolve_build_version(client, logical_name)
    spec = NeonTableSpec(logical_name, version, text_search_config=text_search_config)
    grant = ReadGrantSpec(
        schema=CORPUS_SCHEMA, view=view_name(logical_name), ro_role=RO_ROLE
    )

    for statement in client.create_extensions_sql():
        client.execute(statement)
    client.register_vector_types()
    for statement in client.create_ledger_sql():
        client.execute(statement)

    if mode == "finalized":
        # A prior run built + indexed this version (state ``ready``) but died before
        # activating it; publish it directly rather than re-embed the whole corpus.
        present = len(_present_ids(client, spec))
        if present != len(chunks):
            raise NeonIngestError(
                f"ready version v{version} holds {present} rows, expected {len(chunks)}"
            )
        print(f"activating ready version v{version} without re-embed", flush=True)
        client.activate(spec, grant)
        return version, present

    if mode == "resume":
        print(f"resuming building version v{version}", flush=True)
        if not _table_exists(client, spec):
            client.execute(client.create_table_sql(spec))
    else:  # fresh
        client.allocate_version(spec)
        client.execute(client.create_table_sql(spec))

    _stream_insert(client, spec, chunks, embed_fn, batch_size)
    _finalize_version(client, spec, expected=len(chunks))
    client.activate(spec, grant)
    return version, len(_present_ids(client, spec))


def build_and_ingest(
    *,
    work_dir: Path,
    logical_name: str = LOGICAL_NAME,
    base_domain: str = "castform.dev",
    commit: str = HANDBOOK_COMMIT,
    batch_size: int = 300,
) -> dict[str, object]:
    """Check out, chunk, and stream-ingest the handbook; return the provenance.

    Streams the build (embed + insert one batch at a time, committing each) rather
    than ``NeonChunkSource.populate_from_chunks``, which embeds every chunk up front
    (~3 GB of 3072-dim float lists) and OOMs a memory-bounded host at full-corpus
    scale. The build is crash-resumable: a run that dies mid-load leaves a
    ``building`` version whose committed rows survive, and a re-run reuses it and
    embeds only the chunks still missing. The version is finalized and activated
    only once it holds every chunk, so the reader view never points at a partial
    corpus.

    Args:
        work_dir: Scratch directory for the sparse checkout.
        logical_name: Neon logical corpus name to publish under.
        base_domain: Platform endpoint domain for embeddings (pins the host).
        commit: Exact handbook commit SHA to ingest.
        batch_size: Chunks per embedding/insert batch.

    Returns:
        Build provenance; ``inserted`` is the total row count present after the
        build (including rows carried over from a resumed version).
    """
    llm_url = neon_llm_url(base_domain)
    params = ChunkerParams()

    docs_dir = sparse_checkout(work_dir, commit=commit)
    tracked = git_tracked_docs(work_dir, HANDBOOK_SUBDIR, params.file_extensions)
    collection = build_collection(docs_dir, params=params, files=tracked)
    chunks = list(collection)
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise NeonIngestError(
            f"expected {EXPECTED_CHUNK_COUNT} chunks from the pinned tree, got "
            f"{len(chunks)} — refusing to ingest a corpus of unexpected size"
        )

    embed_fn = platform_embed_fn(base_url=llm_url)
    client = NeonClient(resolve_write_dsn_provider(None))
    version, present = _build_and_activate(
        client, logical_name, chunks, embed_fn, batch_size
    )

    return {
        "logical_name": logical_name,
        "corpus_source": {
            "repo_url": HANDBOOK_REPO_URL,
            "subdir": HANDBOOK_SUBDIR,
            "commit": commit,
        },
        "chunker": params.as_dict(),
        "chunk_count": len(chunks),
        "inserted": present,
        "embedder": {"model": DEFAULT_EMBED_MODEL, "dim": 3072, "base_url": llm_url},
        "neon_version": version,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True, help="checkout scratch dir")
    p.add_argument("--logical-name", default=LOGICAL_NAME)
    p.add_argument("--base-domain", default="castform.dev")
    p.add_argument("--commit", default=HANDBOOK_COMMIT)
    p.add_argument("--batch-size", type=int, default=300)
    p.add_argument(
        "--provenance-out",
        type=Path,
        help="write the build provenance json here (relative to cwd)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    provenance = build_and_ingest(
        work_dir=args.work_dir,
        logical_name=args.logical_name,
        base_domain=args.base_domain,
        commit=args.commit,
        batch_size=args.batch_size,
    )
    provenance["ingest_seconds"] = round(time.time() - t0, 1)
    text = json.dumps(provenance, indent=2, sort_keys=True)
    if args.provenance_out:
        args.provenance_out.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
