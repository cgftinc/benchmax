"""NeonChunkSource — ChunkSource implementation over a Neon corpus.

Structurally implements the ``ChunkSource`` protocol (duck-typed, never inherited)
over the versioned Neon (Postgres + pgvector + BM25) backend built in the lower
slices. It is a *driver-side* orchestrator (qa-gen host), NOT the pickle-safe
env-facing search client — that role belongs to ``search.NeonSearch``. So this
class holds live psycopg connections and is never carried into the sandbox bundle.

Responsibilities and the collaborators each uses:

- **ingest** (``populate_from_folder`` / ``populate_from_chunks``) drives the
  read-write ``NeonClient`` through the frozen versioned-replace lifecycle: chunk
  -> embed via ``embed_fn`` -> build a fresh physical version (``build_version``:
  create + insert + ANN/BM25/aux indexes + ``VACUUM``) -> atomically publish it
  (``activate``: pointer-swap the reader view, prior version retained for
  rollback). Re-ingesting changed content changes ``Chunk.hash`` (F10), so this is
  a versioned *replace*, never an in-place upsert that could leave stale rows.
- **search** (``search`` / ``search_content`` / ``search_text`` /
  ``search_related`` / ``embed_query``) runs through the read-only query layer
  (``query.run_query``) — the single owner of hybrid RRF and the surfaced score.
  This class re-uses it and never re-implements fusion or the filter->SQL path.
- **reads** (``sample_chunks`` / ``scan_chunks`` / ``get_chunk_with_context`` /
  ``get_top_level_chunks`` / ``get_chunk_count``) run the read-only ``NeonClient``
  read SQL against the active-version reader view.

``asearch_related`` is deliberately absent from this class's sync surface: the
async twin lives on a separate async protocol so adding it never widens the
runtime-checkable sync ``ChunkSource`` protocol (which would break the other
backends' conformance). See ``async_source.NeonAsyncChunkSource``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from castform.platform.credentials import TokenProvider
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.credentials import (
    resolve_read_dsn_provider,
    resolve_write_dsn_provider,
)
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA, RO_ROLE
from castform.rag.corpus.neon.query import NeonQueryRequest, run_query
from castform.rag.corpus.neon.schema import (
    DEFAULT_TEXT_SEARCH_CONFIG,
    EMBEDDING_DIM,
    NeonTableSpec,
    ReadGrantSpec,
    view_name,
)
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidSearchSpecError,
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchCapabilities,
    SearchMode,
    SearchSpec,
    validate_search_spec_shape,
)

if TYPE_CHECKING:
    from castform.rag.corpus.neon.query import QueryRow

# Total, pageable scan order (B6). These are TYPED, NOT NULL physical columns —
# ``source_file text`` and ``chunk_index integer`` (populated at ingest from the
# chunk's ``file``/``index`` metadata) plus ``id`` (the chunk hash) as the final
# tiebreak — backed by the ``scan`` btree. Using real columns (not JSONB
# extraction) avoids lexical ``chunk_index`` sorting (1,10,2), NULL-break keysets,
# and collation ambiguity.
SCAN_ORDER_BY = ("source_file", "chunk_index", "id")

# ``mode=None`` resolves to the richest available mode, best-first.
_AUTO_MODE_PREFERENCE: tuple[SearchMode, ...] = ("hybrid", "vector", "lexical")

# The neon filter layer serves the full type-directed operator set (Contract #3).
_FILTER_FIELD_OPS: frozenset[str] = frozenset(
    {"eq", "ne", "in", "gt", "gte", "lt", "lte", "contains_any", "contains_all"}
)
_FILTER_LOGICAL_OPS: frozenset[str] = frozenset({"and", "or", "not"})


class NeonIngestError(RuntimeError):
    """Ingest surfaced one or more per-file chunking failures (B9).

    Raised by ``populate_from_folder`` when any file cannot be read or chunked,
    instead of silently dropping it (which the base ``MarkdownChunker.chunk_folder``
    does). The message names every failing file so the corpus is fully diagnosed
    before any version is built — no partial ingest.
    """


class _QueryRunner:
    """Adapter that runs ``NeonQueryRequest``s through the frozen query layer.

    A thin seam over ``query.run_query`` (the single hybrid-RRF + surfaced-score
    owner) bound to one read-only ``NeonClient`` and the corpus's logical
    name/schema/text-search config. Injected as ``NeonChunkSource._search`` so
    tests can substitute a fake returning canned ``QueryRow``s.
    """

    def __init__(
        self,
        client: NeonClient,
        *,
        logical_name: str,
        schema: str,
        text_search_config: str,
    ) -> None:
        self._client = client
        self._logical_name = logical_name
        self._schema = schema
        self._text_search_config = text_search_config

    def query_rows(self, request: NeonQueryRequest) -> list[QueryRow]:
        """Run one request and return ranked content-bearing rows best-first."""
        return run_query(
            self._client,
            request,
            logical_name=self._logical_name,
            schema=self._schema,
            text_search_config=self._text_search_config,
        )


class NeonChunkSource:
    """Corpus backend backed by a versioned Neon (Postgres + pgvector) table.

    Args:
        logical_name: Stable logical corpus name (the active-version reader view).
        embed_fn: Embedding function for vector/hybrid modes and ingest. The shape
            every provider expects — ``Callable[[list[str]], list[list[float]]]``.
            Required for ingest (the ``embedding`` column is ``NOT NULL``); without
            it only lexical search is available.
        read_dsn_provider: Read-only DSN seam (search + reads). ``None`` reads
            ``NEON_CORPUS_DSN_RO`` from the environment per connection.
        write_dsn_provider: Read-write DSN seam (ingest). Separate from the read
            provider so a search-only handle cannot mutate the corpus.
        schema: Postgres schema the corpus objects live in (qualifies the BM25
            index regclass and the RO grant). Not a credential.
        text_search_config: ``regconfig`` the corpus tsvector is built with; must
            match the version's baked config or BM25 scores drift.
        ro_role: Read-only role the ingest grant publishes ``SELECT`` to.
    """

    def __init__(
        self,
        logical_name: str,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        read_dsn_provider: str | TokenProvider | None = None,
        write_dsn_provider: str | TokenProvider | None = None,
        schema: str = CORPUS_SCHEMA,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
        ro_role: str = RO_ROLE,
    ) -> None:
        self._logical_name = logical_name
        self._embed_fn = embed_fn
        self._read_dsn_provider = read_dsn_provider
        self._write_dsn_provider = write_dsn_provider
        self._schema = schema
        self._text_search_config = text_search_config
        self._ro_role = ro_role
        self.collection: ChunkCollection | None = None  # exposed for advanced users
        self._read_client: NeonClient | None = None
        self._write_client: NeonClient | None = None
        self._search: _QueryRunner | None = None
        self._active_version: int | None = None

    # --- lazy collaborators --------------------------------------------------

    def _reader(self) -> NeonClient:
        """Return the lazily-built read-only client (search + direct reads)."""
        if self._read_client is None:
            self._read_client = NeonClient(
                resolve_read_dsn_provider(self._read_dsn_provider)
            )
        return self._read_client

    def _writer(self) -> NeonClient:
        """Return the lazily-built read-write client (ingest lifecycle)."""
        if self._write_client is None:
            self._write_client = NeonClient(
                resolve_write_dsn_provider(self._write_dsn_provider)
            )
        return self._write_client

    def _search_runner(self) -> _QueryRunner:
        """Return the lazily-built query-layer adapter over the read-only client."""
        if self._search is None:
            self._search = _QueryRunner(
                self._reader(),
                logical_name=self._logical_name,
                schema=self._schema,
                text_search_config=self._text_search_config,
            )
        return self._search

    # --- ingest --------------------------------------------------------------

    def populate_from_folder(
        self,
        docs_path: str,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Chunk a folder deterministically and ingest it as a new corpus version.

        Chunks every matching file, surfacing per-file failures as a
        :class:`NeonIngestError` rather than the silent drop the base
        ``MarkdownChunker.chunk_folder`` performs (B9) — so the ingested chunk set
        is reproducible run to run and nothing is lost unnoticed. On success the
        collection is handed to :meth:`populate_from_chunks` (versioned replace).

        Args:
            docs_path: Folder of documents to chunk.
            min_chars: Minimum characters per chunk.
            max_chars: Maximum characters per chunk.
            overlap_chars: Character overlap between adjacent chunks.
            file_extensions: Extensions to process (default ``[".md", ".mdx"]``).
            batch_size: Chunks per embedding/insert batch.
            show_summary: Print a chunking summary.
        """
        collection = self._chunk_folder_strict(
            docs_path,
            min_chars=min_chars,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            file_extensions=file_extensions,
            show_summary=show_summary,
        )
        self.populate_from_chunks(
            collection, batch_size=batch_size, show_summary=show_summary
        )

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Ingest a pre-built collection as a new, atomically-published version.

        Embeds every chunk via ``embed_fn``, builds a fresh physical version in the
        frozen lifecycle order (:meth:`NeonClient.build_version`), then publishes it
        with an atomic pointer swap (:meth:`NeonClient.activate`) — the prior
        version's tables are retained for O(1) rollback (F10). Requires an
        ``embed_fn`` (the ``embedding`` column is ``NOT NULL``).

        Args:
            collection: Pre-built chunk collection to ingest.
            batch_size: Chunks per embedding batch.
            show_summary: Print an ingest summary.
        """
        if self._embed_fn is None:
            raise ValueError(
                "neon ingest requires an embed_fn — every chunk needs an embedding"
            )
        self.collection = collection
        version = self._next_version()
        spec = NeonTableSpec(
            self._logical_name, version, text_search_config=self._text_search_config
        )
        rows = self._build_rows(collection, batch_size)

        client = self._writer()
        client.build_version(spec, rows)
        client.activate(
            spec,
            ReadGrantSpec(
                schema=self._schema,
                view=view_name(self._logical_name),
                ro_role=self._ro_role,
            ),
        )
        self._active_version = version
        if show_summary:
            print(
                f"ingested {len(rows)} chunks into {self._logical_name} "
                f"v{version} (published)"
            )

    def rollback_version(self, target_version: int) -> None:
        """Re-point the reader view to a prior activated version (non-destructive).

        Delegates to :meth:`NeonClient.rollback`: prior physical tables are retained
        by ingest, so this is an O(1) pointer swap under the per-logical advisory
        lock, never a rebuild.
        """
        self._writer().rollback(self._logical_name, target_version)
        self._active_version = target_version

    def _chunk_folder_strict(
        self,
        docs_path: str,
        *,
        min_chars: int,
        max_chars: int,
        overlap_chars: int,
        file_extensions: list[str] | None,
        show_summary: bool,
    ) -> ChunkCollection:
        """Chunk a folder, raising on any per-file failure instead of dropping it.

        Deterministic: the file list is sorted (``rglob`` order is filesystem
        dependent) and one chunker instance is reused across files so the
        cross-file duplicate-hash guard stays effective. Directory structure is
        preserved via the folder-relative path (not the basename), so nested files
        neither collide nor break top-level detection. Failures are collected and
        raised together (:class:`NeonIngestError`) before any DB write.
        """
        from pathlib import Path

        from castform.rag.chunkers.markdown import MarkdownChunker

        exts = file_extensions if file_extensions is not None else [".md", ".mdx"]
        root = Path(docs_path).resolve()
        files = sorted({p for ext in exts for p in root.rglob(f"*{ext}")})

        chunker = MarkdownChunker(
            min_char=min_chars, max_char=max_chars, chunk_overlap=overlap_chars
        )
        all_chunks: list[Chunk] = []
        errors: dict[str, Exception] = {}
        for file_path in files:
            rel = str(file_path.relative_to(root))
            try:
                content = file_path.read_text(encoding="utf-8")
                is_mdx = file_path.suffix.lower() == ".mdx"
                all_chunks.extend(
                    chunker.chunk(content, file=rel, preprocess_mdx=is_mdx)
                )
            except Exception as exc:  # surface, never swallow (B9)
                errors[rel] = exc

        if errors:
            detail = "; ".join(f"{name}: {exc}" for name, exc in sorted(errors.items()))
            raise NeonIngestError(
                f"failed to chunk {len(errors)} file(s): {detail}"
            )
        if show_summary:
            print(f"chunked {len(all_chunks)} chunks from {len(files)} files")
        return ChunkCollection(all_chunks)

    def _build_rows(
        self, collection: ChunkCollection, batch_size: int
    ) -> list[tuple[Any, ...]]:
        """Embed every chunk and assemble rows in ``NeonClient.INSERT_COLUMNS`` order.

        Row shape ``(id, content, metadata, embedding, source_file, chunk_index)``:
        ``metadata`` is ``Jsonb``-wrapped (psycopg3 does not adapt a bare dict);
        ``source_file``/``chunk_index`` are drawn from the chunk's ``file``/``index``
        metadata (the typed scan-order columns); the embedding is a plain list that
        pgvector adapts once the version's types are registered.
        """
        from psycopg.types.json import Jsonb

        chunks = list(collection)
        if not chunks:
            return []
        vectors = self._embed_all([c.content for c in chunks], batch_size)
        # strict=True is a backstop; _embed_all already guarantees the 1:1 count.
        rows: list[tuple[Any, ...]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            metadata = chunk.metadata_dict
            rows.append(
                (
                    chunk.hash,
                    chunk.content,
                    Jsonb(metadata),
                    list(vector),
                    str(metadata.get("file", "")),
                    int(metadata.get("index", 0) or 0),
                )
            )
        return rows

    def _embed_all(self, contents: list[str], batch_size: int) -> list[list[float]]:
        """Embed *contents* in batches, one vector per input, preserving order.

        Each batch is checked for a 1:1 vector-per-chunk count BEFORE any row is
        built, so an ``embed_fn`` that returns a short batch fails loudly here
        instead of silently truncating the corpus (which would publish a version
        missing chunks). ``embed_fn`` is required (guarded by the caller).
        """
        assert self._embed_fn is not None  # guarded by populate_from_chunks
        step = max(batch_size, 1)
        vectors: list[list[float]] = []
        for start in range(0, len(contents), step):
            batch = contents[start : start + step]
            result = self._embed_fn(batch)
            if len(result) != len(batch):
                raise ValueError(
                    f"embed_fn returned {len(result)} vectors for {len(batch)} "
                    "chunks — each chunk must get exactly one embedding"
                )
            vectors.extend(result)
        return vectors

    def _next_version(self) -> int:
        """Return the next physical version = max existing ledger version + 1.

        Uses a side-effect-free ``to_regclass`` probe for the shared ledger table
        (returns NULL when it does not yet exist -> version 1) rather than letting
        an ``UndefinedTable`` abort the write connection's transaction. Ingest is a
        single-writer operation: the version is chosen here and reserved under the
        advisory lock inside ``build_version`` — a concurrent second ingest would
        fail loudly on the ledger primary key, never corrupt a version.
        """
        from psycopg import sql

        client = self._writer()
        probe = client.execute(
            sql.SQL("SELECT to_regclass(%(name)s)"), {"name": "neon_corpus_versions"}
        )
        if not probe or probe[0][0] is None:
            return 1
        rows = client.execute(client.read_ledger_sql(), {"logical": self._logical_name})
        return max((version for version, _state, _current in rows), default=0) + 1

    # --- reads ---------------------------------------------------------------

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return up to *n* random chunks with content at least *min_chars* long."""
        client = self._reader()
        rows = client.execute(
            client.sample_sql(self._logical_name), {"n": n, "min_chars": min_chars}
        )
        return [self._row_to_chunk(row) for row in rows]

    def scan_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        """Yield every chunk in the stable ``(file, index, id)`` order (B6).

        Streams the whole corpus within ONE read transaction holding the shared
        per-logical advisory lock (:meth:`NeonClient.scan_in_snapshot`), so a
        concurrent activation cannot swap the version mid-scan and interleave rows
        from two versions — full-corpus materialization (qa-gen) is reproducible
        run to run. The scan runs on its own dedicated connection.

        The inner iterator is closed in a ``finally`` so that early-abandoning this
        generator (partial iteration then ``.close()`` or drop) deterministically
        propagates closure inward — a bare ``for`` loop does NOT forward close to
        its iterator — committing/rolling back the transaction, closing the
        connection, and releasing the advisory lock immediately rather than at GC
        (a leaked scan would otherwise hold the shared lock and block activation).
        """
        inner = self._reader().scan_in_snapshot(self._logical_name, batch_size)
        try:
            for row in inner:
                yield self._row_to_chunk(row)
        finally:
            inner.close()

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return *chunk* with truncated previews of its same-file neighbors.

        Returns:
            Dict with keys ``chunk_content``, ``prev_chunk_preview``,
            ``next_chunk_preview``. Neighbors are looked up by the chunk's
            ``file``/``index`` metadata; without them the previews are placeholders.
        """
        source_file = chunk.get_metadata("file")
        index = chunk.get_metadata("index")
        if source_file is None or index is None:
            return {
                "chunk_content": chunk.chunk_str(),
                "prev_chunk_preview": "(no previous chunk)",
                "next_chunk_preview": "(no next chunk)",
            }
        client = self._reader()
        rows = client.execute(
            client.neighbors_sql(self._logical_name),
            {"source_file": source_file, "prev_index": index - 1, "next_index": index + 1},
        )
        by_index = {chunk_index: content for chunk_index, content in rows}
        prev = by_index.get(index - 1)
        nxt = by_index.get(index + 1)
        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": (
                self._preview(prev, max_chars, "leading")
                if prev is not None
                else "(no previous chunk)"
            ),
            "next_chunk_preview": (
                self._preview(nxt, max_chars, "trailing")
                if nxt is not None
                else "(no next chunk)"
            ),
        }

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return the first chunk of every source file (the top-level entry points)."""
        client = self._reader()
        rows = client.execute(client.top_level_sql(self._logical_name))
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk_count(self) -> int:
        """Return the total chunk count of the active corpus version."""
        client = self._reader()
        rows = client.execute(client.count_sql(self._logical_name))
        return int(rows[0][0]) if rows else 0

    # --- search --------------------------------------------------------------

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search related chunks; relevance-descending with a ``max_score`` (NB1).

        Runs each query through the query layer, skips the source chunk and its
        same-file adjacent neighbors, deduplicates by hash keeping the max surfaced
        reciprocal-rank score, and sorts by the full 3-tuple
        ``(len(queries), not same_file, max_score)`` descending. ``native_score``
        carries the raw backend score from the SAME winning hit that supplied
        ``max_score``.

        Returns:
            List of dicts keyed ``chunk``, ``queries``, ``same_file``,
            ``max_score``, ``native_score``. Sorted by relevance descending.
        """
        resolved = self._resolve_mode(mode)
        runner = self._search_runner()
        related: dict[str, dict] = {}
        for query in queries:
            rows = runner.query_rows(
                self._build_request(resolved, query, top_k, hybrid=hybrid)
            )
            self._accumulate_related(related, source, query, rows, top_k)
        return self._sorted_related(related)

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search using a structured spec and return chunks."""
        return [self._row_to_chunk_from_row(row) for row in self._search_rows(spec)]

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings only (cloudpickle-safe rollout path)."""
        return [row.content for row in self._search_rows(spec)]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Lexical text search with an optional metadata filter."""
        return self.search(
            SearchSpec(
                mode="lexical", text_query=text_query, top_k=top_k, filter=filter
            )
        )

    def embed_query(self, text: str) -> list[float] | None:
        """Return an embedding for *text*, or ``None`` when no ``embed_fn`` is set."""
        if self._embed_fn is None:
            return None
        return self._embed_fn([text])[0]

    def get_search_capabilities(self) -> SearchCapabilities:
        """Report Neon backend capabilities (modes gated on ``embed_fn``)."""
        modes = self._modes()
        ranking = {"bm25"}
        if self._embed_fn is not None:
            ranking |= {"cosine", "rrf"}
        return {
            "backend": "neon",
            "modes": modes,
            "filter_ops": {
                "field": set(_FILTER_FIELD_OPS),
                "logical": set(_FILTER_LOGICAL_OPS),
            },
            "ranking": ranking,
            "constraints": {"max_top_k": 10000, "vector_dimensions": EMBEDDING_DIM},
            "graph_expansion": True,
        }

    # --- search helpers ------------------------------------------------------

    def _search_rows(self, spec: SearchSpec) -> list[QueryRow]:
        """Validate a spec and run it, returning ranked rows (shared by search*)."""
        mode = spec.get("mode")
        modes = self._modes()
        if mode not in modes:
            raise UnsupportedSearchModeError(
                backend="neon", mode=str(mode), supported_modes={str(m) for m in modes}
            )
        shape_errors = validate_search_spec_shape(spec)
        if shape_errors:
            raise InvalidSearchSpecError(
                backend="neon", message="; ".join(shape_errors), spec=spec
            )
        vector_query = spec.get("vector_query")
        request = NeonQueryRequest(
            mode=mode,  # type: ignore[arg-type]
            top_k=int(spec.get("top_k", 10)),
            text=spec.get("text_query"),
            vector=tuple(vector_query) if vector_query else None,
            filter=spec.get("filter"),
            hybrid=spec.get("hybrid"),
        )
        return self._search_runner().query_rows(request)

    def _build_request(
        self,
        mode: SearchMode,
        text: str,
        top_k: int,
        *,
        hybrid: HybridOptions | None,
    ) -> NeonQueryRequest:
        """Build a request for one ``search_related`` query, embedding when needed."""
        vector = None
        if mode in ("vector", "hybrid"):
            vector = tuple(self._embed(text))
        return NeonQueryRequest(
            mode=mode, top_k=top_k, text=text, vector=vector, hybrid=hybrid
        )

    def _resolve_mode(self, mode: SearchMode | None) -> SearchMode:
        """Resolve ``mode`` (``None`` -> richest available), else raise."""
        modes = self._modes()
        if mode is None:
            return next(m for m in _AUTO_MODE_PREFERENCE if m in modes)
        if mode not in modes:
            raise UnsupportedSearchModeError(
                backend="neon", mode=str(mode), supported_modes={str(m) for m in modes}
            )
        return mode

    def _modes(self) -> set[SearchMode]:
        """Available modes: lexical always; +vector/+hybrid when an embedder is set."""
        modes: set[SearchMode] = {"lexical"}
        if self._embed_fn is not None:
            modes |= {"vector", "hybrid"}
        return modes

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn is None:
            raise ValueError("vector/hybrid search requires an embed_fn")
        return self._embed_fn([text])[0]

    @staticmethod
    def _accumulate_related(
        related: dict[str, dict],
        source: Chunk,
        query: str,
        rows: list[QueryRow],
        top_k: int,
    ) -> None:
        """Merge one query's rows: skip source + same-file neighbors, dedup by hash.

        Keeps the max surfaced score per chunk and carries ``native_score`` from
        the SAME winning hit (updated only when a strictly higher surfaced score
        replaces the running max).
        """
        source_file = source.get_metadata("file")
        source_index = source.get_metadata("index", 0)
        for row in rows[:top_k]:
            if row.chunk_id == source.hash:
                continue
            same_file = source_file is not None and row.source_file == source_file
            if same_file and abs(row.chunk_index - source_index) <= 1:
                continue
            entry = related.get(row.chunk_id)
            if entry is None:
                related[row.chunk_id] = entry = {
                    "chunk": NeonChunkSource._row_to_chunk_from_row(row),
                    "queries": [],
                    "same_file": same_file,
                    "max_score": row.surfaced_score,
                    "native_score": row.native_score,
                }
            elif row.surfaced_score > entry["max_score"]:
                entry["max_score"] = row.surfaced_score
                entry["native_score"] = row.native_score
            entry["queries"].append(query)

    @staticmethod
    def _sorted_related(related: dict[str, dict]) -> list[dict]:
        """Sort: most matching queries, then cross-file, then max score — all desc."""
        return sorted(
            related.values(),
            key=lambda item: (
                len(item["queries"]),
                not item["same_file"],
                item["max_score"],
            ),
            reverse=True,
        )

    # --- row -> Chunk mapping -------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
        """Map a ``READ_COLUMNS`` row ``(id, content, metadata, file, index)``.

        ``hash=id`` is passed explicitly so identity survives the round trip
        regardless of JSONB key ordering (a recomputed hash could drift).
        """
        chunk_id, content, metadata, _source_file, _chunk_index = row
        return Chunk(
            content=content, metadata=tuple((metadata or {}).items()), hash=chunk_id
        )

    @staticmethod
    def _row_to_chunk_from_row(row: QueryRow) -> Chunk:
        """Map a ranked :class:`QueryRow` to a Chunk, preserving its hash identity."""
        return Chunk(
            content=row.content,
            metadata=tuple((row.metadata or {}).items()),
            hash=row.chunk_id,
        )

    @staticmethod
    def _preview(content: str, max_chars: int, truncate: str) -> str:
        """Truncate *content* to *max_chars* (mirrors ``Chunk.chunk_str`` truncation)."""
        if len(content) <= max_chars:
            return content
        if truncate == "leading":
            return "..." + content[-(max_chars - 3) :]
        return content[: max_chars - 3] + "..."
