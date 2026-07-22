"""NeonSearch — the pickle-safe, env-facing SearchClient over a Neon corpus.

Structurally implements the ``SearchClient`` protocol (``search`` / ``embed`` /
``available_modes`` / ``get_params``) without inheriting it, mirroring
``TpufSearch``: no psycopg import at module load, the connection resolved per call
via a read-only DSN provider, and the live client nulled across pickling so the
env bundle never carries a socket.

Credential seam (Contract #2, moved to Slice 4 per B1)
------------------------------------------------------
The read-only DSN rides the ``str | TokenProvider | None`` seam (see
``credentials.py``): ``None`` reads ``NEON_CORPUS_DSN_RO`` from the environment at
query time (self-serve — the DSN stays OUT of the pickled artifact), a literal
``str`` bakes the resolved DSN into the pickle (the platform-orchestrated path,
where the trainer's Ray actor can't read the env at runtime — an accepted at-rest
tradeoff, guarded by RO-scoping and never logging the DSN), and a callable is used
as-is. The provider is resolved LAZILY in :meth:`_get_client`; search always uses
the RO surface (SELECT only), never the RW ingest role.

The single hybrid-RRF fusion and the surfaced-score formula are owned by
``query.py``; this module re-exports their frozen public names
(:data:`SURFACED_RANK_K`, :class:`QueryHit`, :func:`surfaced_score`,
:class:`NeonQueryRequest`, :func:`fuse_rrf`) so ``castform.rag.corpus.neon.search``
stays a stable import path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from castform.platform.credentials import TokenProvider
from castform.rag.corpus.neon.credentials import resolve_read_dsn_provider
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA
from castform.rag.corpus.neon.query import (
    SURFACED_RANK_K,
    NeonQueryRequest,
    QueryHit,
    fuse_rrf,
    run_query,
    surfaced_score,
)
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG
from castform.rag.corpus.search_schema.search_types import SearchMode

if TYPE_CHECKING:
    from castform.rag.corpus.neon.client import NeonClient
    from castform.rag.corpus.neon.query import QueryRow
    from castform.rag.corpus.neon.schema import NeonTableSpec

__all__ = [
    "SURFACED_RANK_K",
    "NeonQueryRequest",
    "NeonSearch",
    "QueryHit",
    "fuse_rrf",
    "surfaced_score",
]

# ``search(mode="auto")`` resolves to the richest available mode, best-first.
_AUTO_MODE_PREFERENCE: tuple[SearchMode, ...] = ("hybrid", "vector", "lexical")


class NeonSearch:
    """Pickle-safe Neon corpus search client for RL environments.

    Args:
        table: Logical corpus name to query (resolved to the active-version view).
        embed_fn: Embedding function for vector/hybrid modes. ``Callable[[list[str]],
            list[list[float]]]`` — the shape every provider expects. When absent,
            only lexical search is available.
        dsn_provider: Read-only DSN, a provider callable, or ``None`` to read
            ``NEON_CORPUS_DSN_RO`` from the environment at query time.
        schema: Postgres schema the corpus objects live in (qualifies the BM25
            index regclass for the RO invoker). Not a credential.
        text_search_config: ``regconfig`` the corpus tsvector was built with; must
            match the version's baked config or bm25 scores drift.
    """

    def __init__(
        self,
        table: str,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        dsn_provider: str | TokenProvider | None = None,
        schema: str = CORPUS_SCHEMA,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
    ) -> None:
        self._table = table
        self._embed_fn = embed_fn
        self._schema = schema
        self._text_search_config = text_search_config
        self._dsn_provider = resolve_read_dsn_provider(dsn_provider)
        self._client: Any = None

    # --- lazy client + version resolution ------------------------------------

    def _get_client(self) -> NeonClient:
        """Build the ``NeonClient`` lazily from the resolved RO provider.

        Imports ``NeonClient`` here (not at module load) so this class stays
        pickle-safe with the ``neon`` extra absent. The provider — not a resolved
        DSN — is handed to the client, which re-resolves per connect so a rotated
        RO DSN is always fresh.
        """
        if self._client is None:
            from castform.rag.corpus.neon.client import NeonClient

            self._client = NeonClient(self._dsn_provider)
        return self._client

    def _resolve_spec(self, client: NeonClient) -> NeonTableSpec:
        """Resolve the current published version into a ``NeonTableSpec``.

        Reads the ledger for the ``is_current`` row (the version the reader view
        points to) so the BM25 leg can name that version's index regclass.
        """
        from castform.rag.corpus.neon.schema import NeonTableSpec

        rows = client.execute(client.read_ledger_sql(), {"logical": self._table})
        for version, _state, is_current in rows:
            if is_current:
                return NeonTableSpec(
                    self._table,
                    version,
                    text_search_config=self._text_search_config,
                )
        raise LookupError(
            f"neon corpus {self._table!r} has no current published version"
        )

    def _run(self, request: NeonQueryRequest) -> list[QueryRow]:
        client = self._get_client()
        spec = self._resolve_spec(client)
        return run_query(client, spec, request, schema=self._schema)

    # --- SearchClient protocol -----------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return structured results best-first.

        Returns dicts keyed ``content`` / ``source`` / ``metadata`` / ``score``,
        where ``score`` is the surfaced reciprocal-rank (higher-better, uniform
        across modes — never the raw native score). ``mode="auto"`` picks the
        richest available mode (hybrid > vector > lexical). Vector/hybrid embed
        ``query`` via ``embed_fn``.
        """
        resolved = self._resolve_mode(mode)
        request = NeonQueryRequest(
            mode=resolved,
            top_k=top_k,
            text=query,
            vector=self._embed(query) if resolved in ("vector", "hybrid") else None,
        )
        return [
            {
                "content": row.content,
                "source": row.source_file,
                "metadata": row.metadata,
                "score": row.surfaced_score,
            }
            for row in self._run(request)
        ]

    def embed(self, text: str) -> list[float] | None:
        """Return an embedding for *text*, or ``None`` if no embedder is set."""
        if self._embed_fn is None:
            return None
        return self._embed_fn([text])[0]

    @property
    def available_modes(self) -> list[str]:
        """Modes gated on ``embed_fn``: lexical-only without, +vector/+hybrid with."""
        modes = ["lexical"]
        if self._embed_fn is not None:
            modes += ["vector", "hybrid"]
        return sorted(modes)

    def get_params(self) -> dict[str, Any]:
        """Serializable connection params for inspection — NO credential."""
        return {"backend": "neon", "table": self._table, "schema": self._schema}

    # --- richer internal API (QueryRequest-driven) ---------------------------

    def query(self, request: NeonQueryRequest) -> list[QueryHit]:
        """Run one query, returning ``QueryHit`` rows best-first (id + scores)."""
        return [row.to_hit() for row in self._run(request)]

    def search_content(self, request: NeonQueryRequest) -> list[str]:
        """Return content strings only (cloudpickle-safe rollout path)."""
        return [row.content for row in self._run(request)]

    # --- helpers -------------------------------------------------------------

    def _resolve_mode(self, mode: str) -> SearchMode:
        modes = self.available_modes
        if mode == "auto":
            return next(m for m in _AUTO_MODE_PREFERENCE if m in modes)
        if mode not in modes:
            hint = (
                " Provide embed_fn for vector/hybrid."
                if mode in ("vector", "hybrid")
                else ""
            )
            raise ValueError(
                f"NeonSearch: mode {mode!r} not available. Available: {modes}.{hint}"
            )
        return mode  # type: ignore[return-value]

    def _embed(self, query: str) -> tuple[float, ...]:
        if self._embed_fn is None:
            raise ValueError("vector/hybrid search requires embed_fn")
        return tuple(self._embed_fn([query])[0])

    # --- pickle safety -------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None  # never pickle a live psycopg connection
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
