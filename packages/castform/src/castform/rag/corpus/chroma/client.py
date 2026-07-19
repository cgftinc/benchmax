"""ChromaClient — low-level Chroma SDK wrapper.

Handles lazy client/collection init, query execution, and content
extraction.  Used by both ``ChromaChunkSource`` (data prep) and
``ChromaSearch`` (RL env).

**No Chunk / Pydantic imports.**  All methods accept and return plain
Python types.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Sparse-key name used when setting up BM25 schema
BM25_KEY = "bm25_embedding"

# Embedding functions that run server-side on Chroma Cloud (embed.trychroma.com)
# — querying a collection that uses one never downloads a model. Everything else
# (default all-MiniLM, sentence-transformers / HF / Ollama / ONNX locals,
# third-party API EFs, or no EF) is treated as unsafe. Add hosted names here as
# they are verified server-side.
_SERVER_SIDE_EF_NAMES = frozenset({"chroma-cloud-qwen"})


def has_search_api() -> bool:
    """Return True when the chromadb package exposes the Search API."""
    try:
        from chromadb import Knn, Search  # noqa: F401

        return True
    except ImportError:
        return False


class ChromaClient:
    """Thin wrapper around ChromaDB client + collection.

    Handles lazy initialization, BM25 schema creation with graceful
    downgrade, and pickle safety.  No Chunk dependency.

    Args:
        collection_name: Name of the Chroma collection.
        host: Server hostname for client-server mode.
        port: Server port (default 8000).
        path: Directory for persistent-local mode.
        embed_fn: Custom embedding callable.
        content_attr: Metadata fields to treat as content.
        distance_metric: HNSW distance function.
        enable_bm25: Create BM25 sparse index (client-server only).
        rrf_k: RRF smoothing constant.
        rrf_oversample: RRF candidate multiplier.
        rrf_max_candidates: RRF candidate cap.
    """

    def __init__(
        self,
        collection_name: str,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int = 8000,
        path: str | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        content_attr: list[str] | None = None,
        distance_metric: str = "cosine",
        enable_bm25: bool = True,
        rrf_k: int = 60,
        rrf_oversample: int = 20,
        rrf_max_candidates: int = 200,
    ) -> None:
        self.collection_name = collection_name
        self.api_key = api_key
        self.tenant = tenant
        self.database = database
        self.host = host
        self.port = port
        self.path = path
        self.embed_fn = embed_fn
        self.content_attr: list[str] = content_attr or ["content"]
        self.distance_metric = distance_metric
        self.enable_bm25 = enable_bm25
        self.rrf_k = rrf_k
        self.rrf_oversample = rrf_oversample
        self.rrf_max_candidates = rrf_max_candidates

        self._raw_client: Any = None
        self._collection: Any = None
        self._embed_dim: int | None = None
        self._total_count: int | None = None
        self.search_api = has_search_api()

        # Capabilities — may be downgraded on collection creation
        modes: set[str] = {"vector"}
        ranking: set[str] = {"cosine"}
        if self.search_api and enable_bm25 and self._is_remote():
            modes |= {"lexical", "hybrid"}
            ranking.add("bm25")

        self.modes = modes
        self.ranking = ranking

    def _is_cloud(self) -> bool:
        """True when configured for Chroma Cloud."""
        return self.api_key is not None and self.tenant is not None and self.database is not None

    def _is_remote(self) -> bool:
        """True for any client-server backend (Cloud or self-hosted HTTP)."""
        return self._is_cloud() or self.host is not None

    @property
    def is_client_server(self) -> bool:
        """Back-compat alias for any non-local backend (Cloud + HTTP)."""
        return self._is_remote()

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def get_client(self) -> Any:
        if self._raw_client is not None:
            return self._raw_client
        import chromadb

        if self._is_cloud():
            self._raw_client = chromadb.CloudClient(
                api_key=self.api_key,
                tenant=self.tenant,
                database=self.database,
            )
        elif self.host is not None:
            self._raw_client = chromadb.HttpClient(host=self.host, port=self.port)
        elif self.path is not None:
            self._raw_client = chromadb.PersistentClient(path=self.path)
        else:
            self._raw_client = chromadb.Client()
        return self._raw_client

    def get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        client = self.get_client()
        kwargs: dict[str, Any] = {"name": self.collection_name}
        if self.embed_fn is not None:
            kwargs["embedding_function"] = _WrapEmbedFn(self.embed_fn)

        created_with_schema = False
        if self.search_api and self.enable_bm25 and self.is_client_server:
            try:
                schema_kwargs = {**kwargs, "schema": self._build_schema()}
                self._collection = client.get_or_create_collection(**schema_kwargs)
                created_with_schema = True
            except Exception:
                self.modes = {"vector"}
                self.ranking = {"cosine"}

        # get_or_create_collection returns a PRE-EXISTING collection as-is: when
        # one already exists (e.g. linked from Chroma Cloud, built outside
        # benchmax), our BM25 schema is never applied and no exception fires. The
        # collection then has no `bm25_embedding` sparse index, so any lexical /
        # hybrid query raises "key not found in schema". Verify the index is
        # actually present and downgrade to vector-only when it isn't, so the
        # advertised capabilities match what the collection can serve.
        if created_with_schema and not self._has_bm25_index(self._collection):
            self.modes = {"vector"}
            self.ranking = {"cosine"}

        if not created_with_schema:
            metadata: dict[str, str] = {}
            if self.distance_metric:
                metadata["hnsw:space"] = self.distance_metric
            if metadata:
                kwargs["metadata"] = metadata
            self._collection = client.get_or_create_collection(**kwargs)

        # When the caller supplies no embed_fn we rely on the collection's own
        # (server-side) embedding function. Repair it if chromadb can't.
        if self.embed_fn is None:
            self._repair_cloud_embedding_function(self._collection)

        return self._collection

    def dense_embed_is_safe(self) -> bool:
        """True when a dense (vector) query embeds WITHOUT downloading a model.

        Safe only when we can produce vectors without a client-side model
        download: either a caller-supplied ``embed_fn``, or a Chroma-hosted
        server-side embedding function (embeds at embed.trychroma.com). Every
        other embedder — chromadb's default all-MiniLM, sentence-transformers /
        HuggingFace / Ollama / ONNX locals, third-party API EFs we lack keys
        for, or no EF at all — is treated as UNSAFE, so callers refuse the dense
        path rather than trigger a model download. Conservative by design: an
        unknown embedder is unsafe.
        """
        if self.embed_fn is not None:
            return True
        col = self._collection
        if col is None:
            return False
        try:
            ef = (col._model.configuration_json or {}).get("embedding_function") or {}
        except Exception:
            return False
        return ef.get("name") in _SERVER_SIDE_EF_NAMES

    @staticmethod
    def _repair_cloud_embedding_function(collection: Any) -> None:
        """Attach a working EF when chromadb can't rebuild a Cloud hosted one.

        A Chroma Cloud collection embedded with the hosted ``chroma-cloud-qwen``
        function stores ``task=None`` in its config, but chromadb's
        ``build_from_config`` asserts ``task`` is required and raises
        "Could not build embedding function chroma-cloud-qwen" on *any* text
        query (legacy ``query()`` and the Search API alike). The bug is present
        through at least chromadb 1.5.9 (latest at time of writing), so a version
        bump is not a fix.

        ``Collection._embed`` prefers an explicitly-set ``_embedding_function``
        over the config-loaded one, so we build the EF directly (its constructor
        accepts ``task=None``) and attach it — sidestepping the broken loader.
        The embedding itself still happens server-side at embed.trychroma.com, so
        results match what the collection was indexed with. Reads the raw
        ``configuration_json`` dict rather than the ``configuration`` property,
        because the property is exactly what triggers the failing rebuild.

        Best-effort and fully guarded: any deviation in chromadb internals or a
        missing API key leaves the collection untouched (current behavior).
        """
        # chromadb sets _embedding_function to a DefaultEmbeddingFunction when no
        # EF is supplied, and its _embed() ignores that default (falling through
        # to the config loader). So only a real, non-default EF means "already
        # resolved" — don't override that.
        existing = getattr(collection, "_embedding_function", None)
        if existing is not None and type(existing).__name__ != "DefaultEmbeddingFunction":
            return
        try:
            ef_config = (collection._model.configuration_json or {}).get(
                "embedding_function"
            ) or {}
        except Exception:
            return
        if ef_config.get("name") != "chroma-cloud-qwen":
            return
        try:
            from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
                ChromaCloudQwenEmbeddingFunction,
                ChromaCloudQwenEmbeddingModel,
            )

            raw_model = ef_config.get("model")
            model = (
                ChromaCloudQwenEmbeddingModel(raw_model)
                if raw_model
                else ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B
            )
            collection._embedding_function = ChromaCloudQwenEmbeddingFunction(
                model=model,
                task=ef_config.get("task"),
                api_key_env_var=ef_config.get("api_key_env_var", "CHROMA_API_KEY"),
            )
        except Exception:
            return

    @staticmethod
    def _has_bm25_index(collection: Any) -> bool:
        """True when *collection* exposes a usable BM25 sparse index.

        Mirrors Chroma's own query-time gate
        (``CollectionCommon._embed_knn_string_queries``): a string query against
        ``BM25_KEY`` only succeeds when the key is present in the schema with an
        enabled sparse-vector index that carries an embedding function. Any
        introspection failure is treated as "not ready" — vector-only is always
        safe, whereas falsely advertising BM25 surfaces as a hard error mid-run.
        """
        try:
            schema = collection.schema
        except Exception:
            return False
        if schema is None or BM25_KEY not in getattr(schema, "keys", {}):
            return False
        sparse = getattr(schema.keys[BM25_KEY], "sparse_vector", None)
        index = getattr(sparse, "sparse_vector_index", None)
        if index is None or not getattr(index, "enabled", False):
            return False
        return getattr(getattr(index, "config", None), "embedding_function", None) is not None

    def _build_schema(self) -> Any:
        from chromadb import Schema, SparseVectorIndexConfig
        from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

        schema = Schema()
        return schema.create_index(
            key=BM25_KEY,
            config=SparseVectorIndexConfig(
                embedding_function=ChromaBm25EmbeddingFunction(),
                source_key="#document",
                bm25=True,
            ),
        )

    # ------------------------------------------------------------------
    # Query (returns raw dicts)
    # ------------------------------------------------------------------

    def query_raw(
        self,
        text_query: str | None = None,
        vector_query: list[float] | None = None,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a legacy query() and return raw dicts.

        Returns list of dicts with keys: id, content, metadata, score.
        """
        col = self.get_collection()
        kwargs: dict[str, Any] = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if vector_query is not None:
            kwargs["query_embeddings"] = [vector_query]
        else:
            kwargs["query_texts"] = [text_query or ""]
        if where:
            kwargs["where"] = where

        result = col.query(**kwargs)
        return self._legacy_result_to_raw(result)

    def search_api_raw(
        self,
        text_query: str,
        vector_query: list[float] | None = None,
        mode: str = "vector",
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        hybrid_opts: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Search API query and return raw dicts."""
        from chromadb import K, Knn, Search

        search = Search()
        if where:
            search = search.where(where)

        opts = hybrid_opts or {}

        if mode == "lexical":
            rank = Knn(query=text_query, key=BM25_KEY)
        elif mode == "hybrid":
            from chromadb import Rrf

            vector_weight = opts.get("vector_weight", 1.0)
            lexical_weight = opts.get("lexical_weight", 1.0)
            candidates = min(top_k * self.rrf_oversample, self.rrf_max_candidates)
            dense_rank = Knn(
                query=(vector_query if vector_query is not None else text_query),
                return_rank=True,
                limit=candidates,
            )
            sparse_rank = Knn(
                query=text_query,
                key=BM25_KEY,
                return_rank=True,
                limit=candidates,
            )
            rank = Rrf(
                ranks=[dense_rank, sparse_rank],
                weights=[vector_weight, lexical_weight],
                k=self.rrf_k,
            )
        else:  # vector
            if vector_query is not None:
                rank = Knn(query=vector_query)
            else:
                rank = Knn(query=text_query)

        search = search.rank(rank).limit(top_k).select(K.DOCUMENT, K.METADATA, K.SCORE)
        result = self.get_collection().search(search)
        return self._search_api_result_to_raw(result)

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float] | None:
        if self.embed_fn is None:
            return None
        vec = self.embed_fn([text])[0]
        self._validate_embed_dim(vec)
        return vec

    def _validate_embed_dim(self, vec: list[float]) -> None:
        dim = len(vec)
        if self._embed_dim is None:
            self._embed_dim = dim
        elif dim != self._embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embed_dim}, got {dim}."
            )

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def get_total_count(self) -> int:
        if self._total_count is None:
            self._total_count = self.get_collection().count()
        return self._total_count

    # ------------------------------------------------------------------
    # Content extraction (no Chunk)
    # ------------------------------------------------------------------

    def extract_content(self, doc: str | None, meta: dict[str, Any]) -> str:
        """Extract content from a Chroma document + metadata dict."""
        if self.content_attr == ["content"]:
            return doc or ""
        if len(self.content_attr) == 1:
            return str(meta.get(self.content_attr[0], doc or ""))
        import json

        return json.dumps({f: meta.get(f, "") for f in self.content_attr}, default=str)

    # ------------------------------------------------------------------
    # Raw result conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _legacy_result_to_raw(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert legacy query() result to list of raw dicts."""
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        rows: list[dict[str, Any]] = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
            rows.append(
                {
                    "id": ids[i] if i < len(ids) else f"id_{i}",
                    "content": doc,
                    "metadata": _clean_metadata(meta),
                    "score": score,
                }
            )
        return rows

    @staticmethod
    def _search_api_result_to_raw(result: Any) -> list[dict[str, Any]]:
        """Convert Search API result to list of raw dicts."""
        if not hasattr(result, "rows"):
            return []
        payloads = result.rows()
        if not payloads:
            return []
        rows: list[dict[str, Any]] = []
        for row in payloads[0]:
            meta = row.get("metadata") or {}
            rows.append(
                {
                    "id": row.get("id", ""),
                    "content": row.get("document", ""),
                    "metadata": _clean_metadata(meta) if isinstance(meta, dict) else {},
                    "score": row.get("score", 0.0) or 0.0,
                }
            )
        return rows

    @staticmethod
    def _get_result_to_raw(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert get() result to list of raw dicts."""
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ids = result.get("ids") or []
        rows: list[dict[str, Any]] = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            rows.append(
                {
                    "id": ids[i] if i < len(ids) else f"id_{i}",
                    "content": doc,
                    "metadata": _clean_metadata(meta) if isinstance(meta, dict) else {},
                    "score": 0.0,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Pickle
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_raw_client"] = None
        state["_collection"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._raw_client = None
        self._collection = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _WrapEmbedFn:
    """Adapt a plain callable to chromadb's EmbeddingFunction protocol."""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
        self._fn = fn

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: N802
        return self._fn(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: N802
        return self._fn(input)

    @staticmethod
    def name() -> str:
        return "default"

    @staticmethod
    def is_legacy() -> bool:
        return False


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Filter metadata to JSON-serializable scalar values only."""
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, type(None)))}
