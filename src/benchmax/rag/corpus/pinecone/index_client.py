"""PineconeIndexClient — low-level Pinecone SDK wrapper.

Handles query building, batch upsert, and ID pagination.  Used by both
``PineconeChunkSource`` (data prep) and ``PineconeSearch`` (RL env).

**No Chunk / Pydantic imports.**  All methods accept and return plain
Python types (dicts, lists, strings).  Chunk conversion is handled by
the source layer.

Pickle safety: the Pinecone ``Index`` object is created lazily on first use,
so only serializable config (api_key, index_name, etc.) is stored.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm

#: Default mapping from Pinecone metadata keys → internal field names.
#: Used when no ``field_mapping`` is provided to :class:`PineconeIndexClient`.
#: Keys are Pinecone metadata field names, values are internal names used by
#: ``Chunk.get_metadata()``.  Override via ``field_mapping`` for "bring your
#: own index" scenarios.
DEFAULT_FIELD_MAPPING: dict[str, str] = {
    "content": "content",
    "file_path": "file_path",
    "chunk_index": "chunk_index",
    "h1": "h1",
    "h2": "h2",
    "h3": "h3",
    "chunk_hash": "chunk_hash",
    "char_count": "char_count",
}


class PineconeIndexClient:
    """Thin wrapper around a Pinecone index.

    Encapsulates query construction, row conversion, upsert logic, and
    ID management so that ``PineconeChunkSource`` can focus on the
    ChunkSource protocol.

    Embedding is required for all search operations.  Supply **one** of:

    * ``embed_fn`` — a custom callable (e.g. wrapping Azure OpenAI).
    * ``embed_model`` — name of a Pinecone hosted model (e.g.
      ``"multilingual-e5-large"``).  The Pinecone Inference API is
      called automatically using the same ``api_key``.

    Args:
        api_key: Pinecone API key.
        index_name: Name of the Pinecone index.
        index_host: Optional host URL. When provided, ``index_name`` is
            ignored and the client connects directly to this host.
        namespace: Pinecone namespace within the index (default ``""``).
        embed_fn: Custom embedding function ``list[str] → list[list[float]]``.
        embed_model: Pinecone hosted embedding model name.  Ignored when
            ``embed_fn`` is provided.  Defaults to
            ``"multilingual-e5-large"``.
        field_mapping: Low-level escape hatch — maps *Pinecone metadata
            field names* → *internal field names* for schemas that also
            relocate structural fields (``file_path``, ``chunk_index``,
            headers).  For the common "my text is under a different key"
            case, prefer ``content_field``.
        content_field: Pinecone metadata key holding the chunk text, for
            "bring your own index" schemas that don't use ``content`` (e.g.
            ``"summary"`` or ``"passage"``).  The canonical way to point at
            your text column.  Empty / None means the default ``content``
            key.  Raises if ``field_mapping`` already maps a *different*
            key to ``content``.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        *,
        index_host: str | None = None,
        namespace: str = "",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_model: str = "multilingual-e5-large",
        field_mapping: dict[str, str] | None = None,
        content_field: str | None = None,
    ) -> None:
        # Store config for lazy init / pickle safety.
        self._api_key = api_key
        self._index_name = index_name
        self._index_host = index_host
        # Platform codegen may pass None for an unset namespace; Pinecone's
        # default namespace is "".
        self._namespace = namespace or ""
        self._embed_model = embed_model
        self.embed_fn = embed_fn or self._build_pinecone_embed_fn()
        mapping = dict(field_mapping) if field_mapping else dict(DEFAULT_FIELD_MAPPING)
        if content_field and content_field != "content":
            conflicting = [
                k
                for k, v in mapping.items()
                if v == "content" and k not in ("content", content_field)
            ]
            if field_mapping and conflicting:
                raise ValueError(
                    f"content_field={content_field!r} conflicts with field_mapping "
                    f"entries {conflicting} that already map to 'content'. "
                    "Specify the text column one way or the other."
                )
            # Drop the default content→content entry so the reverse mapping
            # resolves "content" to the custom key unambiguously.
            mapping.pop("content", None)
            mapping[content_field] = "content"
        self._field_mapping = mapping
        # Reverse mapping: internal name → pinecone metadata key
        self._reverse_mapping = {v: k for k, v in self._field_mapping.items()}
        self._index: Any | None = None
        # Local ID list for sampling (populated on upsert or explicit refresh).
        self._known_ids: list[str] | None = None
        # Cached vector dimension (detected on first embed or describe_index).
        self._vector_dim: int | None = None
        # Cached index vector type ("dense" | "sparse"), probed lazily.
        self._vector_type: str | None = None

    def _build_pinecone_embed_fn(self) -> Callable[[list[str]], list[list[float]]]:
        """Build an embed_fn using Pinecone's hosted Inference API.

        The Pinecone client is created once (lazily on first call) and
        reused for all subsequent embed calls.  Includes retry with
        exponential backoff for rate limits (429).
        """
        import time as _time

        api_key = self._api_key
        model = self._embed_model
        _pc_cache: list = []  # lazy singleton — avoids creating client on init

        def embed_fn(texts: list[str]) -> list[list[float]]:
            if not _pc_cache:
                from pinecone import Pinecone

                _pc_cache.append(Pinecone(api_key=api_key))
            pc = _pc_cache[0]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = pc.inference.embed(
                        model=model,
                        inputs=texts,
                        parameters={"input_type": "passage"},
                    )
                    return [item.values for item in result.data]
                except Exception as exc:
                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        wait = 2**attempt
                        print(
                            f"Rate limited, retrying in {wait}s"
                            f" (attempt {attempt + 1}/{max_retries})..."
                        )
                        _time.sleep(wait)
                    else:
                        raise
            # Final attempt without catching
            result = pc.inference.embed(
                model=model,
                inputs=texts,
                parameters={"input_type": "passage"},
            )
            return [item.values for item in result.data]

        return embed_fn

    # ------------------------------------------------------------------
    # Lazy client init
    # ------------------------------------------------------------------

    def _get_index(self) -> Any:
        """Return the Pinecone ``Index``, creating it on first access."""
        if self._index is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self._api_key)
            if self._index_host:
                self._index = pc.Index(host=self._index_host)
            else:
                self._index = pc.Index(self._index_name)
        return self._index

    def vector_type(self) -> str:
        """Return the index vector type, ``"dense"`` or ``"sparse"``.

        Probes the index via ``describe_index_stats`` on first call and
        caches the result.
        """
        if self._vector_type is None:
            index = self._get_index()
            stats = index.describe_index_stats()
            self._vector_type = getattr(stats, "vector_type", None) or "dense"
        return self._vector_type

    def namespace_vector_count(self) -> int:
        """Return the vector count for this client's namespace.

        Scoped to the namespace, NOT the index-wide total — an index-wide
        count would disagree with what list/fetch/query in this namespace
        can actually see.  The SDK keys the default namespace as
        ``"__default__"`` (the REST API uses ``""``).
        """
        stats = self._get_index().describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or {}
        ns_stats = namespaces.get(self._namespace or "__default__")
        if ns_stats is None and not self._namespace:
            ns_stats = namespaces.get("")
        if ns_stats is None:
            return 0
        return int(getattr(ns_stats, "vector_count", 0) or 0)

    def zero_vector(self) -> list[float]:
        """Return a zero-vector with the correct dimension for this index.

        Probes the index via ``describe_index_stats`` on first call and
        caches the result.  Used for metadata-only filtered queries where
        the vector value doesn't matter.
        """
        if self._vector_dim is None:
            index = self._get_index()
            stats = index.describe_index_stats()
            self._vector_dim = stats.dimension
        if self._vector_dim is None:
            # Sparse indexes have no fixed dimension.
            raise ValueError(
                f"Pinecone index '{self._index_name}' has no dimension — it is "
                "a sparse index, which has no dense zero-vector."
            )
        return [0.0] * self._vector_dim

    # ------------------------------------------------------------------
    # Field mapping helpers
    # ------------------------------------------------------------------

    def _pc_field(self, internal_name: str) -> str:
        """Return the Pinecone metadata key for an internal field name."""
        return self._reverse_mapping.get(internal_name, internal_name)

    def _internal_field(self, pc_name: str) -> str:
        """Return the internal field name for a Pinecone metadata key."""
        return self._field_mapping.get(pc_name, pc_name)

    # ------------------------------------------------------------------
    # Raw result conversion (no Chunk dependency)
    # ------------------------------------------------------------------

    def match_content(self, match: Any) -> str:
        """Return the text content from a Pinecone query match."""
        metadata = getattr(match, "metadata", {}) or {}
        content_key = self._pc_field("content")
        return str(metadata.get(content_key, ""))

    def match_to_raw(self, match: Any) -> dict[str, Any]:
        """Convert a Pinecone query match to a plain dict.

        Returns:
            Dict with keys: ``id``, ``content``, ``metadata``, ``score``.
            Metadata keys are mapped to internal field names.
        """
        metadata = getattr(match, "metadata", {}) or {}
        content_key = self._pc_field("content")
        content = str(metadata.get(content_key, ""))

        attrs: dict[str, Any] = {}
        for pc_key, value in metadata.items():
            internal = self._internal_field(pc_key)
            if internal != "content":
                attrs[internal] = value

        attrs["_pinecone_id"] = match.id
        return {
            "id": match.id,
            "content": content,
            "metadata": attrs,
            "score": getattr(match, "score", 0.0) or 0.0,
        }

    def fetch_to_raw(self, vec_id: str, vector_data: Any) -> dict[str, Any]:
        """Convert a Pinecone fetch result entry to a plain dict."""
        metadata = getattr(vector_data, "metadata", {}) or {}
        content_key = self._pc_field("content")
        content = str(metadata.get(content_key, ""))

        attrs: dict[str, Any] = {}
        for pc_key, value in metadata.items():
            internal = self._internal_field(pc_key)
            if internal != "content":
                attrs[internal] = value

        attrs["_pinecone_id"] = vec_id
        return {
            "id": vec_id,
            "content": content,
            "metadata": attrs,
            "score": 0.0,
        }

    # ------------------------------------------------------------------
    # Batch upsert
    # ------------------------------------------------------------------

    def upsert_raw(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        batch_size: int = 96,
        show_summary: bool = True,
    ) -> int:
        """Upload raw documents to the index.

        Accepts plain lists — no Chunk dependency.  Embedding is done
        internally via ``embed_fn``.

        Returns the total number of documents written.
        """
        index = self._get_index()

        if show_summary:
            print(f"\nUploading {len(documents)} chunks to Pinecone index '{self._index_name}'...")

        all_ids: list[str] = []

        for batch_start in tqdm(
            range(0, len(documents), batch_size),
            desc="Uploading batches",
            disable=not show_summary,
        ):
            batch_docs = documents[batch_start : batch_start + batch_size]
            batch_ids = ids[batch_start : batch_start + batch_size]
            batch_metas = metadatas[batch_start : batch_start + batch_size]
            vectors = self.embed_fn(batch_docs)

            upsert_batch: list[dict[str, Any]] = []
            for i, (vec_id, meta) in enumerate(zip(batch_ids, batch_metas)):
                all_ids.append(vec_id)
                upsert_batch.append(
                    {
                        "id": vec_id,
                        "values": vectors[i],
                        "metadata": meta,
                    }
                )

            index.upsert(vectors=upsert_batch, namespace=self._namespace)

        self._known_ids = all_ids

        if show_summary:
            print(f"\nUpload complete! {len(documents)} chunks written to index.")

        return len(documents)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> Any:
        """Run a vector query against the index."""
        if self.vector_type() == "sparse":
            # A dense query vector against a sparse index is rejected by
            # Pinecone with an opaque error; fail with an actionable one.
            raise ValueError(
                f"Pinecone index '{self._index_name}' is a sparse index — "
                "search against sparse indexes is not supported yet. "
                "Use a dense index."
            )
        index = self._get_index()
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "namespace": self._namespace,
        }
        if filter:
            kwargs["filter"] = filter
        return index.query(**kwargs)

    # ------------------------------------------------------------------
    # ID management & fetch
    # ------------------------------------------------------------------

    def list_all_ids(self, batch_size: int = 100) -> list[str]:
        """Return all vector IDs in the namespace via pagination.

        Pinecone's ``list_paginated`` returns pages of IDs.
        """
        if self._known_ids is not None:
            return list(self._known_ids)

        index = self._get_index()
        all_ids: list[str] = []
        pagination_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "namespace": self._namespace,
                "limit": batch_size,
            }
            if pagination_token:
                kwargs["pagination_token"] = pagination_token

            response = index.list_paginated(**kwargs)
            page_ids = [v.id for v in (response.vectors or [])]
            all_ids.extend(page_ids)

            pagination_token = getattr(getattr(response, "pagination", None), "next", None)
            if not pagination_token or not page_ids:
                break

        self._known_ids = all_ids
        return list(all_ids)

    def fetch_by_ids_raw(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch vectors by ID and return as plain dicts."""
        if not ids:
            return []
        index = self._get_index()
        response = index.fetch(ids=ids, namespace=self._namespace)
        vectors = response.vectors or {}
        return [
            self.fetch_to_raw(vid, vdata) for vid, vdata in vectors.items() if vdata is not None
        ]

    def sample_ids(self, n: int) -> list[str]:
        """Return up to *n* randomly sampled vector IDs."""
        all_ids = self.list_all_ids()
        if not all_ids:
            return []
        return random.sample(all_ids, min(n, len(all_ids)))

    def invalidate_id_cache(self) -> None:
        """Clear the cached ID list (call after populate/upsert)."""
        self._known_ids = None

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_index"] = None  # Drop non-serializable client
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._index = None
