"""Example-local TurboPuffer adapter for Benchmax ``RagEnv``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from benchmax.rag.search import SearchResult

_RRF_K = 60.0
_BM25_MAX_QUERY_LENGTH = 1024


class TurbopufferSearch:
    """Lazy, pickle-safe lexical/vector/hybrid TurboPuffer search."""

    def __init__(
        self,
        namespace: str,
        *,
        region: str = "aws-us-east-1",
        content_field: str = "content",
        vector_field: str = "vector",
        distance_metric: str = "cosine_distance",
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        api_key: str,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._namespace = namespace
        self._region = region
        self._content_field = content_field
        self._vector_field = vector_field
        self._distance_metric = distance_metric
        self._embed_fn = embed_fn
        self._api_key = api_key
        self._client: Any = None

    @property
    def available_modes(self) -> list[str]:
        """Return configured modes without initializing the SDK or reading a key."""

        modes = ["lexical"]
        if self._embed_fn is not None:
            modes.extend(("vector", "hybrid"))
        return sorted(modes)

    def _get_namespace(self) -> Any:
        if self._client is None:
            import turbopuffer

            client = turbopuffer.Turbopuffer(
                api_key=self._api_key,
                region=self._region,
            )
            self._client = client.namespace(self._namespace)
        return self._client

    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        resolved = self._resolve_mode(mode)
        namespace = self._get_namespace()
        if resolved == "lexical":
            rows = await self._query(
                namespace,
                rank_by=(self._content_field, "BM25", query[:_BM25_MAX_QUERY_LENGTH]),
                top_k=top_k,
            )
            return self._results(rows)

        vector = await self._embed(query)
        if resolved == "vector":
            rows = await self._query(
                namespace,
                rank_by=(self._vector_field, "ANN", vector),
                top_k=top_k,
                distance_metric=self._distance_metric,
            )
            return self._results(rows)

        candidate_count = min(top_k * 2, 10_000)
        lexical, dense = await asyncio.gather(
            self._query(
                namespace,
                rank_by=(self._content_field, "BM25", query[:_BM25_MAX_QUERY_LENGTH]),
                top_k=candidate_count,
            ),
            self._query(
                namespace,
                rank_by=(self._vector_field, "ANN", vector),
                top_k=candidate_count,
                distance_metric=self._distance_metric,
            ),
        )
        return self._fuse(lexical, dense, top_k)

    @staticmethod
    async def _query(namespace: Any, **kwargs: Any) -> list[Any]:
        response = await asyncio.to_thread(
            namespace.query,
            include_attributes=True,
            **kwargs,
        )
        return list(response.rows or [])

    async def _embed(self, query: str) -> list[float]:
        if self._embed_fn is None:
            raise ValueError("vector and hybrid search require an embed_fn")
        return (await self._embed_fn([query]))[0]

    def _resolve_mode(self, mode: str) -> str:
        modes = self.available_modes
        if mode == "auto":
            return "hybrid" if "hybrid" in modes else "lexical"
        if mode not in modes:
            raise ValueError(f"search mode {mode!r} is unavailable; available modes: {modes}")
        return mode

    def _results(self, rows: list[Any]) -> list[SearchResult]:
        # Provider score directions differ between ANN and BM25. A rank score
        # keeps the rollout-facing value consistently higher-is-better.
        return [self._row_to_result(row, 1.0 / (rank + 1)) for rank, row in enumerate(rows)]

    def _fuse(
        self,
        lexical: list[Any],
        dense: list[Any],
        top_k: int,
    ) -> list[SearchResult]:
        fused: dict[Any, tuple[Any, float]] = {}
        for rows in (lexical, dense):
            for rank, row in enumerate(rows):
                current = fused.get(row.id)
                score = (current[1] if current else 0.0) + 1.0 / (_RRF_K + rank)
                fused[row.id] = (current[0] if current else row, score)
        ranked = sorted(fused.items(), key=lambda item: (-item[1][1], str(item[0])))
        return [self._row_to_result(row, score) for _, (row, score) in ranked[:top_k]]

    def _row_to_result(self, row: Any, score: float) -> SearchResult:
        attributes = dict(getattr(row, "model_extra", None) or {})
        content = attributes.pop(self._content_field, "")
        source = attributes.get("file") or attributes.get("file_path") or ""
        metadata = {
            key: value
            for key, value in attributes.items()
            if key not in {self._vector_field, "$dist"} and not key.startswith("$")
        }
        return {
            "content": str(content),
            "source": str(source),
            "metadata": metadata,
            "score": float(score),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None


__all__ = ["TurbopufferSearch"]
