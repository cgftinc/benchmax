"""TpufSearch — pickle-safe search client for RL environments.

Uses :class:`TpufNamespace` for queries, returns content strings only.
No Chunk or Pydantic dependency at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TpufSearch:
    """Pickle-safe Turbopuffer search client for RL environments.

    Supports lexical (BM25), vector, and hybrid search depending on
    whether ``embed_fn`` is provided.

    Args:
        api_key: Turbopuffer API key.
        namespace: Turbopuffer namespace name.
        region: Turbopuffer region (default ``"aws-us-east-1"``).
        content_attr: List of BM25-indexed content fields.
        embed_fn: Custom embedding function. Required for vector/hybrid.
        vector_attr: Vector attribute name (default ``"vector"``).
        distance_metric: Distance metric (default ``"cosine_distance"``).
    """

    def __init__(
        self,
        api_key: str,
        namespace: str,
        *,
        region: str = "aws-us-east-1",
        content_attr: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        vector_attr: str = "vector",
        distance_metric: str = "cosine_distance",
    ) -> None:
        self._api_key = api_key
        self._namespace = namespace
        self._region = region
        self._content_attr = content_attr
        self._embed_fn = embed_fn
        self._vector_attr = vector_attr
        self._distance_metric = distance_metric
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the Turbopuffer namespace client.

        Uses the ``turbopuffer`` SDK directly (no cgft dependency)
        so this class remains pickle-safe for remote training.
        """
        if self._client is None:
            import turbopuffer

            tpuf = turbopuffer.Turbopuffer(api_key=self._api_key, region=self._region)
            self._client = tpuf.namespace(self._namespace)
        return self._client

    @staticmethod
    def _row_content(row: Any) -> str:
        """Extract content string from a Turbopuffer row."""
        # Try model_extra first (newer SDK), then getattr
        extras = getattr(row, "model_extra", None) or {}
        content = extras.get("content") or getattr(row, "content", None)
        return str(content) if content else ""

    def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return structured results."""
        ns = self._get_client()
        modes = self.available_modes
        content_fields = self._content_attr or ["content"]

        if mode == "auto":
            if "hybrid" in modes:
                mode = "hybrid"
            elif "vector" in modes:
                mode = "vector"
            else:
                mode = "lexical"

        if mode not in modes:
            raise ValueError(
                f"TpufSearch: mode '{mode}' not available. "
                f"Available modes: {modes}. "
                f"{'Provide embed_fn for vector/hybrid.' if mode in ('vector', 'hybrid') else ''}"
            )

        if mode == "lexical":
            rank_by = [content_fields[0], "BM25", query]
            result = ns.query(rank_by=rank_by, top_k=top_k, include_attributes=True)
            return [self._row_to_result(row) for row in (result.rows or [])]

        if mode == "vector":
            vec = self._embed_fn([query])[0]
            rank_by = [self._vector_attr, "ANN", vec]
            result = ns.query(
                rank_by=rank_by,
                top_k=top_k,
                include_attributes=True,
                distance_metric=self._distance_metric,
            )
            return [self._row_to_result(row) for row in (result.rows or [])]

        # hybrid: client-side RRF
        vec = self._embed_fn([query])[0]
        lex_rank = [content_fields[0], "BM25", query]
        vec_rank = [self._vector_attr, "ANN", vec]
        oversample_k = min(top_k * 2, 10000)

        lex_result = ns.query(rank_by=lex_rank, top_k=oversample_k, include_attributes=True)
        vec_result = ns.query(
            rank_by=vec_rank,
            top_k=oversample_k,
            include_attributes=True,
            distance_metric=self._distance_metric,
        )

        k = 60.0
        fused: dict[Any, dict[str, Any]] = {}
        for rank, row in enumerate(lex_result.rows or []):
            fused.setdefault(row.id, {"row": row, "score": 0.0})
            fused[row.id]["score"] += 1.0 / (k + rank)
        for rank, row in enumerate(vec_result.rows or []):
            fused.setdefault(row.id, {"row": row, "score": 0.0})
            fused[row.id]["score"] += 1.0 / (k + rank)

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [self._row_to_result(e["row"], score=e["score"]) for e in ranked[:top_k]]

    def _row_to_result(self, row: Any, score: float | None = None) -> dict[str, Any]:
        """Convert a Turbopuffer row to a structured result dict."""
        extras = getattr(row, "model_extra", None) or {}
        content = extras.get("content") or getattr(row, "content", None) or ""
        source = extras.get("file") or extras.get("file_path") or ""
        metadata = {
            k: v
            for k, v in extras.items()
            if k not in ("content", "$dist") and not k.startswith("$")
        }
        return {
            "content": str(content),
            "source": str(source),
            "metadata": metadata,
            "score": float(score if score is not None else getattr(row, "$dist", 0.0) or 0.0),
        }

    def embed(self, text: str) -> list[float] | None:
        if self._embed_fn is None:
            return None
        return self._embed_fn([text])[0]

    @property
    def available_modes(self) -> list[str]:
        modes = ["lexical"]
        if self._embed_fn is not None:
            modes.extend(["vector", "hybrid"])
        return sorted(modes)

    def get_params(self) -> dict[str, Any]:
        return {
            "backend": "turbopuffer",
            "api_key": self._api_key[:8] + "...",
            "namespace": self._namespace,
            "region": self._region,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
