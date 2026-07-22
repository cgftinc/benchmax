"""Platform-backed query/document embedding for RAG provider search clients.

The provider search clients and chunk sources (Turbopuffer / Pinecone / Chroma) accept an
optional ``embed_fn: Callable[[list[str]], list[list[float]]]``. Wiring one in makes vector /
hybrid retrieval work regardless of how the user's index was built — turbopuffer vector/hybrid
(no server-side embed), a pinecone index NOT on the hosted model, or a non-cloud chroma
collection. This builds that ``embed_fn`` over the platform's ``/v1/embeddings`` endpoint
(``text-embedding-3-large``), the same llm-proxy the LLM judge uses.

``qa-gen`` does NOT need this (it reads chunks directly); it's only for retrieval.
"""

from __future__ import annotations

from typing import Any

from castform import config
from castform.platform.credentials import resolve_judge_key

# The served, non-hidden embeddings model in the llm-proxy catalog.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


class PlatformEmbedFn:
    """Pickle-aware, batched ``embed_fn`` over the platform ``/v1/embeddings``.

    A callable object (``Callable[[list[str]], list[list[float]]]``) — the shape every
    provider search client / chunk source expects — that stays cloudpickle-safe **even
    after it has been warmed**. The live ``OpenAI`` (httpx) client is cached in an instance
    attribute and dropped by ``__getstate__``, so pickling a warmed instance never drags a
    live socket into the env bundle; ``__setstate__`` rebuilds it lazily on first post-unpickle
    call. This closes the gap a bare closure leaves: a closure cell holding a warmed client
    cannot be nulled by the enclosing object's ``__getstate__``, so a warm-then-pickle would
    serialize the live client.

    Config (model / base_url / api_key) is resolved at **call** time, not construction, so the
    ``base_url`` and credentials bind in the *sandbox* where the env runs (picking up its
    ``CASTFORM_BASE_DOMAIN``), not the authoring host's. Auth mirrors the LLM judge
    (``resolve_judge_key``): an explicit ``api_key`` wins, else the platform credential seam.

    Args:
        model: Embeddings model id (default ``text-embedding-3-large``).
        base_url: Override for the ``/v1`` base; defaults to ``config.llm_url()`` at call time.
        api_key: Explicit bearer; empty falls through to the credential seam.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str | None = None,
        api_key: str = "",
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client: Any = None  # built on first call; dropped across pickling

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            from openai import OpenAI

            url = self._base_url or config.llm_url()
            self._client = OpenAI(
                base_url=url, api_key=resolve_judge_key(self._api_key, url)
            )
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None  # never serialize a live client (B2)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None


def platform_embed_fn(
    *,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str | None = None,
    api_key: str = "",
) -> PlatformEmbedFn:
    """Build a pickle-aware platform ``embed_fn`` (see :class:`PlatformEmbedFn`).

    Thin factory kept for call-site compatibility: returns a :class:`PlatformEmbedFn`, which
    is a drop-in ``Callable[[list[str]], list[list[float]]]``.
    """
    return PlatformEmbedFn(model=model, base_url=base_url, api_key=api_key)
