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

from collections.abc import Callable

from benchmax import config
from benchmax.platform.credentials import resolve_judge_key

# The served, non-hidden embeddings model in the llm-proxy catalog.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


def platform_embed_fn(
    *,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str | None = None,
    api_key: str = "",
) -> Callable[[list[str]], list[list[float]]]:
    """Build a sync, batched ``embed_fn`` over the platform ``/v1/embeddings``.

    Returns ``Callable[[list[str]], list[list[float]]]`` — the shape every provider search
    client / chunk source expects. The OpenAI client is built lazily on first call so that
    (a) the closure stays cloudpickle-safe — no live httpx client is ever serialized into the
    env bundle — and (b) ``base_url`` and credentials resolve in the *sandbox* where the env
    runs, picking up its ``CASTFORM_PROFILE`` rather than the authoring host's.

    Auth mirrors the LLM judge (``resolve_judge_key``): an explicit ``api_key`` wins, else the
    platform credential seam (``ACT_AS_TOKEN_PATH`` in training, ``PLATFORM_API_KEY`` / session
    JWT otherwise). ``base_url`` defaults to ``config.llm_url()`` and may be overridden.
    """
    client = None  # built on first call; keeps the closure picklable + env-correct

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal client
        if client is None:
            from openai import OpenAI

            url = base_url or config.llm_url()
            client = OpenAI(base_url=url, api_key=resolve_judge_key(api_key, url))
        resp = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]

    return embed
