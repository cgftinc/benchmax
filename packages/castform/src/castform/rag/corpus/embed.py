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

import inspect
import logging
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from castform import config
from castform.platform.credentials import resolve_judge_key_with_source

# The served, non-hidden embeddings model in the llm-proxy catalog.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"

logger = logging.getLogger(__name__)

_BEARER_SET_BY = "castform.rag.corpus.embed.PlatformEmbedFn.__call__"


def _safe_embeddings_endpoint(base_url: str) -> str:
    """Return a credential-free embeddings URL suitable for logs."""
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<invalid>"
    if parsed.scheme not in {"http", "https"} or not hostname:
        return "<invalid>"
    safe_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{safe_host}:{port}" if port is not None else safe_host
    path = f"{parsed.path.rstrip('/')}/embeddings"
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def _caller_path() -> str:
    """Return the immediate caller name without collecting a stack dump."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is None:
            return "<unknown>"
        module = caller.f_globals.get("__name__", "")
        name = caller.f_code.co_qualname
        return f"{module}.{name}" if module else name
    finally:
        del frame


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

    The base URL and credentials first resolve in the *sandbox* where the env
    runs (picking up its ``CASTFORM_BASE_DOMAIN``), not the authoring host's.
    Auth prefers an explicit ``api_key``, then the non-expiring
    ``PLATFORM_API_KEY``; clients using either stable bearer are cached. If
    neither exists, the general LLM-judge credential seam remains the
    compatibility fallback. That fallback is re-resolved into a request-scoped
    client on every call so a rotating ``ACT_AS_TOKEN_PATH`` token is never
    frozen in a cached embeddings client.

    Args:
        model: Embeddings model id (default ``text-embedding-3-large``).
        base_url: Override for the ``/v1`` base; defaults to ``config.llm_url()`` at call time.
        api_key: Explicit static bearer; empty prefers ``PLATFORM_API_KEY``
            before falling through to the general credential seam.
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
        self._endpoint = "<unresolved>"
        self._bearer_source_class = "unresolved"
        self._bearer_source = "unresolved"
        self._traced_bearer_states: set[tuple[str, str]] = set()
        self._traced_token_states: set[str] = set()
        self._traced_failure = False

    def __call__(self, texts: list[str]) -> list[list[float]]:
        client = self._client
        caller_path = _caller_path()
        bearer_resolution = "reused_cached"
        client_cache_state = "hit_reused"
        token_state = "stable_cached"
        close_client = False
        if client is None:
            from openai import OpenAI

            url = self._base_url or config.llm_url()
            platform_api_key = os.environ.get("PLATFORM_API_KEY", "")
            bearer_resolution = "fresh_resolved"
            client_cache_state = "miss_stored"
            if self._api_key:
                resolved_key = self._api_key
                self._bearer_source_class = "explicit_api_key"
                self._bearer_source = "constructor_arg"
            elif platform_api_key:
                resolved_key = platform_api_key
                self._bearer_source_class = "platform_api_key"
                self._bearer_source = "PLATFORM_API_KEY"
            else:
                resolved_key, self._bearer_source = resolve_judge_key_with_source(
                    "",
                    url,
                )
                if resolved_key is None:
                    self._bearer_source_class = "openai_api_key"
                else:
                    self._bearer_source_class = "credential_seam"
            self._endpoint = _safe_embeddings_endpoint(url)
            client = OpenAI(
                base_url=url,
                api_key=resolved_key,
            )
            if self._bearer_source_class == "credential_seam":
                token_state = "fresh_resolved"
                client_cache_state = "request_scoped"
                close_client = True
            else:
                self._client = client
        self._trace_bearer(
            bearer_resolution=bearer_resolution,
            client_cache_state=client_cache_state,
            caller_path=caller_path,
        )
        try:
            raw_response = client.embeddings.with_raw_response.create(
                model=self._model,
                input=texts,
            )
            status_code: int | str = raw_response.status_code
            resp = raw_response.parse()
        except Exception as exc:
            status_code = getattr(exc, "status_code", "unavailable")
            if not isinstance(status_code, int):
                status_code = "unavailable"
            self._trace_request(status_code, token_state)
            raise
        finally:
            if close_client:
                client.close()
        self._trace_request(status_code, token_state)
        return [item.embedding for item in resp.data]

    def _trace_bearer(
        self,
        *,
        bearer_resolution: str,
        client_cache_state: str,
        caller_path: str,
    ) -> None:
        """Log the first occurrence of each bearer/client cache state."""
        runtime_state = (bearer_resolution, client_cache_state)
        if runtime_state in self._traced_bearer_states:
            return
        self._traced_bearer_states.add(runtime_state)
        logger.info(
            "[PlatformEmbedFn] embeddings_bearer endpoint=%s bearer_source=%s "
            "bearer_resolution=%s client_cache_state=%s bearer_set_by=%s "
            "caller_path=%s",
            self._endpoint,
            self._bearer_source,
            bearer_resolution,
            client_cache_state,
            _BEARER_SET_BY,
            caller_path,
        )

    def _trace_request(
        self,
        status_code: int | str,
        token_state: str,
    ) -> None:
        """Log first auth-state occurrences and the first failed request."""
        failed = not isinstance(status_code, int) or not 200 <= status_code < 300
        if failed:
            if self._traced_failure:
                return
            self._traced_failure = True
        else:
            if token_state in self._traced_token_states:
                return
            self._traced_token_states.add(token_state)
        logger.info(
            "[PlatformEmbedFn] embeddings_request endpoint=%s status_code=%s "
            "bearer_source_class=%s bearer_source=%s token_state=%s",
            self._endpoint,
            status_code,
            self._bearer_source_class,
            self._bearer_source,
            token_state,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None  # never serialize a live client (B2)
        state["_endpoint"] = "<unresolved>"
        state["_bearer_source_class"] = "unresolved"
        state["_bearer_source"] = "unresolved"
        state["_traced_bearer_states"] = set()
        state["_traced_token_states"] = set()
        state["_traced_failure"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
        self._endpoint = "<unresolved>"
        self._bearer_source_class = "unresolved"
        self._bearer_source = "unresolved"
        self._traced_bearer_states = set()
        self._traced_token_states = set()
        self._traced_failure = False


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
