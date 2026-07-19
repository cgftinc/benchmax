"""Corpus API client for uploading chunks and searching."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from tqdm.auto import tqdm

from castform.platform.credentials import TokenProvider, platform_bearer

from .exceptions import (
    AuthenticationError,
    CorpusAPIError,
    CorpusLimitError,
    CorpusNotFoundError,
)
from .models import Corpus, CorpusChunk, SearchResult, UploadResult

if TYPE_CHECKING:
    from castform.rag.chunkers.models import Chunk, ChunkCollection

logger = logging.getLogger(__name__)


@dataclass
class CorpusClient:
    """Client for interacting with the Corpora API.

    Example:
        >>> client = CorpusClient(base_url="http://localhost:3000")
        >>> corpus = client.create_corpus("my-docs")
        >>> result = client.upload_chunks(corpus.id, collection)
        >>> print(f"Uploaded {result.inserted_count} chunks")

    The bearer token is resolved **per request** via ``token_provider`` (default:
    the platform credential resolver), so a rotating act-as token stays valid
    across a long run and is never frozen into a pickled env. Callers with an
    explicit key inject one, e.g. ``token_provider=lambda: my_key``.
    """

    base_url: str = "http://localhost:3000"
    timeout: float = 30.0
    # Shared budget for network-error retries AND 429 rate-limit backoff. A
    # large corpus can span several rate-limit windows, so a late batch may be
    # throttled more than once — give it enough attempts to ride them out.
    max_retries: int = 5
    retry_backoff_seconds: float = 0.5
    token_provider: TokenProvider = platform_bearer
    # Enable HTTP/2 multiplexing on the async client. Safe there (one client
    # bound to one event loop), unlike the shared sync client across threads.
    async_http2: bool = True
    # HTTP clients are created lazily, one per thread (see ``_http_client``).
    # httpx.Client's connection pool is not safe to share across threads at high
    # parallelism: the QA-gen work queue hits this client from every batch
    # thread, which raced the shared pool's sockets into
    # ``ReadError: [Errno 9] Bad file descriptor``. A client-per-thread avoids
    # the shared-pool race entirely.
    _local: threading.local = field(
        init=False, repr=False, default_factory=threading.local
    )
    _client_override: httpx.Client | None = field(
        init=False, repr=False, default=None
    )
    _client_registry: list[httpx.Client] = field(
        init=False, repr=False, default_factory=list
    )
    _registry_lock: threading.Lock = field(
        init=False, repr=False, default_factory=threading.Lock
    )
    # Single async client, lazily bound to the running event loop (rebuilt if the
    # loop changes — asyncio.run() mints a fresh loop per Pipeline.run()).
    _async_client: httpx.AsyncClient | None = field(
        init=False, repr=False, default=None
    )
    _async_client_loop: Any = field(init=False, repr=False, default=None)

    def _build_http_client(self) -> httpx.Client:
        """Create a new HTTP client and register it for ``close()``.

        Auth is attached per request in ``_request`` — not baked here."""
        timeout_config = httpx.Timeout(
            timeout=self.timeout,
            connect=self.timeout,
            read=self.timeout,
            write=self.timeout,
            pool=self.timeout,
        )
        client = httpx.Client(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=timeout_config,
        )
        with self._registry_lock:
            self._client_registry.append(client)
        return client

    @property
    def _http_client(self) -> httpx.Client:
        """The HTTP client for the calling thread.

        Returns an explicitly installed override if present (e.g. a profiling
        harness swapping in an HTTP/2 client); otherwise a lazily-created
        per-thread client so concurrent batch threads never share a pool."""
        if self._client_override is not None:
            return self._client_override
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._build_http_client()
            self._local.client = client
        return client

    @_http_client.setter
    def _http_client(self, value: httpx.Client) -> None:
        """Install a single client shared across all threads. Intended for
        single-threaded or multiplexed (HTTP/2) setups, not pool-per-thread."""
        self._client_override = value
        with self._registry_lock:
            self._client_registry.append(value)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Execute an HTTP request with retry/backoff for transient network failures.

        Resolves the bearer per request via ``token_provider`` so a rotating
        act-as token stays valid across a long run."""
        try:
            bearer = self.token_provider()
        except RuntimeError as exc:
            # The seam (platform_bearer) raises when no credential resolves; surface
            # it as an auth error so callers catch it like any other Corpora failure.
            raise AuthenticationError(
                f"No Castform platform credential available for the Corpora API: {exc}"
            ) from exc
        headers = {
            **kwargs.pop("headers", {}),
            "Authorization": f"Bearer {bearer}",
        }
        retries = max(1, int(self.max_retries))
        attempt = 1
        while True:
            try:
                response = self._http_client.request(
                    method, path, headers=headers, **kwargs
                )
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as exc:
                if attempt >= retries:
                    raise CorpusAPIError(
                        (
                            "Corpora API request failed after retries due to a network timeout/error. "
                            f"method={method} path={path} base_url={self.base_url} "
                            f"attempts={retries} last_error={exc!s}"
                        ),
                        status_code=503,
                    ) from exc
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Corpora API request attempt %s/%s failed (%s). Retrying in %.2fs. "
                    "method=%s path=%s base_url=%s",
                    attempt,
                    retries,
                    type(exc).__name__,
                    delay,
                    method,
                    path,
                    self.base_url,
                )
                time.sleep(delay)
                attempt += 1
                continue

            # Server-side rate limiting (429): the chunk-upload endpoint caps
            # requests per user, and a bulk upload runs many batches in
            # parallel — so 429s are expected, not fatal. Honor the server's
            # Retry-After and retry within the budget instead of raising and
            # failing the whole upload partway through.
            if response.status_code == 429 and attempt < retries:
                delay = self._retry_after_delay(response, attempt)
                logger.warning(
                    "Corpora API rate-limited (429) on attempt %s/%s. Retrying in %.2fs. "
                    "method=%s path=%s base_url=%s",
                    attempt,
                    retries,
                    delay,
                    method,
                    path,
                    self.base_url,
                )
                time.sleep(delay)
                attempt += 1
                continue

            return response

    def _retry_after_delay(self, response: httpx.Response, attempt: int) -> float:
        """Seconds to wait before retrying a 429.

        Prefer the server's ``Retry-After`` header (delta-seconds); fall back
        to exponential backoff when it's absent or unparseable."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.retry_backoff_seconds * (2 ** (attempt - 1))

    def _get_async_client(self) -> httpx.AsyncClient:
        """The async client bound to the running event loop.

        Rebuilt when the loop changes (``asyncio.run`` mints a fresh loop per
        ``Pipeline.run``) or the client was closed. Creation does not ``await``,
        so on a single event loop the check-then-build is race-free."""
        loop = asyncio.get_running_loop()
        client = self._async_client
        if client is None or client.is_closed or self._async_client_loop is not loop:
            timeout_config = httpx.Timeout(
                timeout=self.timeout,
                connect=self.timeout,
                read=self.timeout,
                write=self.timeout,
                pool=self.timeout,
            )
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Content-Type": "application/json"},
                timeout=timeout_config,
                http2=self.async_http2,
            )
            self._async_client = client
            self._async_client_loop = loop
        return client

    async def _arequest(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Async twin of ``_request`` — same retry/backoff and 429 handling, with
        ``await asyncio.sleep`` instead of ``time.sleep`` so the loop stays free."""
        try:
            bearer = self.token_provider()
        except RuntimeError as exc:
            # The seam (platform_bearer) raises when no credential resolves; surface
            # it as an auth error so callers catch it like any other Corpora failure.
            raise AuthenticationError(
                f"No Castform platform credential available for the Corpora API: {exc}"
            ) from exc
        headers = {
            **kwargs.pop("headers", {}),
            "Authorization": f"Bearer {bearer}",
        }
        retries = max(1, int(self.max_retries))
        client = self._get_async_client()
        attempt = 1
        while True:
            try:
                response = await client.request(method, path, headers=headers, **kwargs)
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as exc:
                if attempt >= retries:
                    raise CorpusAPIError(
                        (
                            "Corpora API request failed after retries due to a network timeout/error. "
                            f"method={method} path={path} base_url={self.base_url} "
                            f"attempts={retries} last_error={exc!s}"
                        ),
                        status_code=503,
                    ) from exc
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Corpora API request attempt %s/%s failed (%s). Retrying in %.2fs. "
                    "method=%s path=%s base_url=%s",
                    attempt,
                    retries,
                    type(exc).__name__,
                    delay,
                    method,
                    path,
                    self.base_url,
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue

            if response.status_code == 429 and attempt < retries:
                delay = self._retry_after_delay(response, attempt)
                logger.warning(
                    "Corpora API rate-limited (429) on attempt %s/%s. Retrying in %.2fs. "
                    "method=%s path=%s base_url=%s",
                    attempt,
                    retries,
                    delay,
                    method,
                    path,
                    self.base_url,
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue

            return response

    def __enter__(self) -> "CorpusClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close every HTTP client this instance created (one per thread, plus
        any installed override)."""
        with self._registry_lock:
            clients = list(self._client_registry)
            self._client_registry.clear()
        self._client_override = None
        for client in clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Error closing corpus HTTP client", exc_info=True)

    async def aclose(self) -> None:
        """Close the async client. Call from within its event loop."""
        client = self._async_client
        self._async_client = None
        self._async_client_loop = None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Error closing corpus async client", exc_info=True)

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Convert HTTP errors to appropriate exceptions."""
        if response.status_code in (200, 201):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message)

        if response.status_code == 400:
            if "Maximum of 20 corpora" in message:
                raise CorpusLimitError()
            if "Chunk limit exceeded" in message:
                raise CorpusAPIError(message, 400)
            raise CorpusAPIError(message, 400)

        if response.status_code == 404:
            raise CorpusNotFoundError(message)

        raise CorpusAPIError(message, response.status_code)

    # === Corpus Management ===

    def create_corpus(self, name: str) -> Corpus:
        """Create a new corpus.

        Args:
            name: Name for the corpus

        Returns:
            Corpus object with id, name, timestamps

        Raises:
            CorpusLimitError: If max 20 corpora limit reached
            AuthenticationError: If API key is invalid
        """
        response = self._request("POST", "/v1/corpora", json={"name": name})
        self._handle_response_errors(response)
        return Corpus.from_api_response(response.json())

    def list_corpora(self) -> list[Corpus]:
        """List all corpora for the authenticated user.

        Returns:
            List of Corpus objects
        """
        response = self._request("GET", "/v1/corpora")
        self._handle_response_errors(response)
        return [Corpus.from_api_response(c) for c in response.json()]

    def get_corpus(self, corpus_id: str) -> Corpus:
        """Get a corpus by ID from the authenticated user's corpus list.

        Args:
            corpus_id: Corpus ID to resolve

        Returns:
            The matching Corpus object

        Raises:
            CorpusNotFoundError: If no corpus matches the given ID
        """
        for corpus in self.list_corpora():
            if corpus.id == corpus_id:
                return corpus
        raise CorpusNotFoundError(corpus_id)

    def delete_corpus(self, corpus_id: str) -> bool:
        """Delete a corpus and all its chunks.

        Args:
            corpus_id: ID of the corpus to delete

        Returns:
            True if deletion was successful
        """
        response = self._request("DELETE", f"/v1/corpora/{corpus_id}")
        self._handle_response_errors(response)
        return response.json().get("success", False)

    def get_or_create_corpus(self, name: str, on_limit: str = "prompt") -> Corpus:
        """Get existing corpus by name or create new one.

        Args:
            name: Corpus name
            on_limit: Behavior when limit reached:
                - "prompt": Interactive prompt to delete (notebook)
                - "error": Raise CorpusLimitError
                - "oldest": Auto-delete oldest corpus

        Returns:
            Corpus object
        """
        # First, try to find existing corpus by name
        existing = self.list_corpora()
        for corpus in existing:
            if corpus.name == name:
                return corpus

        # Try to create new corpus
        try:
            return self.create_corpus(name)
        except CorpusLimitError as e:
            e.existing_corpora = existing

            if on_limit == "error":
                raise

            if on_limit == "oldest":
                oldest = min(existing, key=lambda c: c.created_at)
                print(
                    f"Deleting oldest corpus: {oldest.name} (created {oldest.created_at})"
                )
                self.delete_corpus(oldest.id)
                return self.create_corpus(name)

            if on_limit == "prompt":
                return self._interactive_corpus_selection(name, existing)

            raise ValueError(f"Unknown on_limit strategy: {on_limit}")

    def _interactive_corpus_selection(
        self, new_name: str, existing: list[Corpus]
    ) -> Corpus:
        """Interactive corpus selection for Jupyter notebooks."""
        print("\n" + "=" * 60)
        print("CORPUS LIMIT REACHED (max 5)")
        print("=" * 60)
        print("\nExisting corpora:")

        for i, corpus in enumerate(existing, 1):
            print(f"  {i}. {corpus.name}")
            print(f"     ID: {corpus.id}")
            print(f"     Created: {corpus.created_at}")

        print("\n  0. Cancel operation")
        print()

        while True:
            try:
                choice = input(
                    f"Enter number to delete (1-{len(existing)}) or 0 to cancel: "
                )
                choice_int = int(choice)

                if choice_int == 0:
                    raise CorpusLimitError(existing)

                if 1 <= choice_int <= len(existing):
                    to_delete = existing[choice_int - 1]
                    confirm = input(f"Delete '{to_delete.name}'? (y/N): ")

                    if confirm.lower() == "y":
                        self.delete_corpus(to_delete.id)
                        print(f"Deleted corpus: {to_delete.name}")
                        return self.create_corpus(new_name)
                    print("Cancelled.")
                    continue

                print(f"Please enter a number between 0 and {len(existing)}")

            except ValueError:
                print("Please enter a valid number")

    # === Chunk Upload ===

    def upload_chunks(
        self,
        corpus_id: str,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_progress: bool = True,
        max_workers: int = 8,
    ) -> UploadResult:
        """Upload a ChunkCollection to a corpus.

        Args:
            corpus_id: Target corpus ID
            collection: ChunkCollection to upload
            batch_size: Number of chunks per batch (default 100)
            show_progress: Show tqdm progress bar
            max_workers: Number of parallel upload workers (default 8)

        Returns:
            UploadResult with counts and ID list

        Raises:
            ChunkLimitError: If would exceed 10,000 chunk limit
        """
        chunks_list = list(collection)
        total_chunks = len(chunks_list)
        if total_chunks == 0:
            return UploadResult(success=True, inserted_count=0, chunk_ids=[])
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_workers <= 0:
            raise ValueError("max_workers must be > 0")

        total_batches = (total_chunks + batch_size - 1) // batch_size
        all_chunk_ids: list[str] = []
        total_inserted = 0

        def _upload_batch(batch_index: int) -> tuple[int, list[str]]:
            start = batch_index * batch_size
            end = min(start + batch_size, total_chunks)
            batch = chunks_list[start:end]
            # Prepare batch payload - include hash in metadata for later matching
            payload = {
                "chunks": [
                    {
                        "id": chunk.hash,
                        "content": chunk.content,
                        "metadata": {**chunk.metadata_dict, "_local_hash": chunk.hash},
                    }
                    for chunk in batch
                ]
            }

            response = self._request(
                "POST", f"/v1/corpora/{corpus_id}/chunks", json=payload
            )
            self._handle_response_errors(response)

            data = response.json()
            inserted_count = data.get("insertedCount", 0)
            chunk_ids = data.get("chunkIds", [])
            return inserted_count, chunk_ids

        pbar = tqdm(
            total=total_batches,
            desc="Uploading chunks",
            disable=not show_progress,
            unit="batch",
        )

        if max_workers == 1:
            for batch_index in range(total_batches):
                inserted_count, chunk_ids = _upload_batch(batch_index)
                total_inserted += inserted_count
                all_chunk_ids.extend(chunk_ids)
                pbar.update(1)
                pbar.set_postfix({"inserted": total_inserted})
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_upload_batch, batch_index)
                    for batch_index in range(total_batches)
                ]
                for future in as_completed(futures):
                    inserted_count, chunk_ids = future.result()
                    total_inserted += inserted_count
                    all_chunk_ids.extend(chunk_ids)
                    pbar.update(1)
                    pbar.set_postfix({"inserted": total_inserted})

        pbar.close()

        return UploadResult(
            success=True,
            inserted_count=total_inserted,
            chunk_ids=all_chunk_ids,
        )

    # === Search ===

    def list_corpus_chunks(
        self,
        corpus_id: str,
        limit: int = 500,
        cursor: str | None = None,
    ) -> tuple[list[CorpusChunk], str | None]:
        """List chunks from a corpus using cursor-based pagination.

        Args:
            corpus_id: Corpus to read chunks from
            limit: Maximum results to return per page
            cursor: Last chunk ID from previous page (None for first page)

        Returns:
            Tuple of (chunks, next_cursor). next_cursor is None when
            there are no more pages.
        """
        params: dict[str, str | int] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        response = self._request(
            "GET",
            f"/v1/corpora/{corpus_id}/chunks",
            params=params,
        )
        self._handle_response_errors(response)

        data = response.json()
        rows = data.get("results") or data.get("chunks") or []
        results = [
            CorpusChunk(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata") or {},
                score=r.get("score"),
            )
            for r in rows
        ]

        next_cursor = data.get("nextCursor")
        return results, next_cursor

    def search(
        self,
        corpus_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """BM25 search within a corpus.

        Args:
            corpus_id: Corpus to search
            query: Search query string
            limit: Maximum results to return
            offset: Pagination offset
            metadata: Optional metadata filter
            filters: Optional structured filter DSL.
                Example:
                {
                    "and": [
                        {"field": "date_start", "op": "gte", "value": "2017-01-01"},
                        {"field": "participants", "op": "contains_any", "value": ["angel"]},
                    ]
                }

        Returns:
            SearchResult with results and total count
        """
        payload: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if metadata:
            payload["metadata"] = metadata
        if filters:
            payload["filters"] = filters

        response = self._request(
            "POST", f"/v1/corpora/{corpus_id}/search", json=payload
        )
        self._handle_response_errors(response)

        data = response.json()
        results = [
            CorpusChunk(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata") or {},
                score=r.get("score"),
            )
            for r in data.get("results", [])
        ]

        return SearchResult(results=results, total=data.get("total", 0), query=query)

    def search_with_chunks(
        self,
        corpus_id: str,
        query: str,
        collection: ChunkCollection,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search and return matching local Chunk objects with scores.

        Requires chunks to have been uploaded (uses hash for matching).

        Args:
            corpus_id: Corpus to search
            query: Search query
            collection: Local ChunkCollection
            limit: Max results
            metadata: Optional exact-match metadata filter
            filters: Optional structured filter DSL

        Returns:
            List of (Chunk, score) tuples
        """
        result = self.search(
            corpus_id=corpus_id,
            query=query,
            limit=limit,
            metadata=metadata,
            filters=filters,
        )

        matched: list[tuple[Chunk, float]] = []
        for corpus_chunk in result.results:
            # The chunk ID is the hash, so we can look up directly
            local_chunk = collection.get_chunk_by_hash(corpus_chunk.id)
            if local_chunk:
                matched.append((local_chunk, corpus_chunk.score or 0.0))

        return matched

    async def asearch(
        self,
        corpus_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Async twin of ``search``. Same payload + response shape, async I/O."""
        payload: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if metadata:
            payload["metadata"] = metadata
        if filters:
            payload["filters"] = filters

        response = await self._arequest(
            "POST", f"/v1/corpora/{corpus_id}/search", json=payload
        )
        self._handle_response_errors(response)

        data = response.json()
        results = [
            CorpusChunk(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata") or {},
                score=r.get("score"),
            )
            for r in data.get("results", [])
        ]

        return SearchResult(results=results, total=data.get("total", 0), query=query)

    async def asearch_with_chunks(
        self,
        corpus_id: str,
        query: str,
        collection: ChunkCollection,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Async twin of ``search_with_chunks``."""
        result = await self.asearch(
            corpus_id=corpus_id,
            query=query,
            limit=limit,
            metadata=metadata,
            filters=filters,
        )

        matched: list[tuple[Chunk, float]] = []
        for corpus_chunk in result.results:
            local_chunk = collection.get_chunk_by_hash(corpus_chunk.id)
            if local_chunk:
                matched.append((local_chunk, corpus_chunk.score or 0.0))

        return matched
