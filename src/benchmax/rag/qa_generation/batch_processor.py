"""Generic batch processing for LLM calls using OpenAI client.

This module provides a simple, generic way to process multiple prompts
in parallel using any OpenAI-compatible client. It handles rate limiting,
error handling, and provides token usage and latency tracking.

Example:
    ```python
    from openai import OpenAI
    from benchmax.rag.qa_generation import batch_process_sync

    client = OpenAI(api_key="...")
    prompts = ["Explain AI", "Explain ML", "Explain DL"]

    result = batch_process_sync(
        client=client,
        model="gpt-4",
        prompts=prompts,
        max_concurrent=5
    )

    for i, response in enumerate(result.responses):
        if response is None:
            continue
        print(f"Prompt {i}: {response.answer}")
    ```
"""

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from tqdm.auto import tqdm

# --- Process-global LLM concurrency cap -------------------------------------
#
# QA-gen fans batches across an outer ThreadPoolExecutor, and each batch/stage
# spins an ephemeral event loop with its OWN ``asyncio.Semaphore(max_concurrent)``
# — so that per-stage limit bounds nothing globally and real in-flight LLM
# concurrency is ``max_parallel_batches × max_concurrent`` (e.g. 40×8 = 320).
# That overshoots the serving knee (~10) by ~30× and tips the LLM path into
# congestion collapse.
#
# This single ``threading.Semaphore`` — acquired in the worker thread that runs
# the blocking OpenAI call, so it bounds across every ephemeral loop — caps total
# in-flight LLM calls regardless of batch/stage fan-out. The cap bounds in-flight
# at ALL ``max_parallel`` levels, so it's what protects large/high-parallel jobs
# from the 320–480 overshoot; the default is sized to the measured P=60 sweet spot
# (cap=40 → ~51 q/min on the live cluster) rather than the latency-bound low-P
# regime where the value barely moves throughput. The serving knee is highly
# time-variable (collapse ~30 one day, tolerant of 120 the next), so this static
# value is a stopgap — override with ``BENCHMAX_LLM_MAX_CONCURRENCY`` (<= 0
# disables). An adaptive/latency-aware throttle is the durable fix (Step 3).
_GLOBAL_LLM_CONCURRENCY_ENV = "BENCHMAX_LLM_MAX_CONCURRENCY"
_DEFAULT_GLOBAL_LLM_CONCURRENCY = 40

_global_sem_lock = threading.Lock()
_global_sem: threading.Semaphore | None = None
_global_sem_limit: int | None = None


def _resolve_global_concurrency_limit() -> int:
    """Cap from ``BENCHMAX_LLM_MAX_CONCURRENCY`` (unset/invalid → default)."""
    raw = os.environ.get(_GLOBAL_LLM_CONCURRENCY_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_GLOBAL_LLM_CONCURRENCY
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_GLOBAL_LLM_CONCURRENCY


def _get_global_llm_semaphore() -> threading.Semaphore | None:
    """Return the process-global LLM concurrency semaphore, or ``None`` when the
    cap is disabled (limit <= 0). Rebuilt if the configured limit changes."""
    global _global_sem, _global_sem_limit
    limit = _resolve_global_concurrency_limit()
    if limit <= 0:
        return None
    with _global_sem_lock:
        if _global_sem is None or _global_sem_limit != limit:
            _global_sem = threading.Semaphore(limit)
            _global_sem_limit = limit
        return _global_sem


def _chat_completion_with_token_fallback(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    temperature: float = 1.0,
):
    """Call chat completions with both token params, then fallback by error hint.

    Some providers only accept ``max_tokens`` while others only accept
    ``max_completion_tokens``. Start by sending both and retry once with the
    unsupported one removed if needed.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "timeout": timeout,
        "temperature": temperature,
    }
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "unsupported parameter" in msg and "max_tokens" in msg:
            kwargs.pop("max_tokens", None)
            return client.chat.completions.create(**kwargs)
        if "unsupported parameter" in msg and "max_completion_tokens" in msg:
            kwargs.pop("max_completion_tokens", None)
            return client.chat.completions.create(**kwargs)
        raise


try:  # openai is a hard dependency; guard only against import-order surprises.
    from openai import AsyncOpenAI as _AsyncOpenAI
except Exception:  # noqa: BLE001
    _AsyncOpenAI = None


def _is_async_client(client: Any) -> bool:
    """True for an ``AsyncOpenAI`` client (real ``await``), False for sync ``OpenAI``
    (offloaded to a thread). Lets ``call_openai_async`` serve both."""
    return _AsyncOpenAI is not None and isinstance(client, _AsyncOpenAI)


async def _achat_completion_with_token_fallback(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    temperature: float = 1.0,
):
    """Async twin of ``_chat_completion_with_token_fallback`` for an AsyncOpenAI
    client — same dual token-param send + fallback, with ``await``."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "timeout": timeout,
        "temperature": temperature,
    }
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "unsupported parameter" in msg and "max_tokens" in msg:
            kwargs.pop("max_tokens", None)
            return await client.chat.completions.create(**kwargs)
        if "unsupported parameter" in msg and "max_completion_tokens" in msg:
            kwargs.pop("max_completion_tokens", None)
            return await client.chat.completions.create(**kwargs)
        raise


@dataclass
class BatchResponse:
    """Response from a single LLM call.

    Attributes:
        answer: The text response from the model
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        latency_ms: Request latency in milliseconds
    """

    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class BatchResult:
    """Results from batch processing multiple prompts.

    Attributes:
        responses: List aligned to input prompts. Failed prompts are None.
        total_latency_ms: Total time taken for all requests
    """

    responses: list[BatchResponse | None]
    total_latency_ms: float

    @property
    def num_responses(self) -> int:
        """Number of successful responses."""
        return sum(1 for r in self.responses if r is not None)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all responses."""
        return sum(r.input_tokens for r in self.responses if r is not None)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all responses."""
        return sum(r.output_tokens for r in self.responses if r is not None)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all responses."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request in milliseconds."""
        successful_count = self.num_responses
        if successful_count == 0:
            return 0.0
        return sum(r.latency_ms for r in self.responses if r is not None) / successful_count


async def call_openai_async(
    client: Any,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 500,
    timeout: float = 60.0,
    temperature: float = 1.0,
) -> BatchResponse:
    """Call OpenAI API asynchronously for a single prompt.

    Args:
        client: OpenAI client instance
        model: Model name to use
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        temperature: Sampling temperature

    Returns:
        BatchResponse with the result
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if _is_async_client(client):
        # Real async I/O. Concurrency is bounded by the caller's asyncio
        # semaphore (the role-scoped one in batch_process_async), so no global
        # threading cap here. ``latency_ms`` times only the call.
        start_time = time.time()
        completion = await _achat_completion_with_token_fallback(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
        )
        latency_ms = (time.time() - start_time) * 1000
    else:
        # Sync client: offload the blocking call to a thread, bounded by the
        # process-global cap (acquired in the worker so the wait parks there and
        # ``latency_ms`` still times only the call). ``None`` when disabled.
        semaphore = _get_global_llm_semaphore()

        def _call_with_cap() -> tuple[Any, float]:
            if semaphore is not None:
                semaphore.acquire()
            try:
                start_time = time.time()
                completion = _chat_completion_with_token_fallback(
                    client,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                )
                return completion, (time.time() - start_time) * 1000
            finally:
                if semaphore is not None:
                    semaphore.release()

        loop = asyncio.get_event_loop()
        completion, latency_ms = await loop.run_in_executor(None, _call_with_cap)

    # Extract response
    answer = completion.choices[0].message.content
    usage = completion.usage

    return BatchResponse(
        answer=answer,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
    )


async def batch_process_async(
    client: Any,
    model: str,
    prompts: list[str],
    system_prompt: str | list[str] | None = None,
    max_tokens: int = 500,
    timeout: float = 60.0,
    max_concurrent: int = 10,
    show_progress: bool = True,
    temperature: float = 1.0,
    desc: str = "Processing prompts",
    semaphore: asyncio.Semaphore | None = None,
) -> BatchResult:
    """Process multiple prompts in parallel with rate limiting.

    ``semaphore``: pass a shared ``asyncio.Semaphore`` (e.g. a role-scoped one
    from ``PipelineContext``) to bound concurrency across *all* batches on the
    event loop. When ``None``, a per-call ``Semaphore(max_concurrent)`` is minted
    (the standalone / sync-wrapper behavior).

    Args:
        client: OpenAI client instance
        model: Model name to use
        prompts: List of user prompts
        system_prompt: Optional system prompt. Pass a single string to use the same
            prompt for all requests, or a list aligned to ``prompts`` for per-prompt
            system prompts.
        max_tokens: Maximum tokens per response
        timeout: Request timeout in seconds
        max_concurrent: Maximum concurrent requests
        show_progress: Whether to print progress updates
        temperature: Sampling temperature

    Returns:
        BatchResult with all responses

    Example:
        ```python
        from openai import OpenAI

        client = OpenAI(api_key="...")
        prompts = ["Explain AI", "Explain ML"]

        result = await batch_process_async(
            client=client,
            model="gpt-4",
            prompts=prompts,
            max_concurrent=5
        )

        print(f"Processed {result.num_responses} prompts")
        print(f"Total tokens: {result.total_tokens}")
        ```
    """
    sem = semaphore if semaphore is not None else asyncio.Semaphore(max_concurrent)
    start_time = time.time()

    # Create progress bar if enabled
    pbar = tqdm(total=len(prompts), desc=desc, disable=not show_progress)

    async def process_with_semaphore(idx: int, prompt: str) -> BatchResponse:
        async with sem:
            per_prompt_system = (
                system_prompt[idx] if isinstance(system_prompt, list) else system_prompt
            )
            try:
                return await call_openai_async(
                    client=client,
                    model=model,
                    prompt=prompt,
                    system_prompt=per_prompt_system,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                )
            finally:
                pbar.update(1)

    # Process all prompts concurrently
    tasks = [process_with_semaphore(i, prompt) for i, prompt in enumerate(prompts)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()

    # Preserve prompt alignment: failed prompts map to None at the same index
    aligned_responses: list[BatchResponse | None] = []
    failed_count = 0
    first_exc: Exception | None = None
    for response in responses:
        if isinstance(response, Exception):
            failed_count += 1
            if first_exc is None:
                first_exc = response
            aligned_responses.append(None)
        else:
            aligned_responses.append(response)

    if failed_count > 0:
        # Surfacing the first exception's class + message keeps a single bug
        # (e.g., wrong model name, auth header missing, base_url misrouted)
        # from looking like "the pipeline just rejected everything." Include
        # OpenAI SDK metadata (request URL + response body) when present so
        # 403/401 root causes don't get clipped to "Your request was blocked."
        detail = ""
        if first_exc is not None:
            detail = f": {type(first_exc).__name__}: {first_exc}"
            req = getattr(first_exc, "request", None)
            if req is not None:
                detail += f" | url={getattr(req, 'url', '?')}"
            body = getattr(first_exc, "body", None) or getattr(
                getattr(first_exc, "response", None), "text", None
            )
            if body:
                detail += f" | body={str(body)[:400]}"
        tqdm.write(f"Warning: {failed_count} prompt(s) failed to process{detail}")

    total_latency_ms = (time.time() - start_time) * 1000

    return BatchResult(
        responses=aligned_responses,
        total_latency_ms=total_latency_ms,
    )


def batch_process_sync(
    client: Any,
    model: str,
    prompts: list[str],
    system_prompt: str | list[str] | None = None,
    max_tokens: int = 500,
    timeout: float = 60.0,
    max_concurrent: int = 10,
    show_progress: bool = True,
    temperature: float = 1.0,
    desc: str = "Processing prompts",
) -> BatchResult:
    """Synchronous wrapper for batch_process_async.

    Handles both cases: when called from within an existing event loop
    (e.g., Jupyter) and when no loop is running.

    Args:
        Same as batch_process_async

    Returns:
        BatchResult with all responses

    Example:
        ```python
        from openai import OpenAI

        client = OpenAI(api_key="...")
        prompts = ["Explain AI", "Explain ML"]

        # Works in notebooks and regular Python scripts
        result = batch_process_sync(
            client=client,
            model="gpt-4",
            prompts=prompts,
            max_concurrent=5
        )
        ```
    """
    coro = batch_process_async(
        client=client,
        model=model,
        prompts=prompts,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
        temperature=temperature,
        desc=desc,
    )

    try:
        # Check if we're already in an event loop (e.g., Jupyter notebook)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run()
        return asyncio.run(coro)

    # We're in a running loop (Jupyter), use nest_asyncio
    import nest_asyncio

    nest_asyncio.apply()
    return loop.run_until_complete(coro)
