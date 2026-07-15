"""Core functions for calling multiple models."""

import asyncio
import time
from datetime import datetime
from typing import Any

from .clients import ClientRegistry
from .models import BenchmarkResult, ModelResponse, ModelResult, OAIArgs
from .pricing import ALL_MODELS, get_pricing


class _Semaphore:
    """Simple semaphore for rate limiting concurrent requests."""

    def __init__(self, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


def _extract_anthropic_response(response: Any) -> tuple[str, str | None]:
    """Extract answer and thinking from Anthropic response structure."""
    thinking = None
    answer = None

    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text

    return answer or "", thinking


def _extract_openai_chat_response(response: Any) -> tuple[str, str | None, str | None]:
    """Extract answer, thinking, and reasoning summary from OpenAI Chat Completions response.

    Returns:
        Tuple of (answer, thinking, reasoning_summary)
    """
    message = response.choices[0].message
    answer = message.content or ""

    # Check for reasoning_content field (some models like Grok)
    thinking = getattr(message, "reasoning_content", None)

    # Check for refusal (content moderation)
    refusal = getattr(message, "refusal", None)
    if refusal and not answer:
        answer = f"[REFUSAL]: {refusal}"

    # Extract reasoning summary if present (OAI reasoning models)
    reasoning_summary = None
    reasoning = getattr(response, "reasoning", None)
    if reasoning:
        reasoning_summary = getattr(reasoning, "summary", None)

    return answer, thinking, reasoning_summary


def _extract_openai_responses_api(response: Any) -> tuple[str, str | None, str | None]:
    """Extract answer, thinking, and reasoning summary from OpenAI Responses API.

    Returns:
        Tuple of (answer, thinking, reasoning_summary)
    """
    # Responses API uses output_text for the main content
    answer = getattr(response, "output_text", "") or ""

    # No thinking field in Responses API (reasoning is separate)
    thinking = None

    # Extract reasoning summary if present
    reasoning_summary = None
    reasoning = getattr(response, "reasoning", None)
    if reasoning:
        reasoning_summary = getattr(reasoning, "summary", None)

    return answer, thinking, reasoning_summary


async def call_model(
    registry: ClientRegistry,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    oai_args: OAIArgs | None = None,
) -> ModelResponse:
    """Call a single model once.

    Args:
        registry: Client registry with API clients
        model: Model name to call
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        oai_args: Optional OpenAI-specific arguments (verbosity, reasoning).
            Only applies to openai-responses client type.
            - verbosity: "low", "medium", or "high" for output verbosity
            - reasoning: dict with "effort" and/or "summary" for reasoning models

    Returns:
        ModelResponse with answer, thinking, tokens, and latency
    """
    client, config = registry.get_client(model)

    start_time = time.time()

    # Run the sync API call in a thread pool to not block the event loop
    loop = asyncio.get_event_loop()

    if config.client_type == "anthropic":
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": config.deployment_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await loop.run_in_executor(None, lambda: client.messages.create(**kwargs))
        latency_ms = (time.time() - start_time) * 1000

        answer, thinking = _extract_anthropic_response(response)

        return ModelResponse(
            answer=answer,
            thinking=thinking,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw_response=response,
        )

    elif config.client_type == "openai-responses":
        # Responses API format
        input_content = prompt
        if system_prompt:
            input_content = f"{system_prompt}\n\n{prompt}"

        kwargs = {
            "model": config.deployment_name,
            "input": input_content,
            "max_output_tokens": max_tokens,
        }

        # Apply OAI-specific arguments if provided
        if oai_args:
            if verbosity := oai_args.get("verbosity"):
                kwargs["text"] = {"verbosity": verbosity}
            if reasoning := oai_args.get("reasoning"):
                # Filter out None values from reasoning config
                reasoning_config = {k: v for k, v in reasoning.items() if v is not None}
                if reasoning_config:
                    kwargs["reasoning"] = reasoning_config

        response = await loop.run_in_executor(None, lambda: client.responses.create(**kwargs))
        latency_ms = (time.time() - start_time) * 1000

        answer, thinking, reasoning_summary = _extract_openai_responses_api(response)

        return ModelResponse(
            answer=answer,
            thinking=thinking,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw_response=response,
            reasoning_summary=reasoning_summary,
        )

    else:  # openai-chat
        # Chat Completions API format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": config.deployment_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_tokens,
            "timeout": timeout,
        }

        def _create_chat_completion_with_fallback():
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                retry_kwargs = dict(kwargs)
                if "unsupported parameter" in msg and "max_tokens" in msg:
                    retry_kwargs.pop("max_tokens", None)
                    return client.chat.completions.create(**retry_kwargs)
                if "unsupported parameter" in msg and "max_completion_tokens" in msg:
                    retry_kwargs.pop("max_completion_tokens", None)
                    return client.chat.completions.create(**retry_kwargs)
                raise

        response = await loop.run_in_executor(None, _create_chat_completion_with_fallback)
        latency_ms = (time.time() - start_time) * 1000

        answer, thinking, reasoning_summary = _extract_openai_chat_response(response)

        return ModelResponse(
            answer=answer,
            thinking=thinking,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_response=response,
            reasoning_summary=reasoning_summary,
        )


async def _call_model_n_times(
    registry: ClientRegistry,
    model: str,
    prompt: str,
    system_prompt: str | None,
    n: int,
    max_tokens: int,
    timeout: float,
    oai_args: OAIArgs | None = None,
) -> ModelResult:
    """Call a single model N times and aggregate results."""
    tasks = [
        call_model(registry, model, prompt, system_prompt, max_tokens, timeout, oai_args)
        for _ in range(n)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and keep successful responses
    successful_responses: list[ModelResponse] = []
    for r in responses:
        if isinstance(r, BaseException):
            print(f"Warning: {model} call failed: {r}")
        else:
            successful_responses.append(r)

    input_price, output_price = get_pricing(model)

    return ModelResult(
        model=model,
        responses=successful_responses,
        input_price_per_1k=input_price,
        output_price_per_1k=output_price,
    )


async def call_multi_model(
    registry: ClientRegistry,
    prompt: str,
    models: list[str] | None = None,
    system_prompt: str | None = None,
    n: int = 1,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    oai_args: OAIArgs | None = None,
) -> BenchmarkResult:
    """Call multiple models concurrently and collect results.

    Args:
        registry: Client registry with API clients
        prompt: User prompt to send to all models
        models: List of model names to call. If None, uses ALL_MODELS.
        system_prompt: Optional system prompt
        n: Number of responses to generate per model
        max_tokens: Maximum tokens in each response
        timeout: Request timeout in seconds per call
        oai_args: Optional OpenAI-specific arguments (verbosity, reasoning).
            Only applies to openai-responses client type.

    Returns:
        BenchmarkResult with all model results
    """
    if models is None:
        models = ALL_MODELS

    # Filter to only models available in registry
    available = set(registry.available_models)
    models_to_call = [m for m in models if m in available]

    if not models_to_call:
        raise ValueError(
            f"No valid models to call. Requested: {models}, Available: {list(available)}"
        )

    # Call all models concurrently
    tasks = [
        _call_model_n_times(
            registry, model, prompt, system_prompt, n, max_tokens, timeout, oai_args
        )
        for model in models_to_call
    ]
    results = await asyncio.gather(*tasks)

    # Build result dict
    results_dict = {r.model: r for r in results}

    return BenchmarkResult(
        prompt=prompt,
        system_prompt=system_prompt,
        results=results_dict,
        timestamp=datetime.now(),
    )


def call_multi_model_sync(
    registry: ClientRegistry,
    prompt: str,
    models: list[str] | None = None,
    system_prompt: str | None = None,
    n: int = 1,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    oai_args: OAIArgs | None = None,
) -> BenchmarkResult:
    """Synchronous wrapper for call_multi_model.

    Convenience function for use in notebooks and scripts that don't
    use asyncio directly. Handles both cases: when called from within
    an existing event loop (e.g., Jupyter) and when no loop is running.

    Args:
        Same as call_multi_model

    Returns:
        BenchmarkResult with all model results
    """
    coro = call_multi_model(
        registry=registry,
        prompt=prompt,
        models=models,
        system_prompt=system_prompt,
        n=n,
        max_tokens=max_tokens,
        timeout=timeout,
        oai_args=oai_args,
    )

    try:
        # Check if we're already in an event loop (e.g., Jupyter notebook)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run()
        return asyncio.run(coro)

    # We're in a running loop (Jupyter), use nest_asyncio or run directly
    import nest_asyncio

    nest_asyncio.apply()
    return loop.run_until_complete(coro)


async def call_model_batch(
    registry: ClientRegistry,
    model: str,
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    max_concurrent: int = 10,
    oai_args: OAIArgs | None = None,
) -> ModelResult:
    """Process multiple prompts with a single model in parallel with rate limiting.

    This function is optimized for batch processing tasks like entity recognition
    where you need to process many prompts with the same model. It includes:
    - Parallel processing with configurable concurrency limits
    - Automatic cost and latency tracking
    - Error handling that continues processing even if some requests fail

    Args:
        registry: Client registry with API clients
        model: Model name to use for all prompts
        prompts: List of user prompts to process
        system_prompt: Optional system prompt (same for all requests)
        max_tokens: Maximum tokens per response
        timeout: Request timeout in seconds per call
        max_concurrent: Maximum number of concurrent requests (default: 10)
        oai_args: Optional OpenAI-specific arguments (verbosity, reasoning)

    Returns:
        ModelResult with all successful responses, aggregated cost and latency stats

    Example:
        ```python
        registry = ClientRegistry(api_key="...", endpoints={...})
        prompts = ["Extract entities from: ...", "Extract entities from: ...", ...]

        result = await call_model_batch(
            registry=registry,
            model="gpt-5-nano",
            prompts=prompts,
            max_concurrent=20  # Process 20 at a time
        )

        print(f"Processed {result.num_responses} prompts")
        print(f"Total cost: ${result.total_cost:.4f}")
        print(f"Avg latency: {result.avg_latency_ms:.0f}ms")
        ```
    """
    if not prompts:
        raise ValueError("prompts list cannot be empty")

    semaphore = _Semaphore(max_concurrent)

    async def _call_with_limit(prompt: str) -> ModelResponse | Exception:
        """Call model with semaphore rate limiting."""
        async with semaphore:
            try:
                return await call_model(
                    registry, model, prompt, system_prompt, max_tokens, timeout, oai_args
                )
            except Exception as e:
                return e

    # Process all prompts in parallel with rate limiting
    tasks = [_call_with_limit(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    # Filter out exceptions and keep successful responses
    successful_responses = []
    failed_count = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"Warning: Request {i + 1}/{len(prompts)} failed: {r}")
            failed_count += 1
        else:
            successful_responses.append(r)

    if failed_count > 0:
        print(
            f"Batch processing completed: {len(successful_responses)}/{len(prompts)} successful, "
            f"{failed_count} failed"
        )

    input_price, output_price = get_pricing(model)

    return ModelResult(
        model=model,
        responses=successful_responses,
        input_price_per_1k=input_price,
        output_price_per_1k=output_price,
    )


def call_model_batch_sync(
    registry: ClientRegistry,
    model: str,
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    max_concurrent: int = 10,
    oai_args: OAIArgs | None = None,
) -> ModelResult:
    """Synchronous wrapper for call_model_batch.

    Convenience function for use in notebooks and scripts that don't
    use asyncio directly. See call_model_batch for full documentation.

    Args:
        Same as call_model_batch

    Returns:
        ModelResult with all successful responses
    """
    coro = call_model_batch(
        registry=registry,
        model=model,
        prompts=prompts,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrent=max_concurrent,
        oai_args=oai_args,
    )

    try:
        # Check if we're already in an event loop (e.g., Jupyter notebook)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run()
        return asyncio.run(coro)

    # We're in a running loop (Jupyter), use nest_asyncio or run directly
    import nest_asyncio

    nest_asyncio.apply()
    return loop.run_until_complete(coro)
