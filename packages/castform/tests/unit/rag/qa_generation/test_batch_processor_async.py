"""Step-2 async path of batch_processor + model-scoped context semaphores.

Covers: ``call_openai_async`` taking the real-``await`` path for an async client,
``batch_process_async`` honoring an externally-supplied (model-scoped) semaphore as
a global bound, index-aligned ``None``-on-failure + ordering, and
``PipelineContext.model_semaphore`` lazy per-model/per-loop construction.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import castform.rag.qa_generation.batch_processor as bp
import pytest
from castform.rag.qa_generation.batch_processor import batch_process_async
from castform.rag.qa_generation.pipeline_config import PipelineContext


def _completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class _FakeAsyncClient:
    """Async-create client that tracks peak concurrency; ``fail_on`` prompts raise."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self._fail_on = fail_on or set()
        self.in_flight = 0
        self.max_in_flight = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            prompt = kwargs["messages"][-1]["content"]
            if prompt in self._fail_on:
                raise RuntimeError(f"boom:{prompt}")
            return _completion(f"ans:{prompt}")
        finally:
            self.in_flight -= 1


@pytest.fixture
def force_async(monkeypatch):
    """Treat the fake as an async client so the real-await branch is exercised."""
    monkeypatch.setattr(bp, "_is_async_client", lambda c: isinstance(c, _FakeAsyncClient))


async def test_async_path_awaits_real_client(force_async):
    client = _FakeAsyncClient()
    result = await batch_process_async(
        client=client, model="m", prompts=["a", "b"], show_progress=False
    )
    assert [r.answer for r in result.responses] == ["ans:a", "ans:b"]
    assert client.max_in_flight >= 1  # the async create actually ran


async def test_shared_semaphore_bounds_concurrency_globally(force_async):
    client = _FakeAsyncClient()
    sem = asyncio.Semaphore(3)
    # Two concurrent batches sharing one semaphore — peak in-flight must stay <= 3.
    await asyncio.gather(
        batch_process_async(
            client=client,
            model="m",
            prompts=[f"a{i}" for i in range(8)],
            show_progress=False,
            semaphore=sem,
        ),
        batch_process_async(
            client=client,
            model="m",
            prompts=[f"b{i}" for i in range(8)],
            show_progress=False,
            semaphore=sem,
        ),
    )
    assert client.max_in_flight <= 3, f"semaphore breached: {client.max_in_flight}"


async def test_unbounded_without_shared_semaphore_uses_max_concurrent(force_async):
    client = _FakeAsyncClient()
    await batch_process_async(
        client=client,
        model="m",
        prompts=[f"p{i}" for i in range(10)],
        show_progress=False,
        max_concurrent=4,
    )
    assert client.max_in_flight <= 4  # per-call Semaphore(max_concurrent)


async def test_failures_map_to_none_aligned_and_ordered(force_async):
    client = _FakeAsyncClient(fail_on={"b"})
    result = await batch_process_async(
        client=client, model="m", prompts=["a", "b", "c"], show_progress=False
    )
    assert len(result.responses) == 3
    assert result.responses[0].answer == "ans:a"
    assert result.responses[1] is None  # failed prompt -> None at same index
    assert result.responses[2].answer == "ans:c"


async def test_model_semaphore_keyed_per_model_and_endpoint(monkeypatch):
    monkeypatch.delenv("BENCHMAX_LLM_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("BENCHMAX_MODEL_MAX_CONCURRENCY", "gpt-5.4=7")
    ctx = PipelineContext(config=None, source=None)

    big = ctx.model_semaphore("gpt-5.4", base_url="u")
    assert big._value == 7  # per-model override
    assert ctx.model_semaphore("gpt-5.4", base_url="u") is big  # cached, same key

    mini = ctx.model_semaphore("gpt-5.4-mini", base_url="u")
    assert mini is not big  # different model -> independent cap
    assert mini._value == 40  # default

    # Same model name, different endpoint -> different deployment -> own semaphore.
    assert ctx.model_semaphore("gpt-5.4", base_url="other") is not big


async def test_same_model_endpoint_shared_across_callers(monkeypatch):
    monkeypatch.delenv("BENCHMAX_MODEL_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("BENCHMAX_LLM_MAX_CONCURRENCY", "12")
    ctx = PipelineContext(config=None, source=None)

    # e.g. generation and a judge that happen to use the same model+endpoint
    # share ONE backend cap (no double-counting).
    a = ctx.model_semaphore("m", base_url="u")
    b = ctx.model_semaphore("m", base_url="u")
    assert a is b
    assert a._value == 12  # global default applies to any model


async def test_search_semaphore_independent_of_llm(monkeypatch):
    monkeypatch.setenv("BENCHMAX_SEARCH_MAX_CONCURRENCY", "9")
    ctx = PipelineContext(config=None, source=None)
    s = ctx.search_semaphore()
    assert s._value == 9
    assert ctx.search_semaphore() is s
