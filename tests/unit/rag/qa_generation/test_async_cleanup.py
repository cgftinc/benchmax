"""Cleanup contract for the async work queue (Finding #1 hardening).

Every per-loop async client built during a run (corpus ``AsyncClient`` +
``AsyncOpenAI`` generator/judge clients) must be closed when the run finishes —
including when the work queue raises — so a long-lived host does not leak transports.
"""

from __future__ import annotations

import types

import pytest

from benchmax.rag.qa_generation.generators.direct_llm import DirectLLMGenerator
from benchmax.rag.qa_generation.pipeline import Pipeline


class _AcloseSpy:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


async def test_cleanup_closes_all_components_on_success():
    comps = {k: _AcloseSpy() for k in ("gen", "tr", "guard", "f1", "f2", "src")}

    async def fake_arun(**_kw):
        return ("ok",)

    fake_self = types.SimpleNamespace(_arun_work_queue=fake_arun)
    result = await Pipeline._arun_work_queue_with_cleanup(
        fake_self,
        source=comps["src"],
        generator=comps["gen"],
        guard_filter=comps["guard"],
        filter_stage_names=[],
        filter_chain=[comps["f1"], comps["f2"]],
        transformer=comps["tr"],
        context=None,
    )
    assert result == ("ok",)
    assert all(c.closed == 1 for c in comps.values())


async def test_cleanup_runs_even_when_work_queue_raises():
    comps = {k: _AcloseSpy() for k in ("gen", "src")}

    async def boom(**_kw):
        raise RuntimeError("batch blew up")

    fake_self = types.SimpleNamespace(_arun_work_queue=boom)
    with pytest.raises(RuntimeError, match="batch blew up"):
        await Pipeline._arun_work_queue_with_cleanup(
            fake_self,
            source=comps["src"],
            generator=comps["gen"],
            guard_filter=object(),  # no aclose -> skipped, no error
            filter_stage_names=[],
            filter_chain=[],
            transformer=object(),  # no aclose -> skipped, no error
            context=None,
        )
    assert all(c.closed == 1 for c in comps.values())


async def test_generator_aclose_clears_cache_and_closes_clients():
    gen = DirectLLMGenerator(client=None, linker=None, cfg=None)
    c1, c2 = _FakeAsyncClient(), _FakeAsyncClient()
    gen._async_clients = {1: c1, 2: c2}

    await gen.aclose()

    assert gen._async_clients == {}
    assert c1.closed == 1 and c2.closed == 1
    # idempotent: a second call is a no-op, not an error
    await gen.aclose()
