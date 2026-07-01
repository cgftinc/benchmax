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

    def is_closed(self) -> bool:
        return self.closed > 0

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


async def test_generator_aclose_closes_loop_bound_client():
    gen = DirectLLMGenerator(client=None, linker=None, cfg=None)
    fake = _FakeAsyncClient()
    gen._async_client = fake
    gen._async_client_loop = object()

    await gen.aclose()

    assert gen._async_client is None
    assert gen._async_client_loop is None
    assert fake.closed == 1
    # idempotent: a second call is a no-op, not an error
    await gen.aclose()
    assert fake.closed == 1


async def test_get_async_client_rebuilds_after_close():
    """#3: the getter compares the loop object + is_closed(), so a closed client is
    rebuilt rather than handed back stale."""
    gen = DirectLLMGenerator(
        client=None,
        linker=None,
        cfg=types.SimpleNamespace(api_key="x", base_url="http://localhost"),
    )
    c1 = gen._get_async_client()
    assert gen._get_async_client() is c1  # cached within the same loop
    await c1.close()
    c2 = gen._get_async_client()  # rebuilt because the cached client was closed
    assert c2 is not c1
    await c2.close()
