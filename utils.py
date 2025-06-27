from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

class AsyncLoopThread:
    """Run an asyncio event‑loop in a background thread and let callers submit
    coroutines from synchronous context.  We keep the loop alive for the whole
    lifespan of the manager so tool calls are cheap."""

    def __init__(self) -> None:
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._loop_ready.wait()

    # ---------------------------------------------------------------------
    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    # ---------------------------------------------------------------------
    @property
    def loop(self) -> asyncio.AbstractEventLoop:  # noqa: D401 – simple attr
        return self._loop

    def submit(self, coro: Coroutine[Any, Any, Any]):
        """Schedule *coro* on the background loop and return a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self):
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
