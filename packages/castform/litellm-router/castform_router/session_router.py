"""Route-once session affinity around a replaceable policy."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from castform_router.policy import RoutePolicy
from castform_router.types import RouteDecision, RouteRequest


@dataclass(frozen=True, slots=True)
class _PinnedDecision:
    expires_at: float
    decision: RouteDecision


class SessionRouter:
    """Pin the first decision for a session.

    The in-memory store is appropriate for the one-worker local spike. Replace
    it with Redis before running multiple LiteLLM workers or replicas.
    """

    def __init__(
        self,
        *,
        policy: RoutePolicy,
        ttl_seconds: int = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        self._policy = policy
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._pinned: dict[str, _PinnedDecision] = {}
        self._lock = asyncio.Lock()

    async def route(self, request: RouteRequest) -> RouteDecision:
        if request.session_id is None:
            return await self._policy.decide(request)

        # Holding the lock through the first policy call provides single-flight
        # behavior for concurrent first turns of the same local session.
        async with self._lock:
            now = self._clock()
            pinned = self._pinned.get(request.session_id)
            if pinned is not None and pinned.expires_at > now:
                return replace(pinned.decision, cache_hit=True)
            if pinned is not None:
                self._pinned.pop(request.session_id, None)

            decision = await self._policy.decide(request)
            self._pinned[request.session_id] = _PinnedDecision(
                expires_at=now + self._ttl_seconds,
                decision=decision,
            )
            return decision

    async def clear(self) -> None:
        """Clear local pins, primarily for tests and development."""

        async with self._lock:
            self._pinned.clear()
