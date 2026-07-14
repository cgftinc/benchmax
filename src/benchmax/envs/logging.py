"""Attach ordinary Python logs to the active rollout.

Environment authors can call ``logging.getLogger(__name__).info(...)`` anywhere
inside ``run_rollout`` or code it awaits. The executor binds the rollout ID, and
the trainer captures those records under that rollout's logs.

Example::

    import logging
    from benchmax.envs import BaseEnv

    logger = logging.getLogger(__name__)

    class DummyEnv(BaseEnv):
        ...

        async def run_tool(self, rollout_id, tool_name, **tool_args):
            logger.info("Running tool %s", tool_name)
            ...

        async def compute_reward(
            self, rollout_id, messages, task, *, termination_reason, **kwargs
        ):
            logger.info("Scoring rollout")
            ...
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import Iterable

__all__ = [
    "rollout_context",
    "group_context",
    "_CURRENT_ROLLOUT_ID",
    "_CURRENT_GROUP_RIDS",
]


_CURRENT_ROLLOUT_ID: ContextVar[str | None] = ContextVar(
    "benchmax_capture_rollout_id", default=None
)

_CURRENT_GROUP_RIDS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "benchmax_capture_group_rids", default=None
)


class _RolloutContext(AbstractContextManager[None]):
    __slots__ = ("_rid", "_tok")

    def __init__(self, rid: str) -> None:
        self._rid = str(rid)

    def __enter__(self) -> None:
        self._tok = _CURRENT_ROLLOUT_ID.set(self._rid)

    def __exit__(self, *exc: object) -> None:
        try:
            _CURRENT_ROLLOUT_ID.reset(self._tok)
        except (LookupError, ValueError):
            _CURRENT_ROLLOUT_ID.set(None)


class _GroupContext(AbstractContextManager[None]):
    __slots__ = ("_rids", "_tok")

    def __init__(self, rids: Iterable[str]) -> None:
        self._rids = tuple(str(r) for r in rids)

    def __enter__(self) -> None:
        self._tok = _CURRENT_GROUP_RIDS.set(self._rids)

    def __exit__(self, *exc: object) -> None:
        try:
            _CURRENT_GROUP_RIDS.reset(self._tok)
        except (LookupError, ValueError):
            _CURRENT_GROUP_RIDS.set(None)


def rollout_context(rollout_id: str) -> AbstractContextManager[None]:
    """Attribute logs in the context to one rollout."""
    return _RolloutContext(rollout_id)


def group_context(rollout_ids: Iterable[str]) -> AbstractContextManager[None]:
    """Attribute logs in the context to a rollout group."""
    return _GroupContext(rollout_ids)
