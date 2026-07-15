"""Per-rollout logging context binding for benchmax env code.

This module is the small public surface env authors use to scope
``logging`` records to a particular rollout. It deliberately holds only
the :class:`~contextvars.ContextVar` instances and the context managers
that bind them — the heavy capture machinery (handler, buffer, drain
helpers, counters) lives in the trainer process at
``trainer.data.async_workers.capture``. The trainer reads from this
module's ContextVars; env code writes to them.

Env authors do not normally import from here. The trainer's env_service
binds ``rollout_context(rid)`` around every per-rollout method
(:meth:`benchmax.envs.base_env.BaseEnv.run_tool`,
:meth:`~benchmax.envs.base_env.BaseEnv.compute_reward`, etc.) and
``group_context(rollout_ids)`` around
:meth:`~benchmax.envs.base_env.BaseEnv.compute_group_reward`, so calls
like ``logging.getLogger(__name__).info(...)`` in the body Just Work.

The one place an env author may want to import from here is inside a
``compute_group_reward`` override: by default, the surrounding
``group_context`` makes logs fan out to every rid in the group (good for
group-level summaries). To narrow a per-rid loop iteration's logs to
just that rid, wrap the iteration body in ``rollout_context(rid)`` —
the per-rid binding takes precedence over the group fan-out.
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

# Bound by env_service around ``compute_group_reward`` calls. Tuple instead
# of list so it's hashable + immutable (safe to share across the fan-out
# loop in the handler without defensive copy).
_CURRENT_GROUP_RIDS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "benchmax_capture_group_rids", default=None
)


class _RolloutContext(AbstractContextManager[None]):
    __slots__ = ("_rid", "_tok")

    def __init__(self, rid: str) -> None:
        # Coerce to str — callers sometimes pass uuid.UUID instances; the
        # trainer's buffer is dict-keyed on str so mixing types would
        # silently miss.
        self._rid = str(rid)

    def __enter__(self) -> None:
        self._tok = _CURRENT_ROLLOUT_ID.set(self._rid)

    def __exit__(self, *exc: object) -> None:
        # Swallow LookupError/ValueError from reset() so we never mask the
        # original exception from the with-body. Token-from-wrong-context
        # only happens in pathological cases (CM reused, task escaped the
        # block) — best-effort clear is the right fallback.
        try:
            _CURRENT_ROLLOUT_ID.reset(self._tok)
        except (LookupError, ValueError):
            _CURRENT_ROLLOUT_ID.set(None)


class _GroupContext(AbstractContextManager[None]):
    __slots__ = ("_rids", "_tok")

    def __init__(self, rids: Iterable[str]) -> None:
        # Coerce each rid to str (uuid.UUID, etc.) to match the buffer keying.
        # tuple() materializes the iterable once so the ContextVar holds a
        # stable, immutable snapshot — callers can't mutate it mid-handler.
        self._rids = tuple(str(r) for r in rids)

    def __enter__(self) -> None:
        self._tok = _CURRENT_GROUP_RIDS.set(self._rids)

    def __exit__(self, *exc: object) -> None:
        try:
            _CURRENT_GROUP_RIDS.reset(self._tok)
        except (LookupError, ValueError):
            _CURRENT_GROUP_RIDS.set(None)


def rollout_context(rollout_id: str) -> AbstractContextManager[None]:
    """Bind ``rollout_id`` to the current asyncio task for the duration of
    the ``with`` block.

    Logs emitted within the block are attributed to ``rollout_id`` by the
    trainer's capture handler. Nested binds for the same task replace the
    outer binding for their scope and restore it on exit (standard
    ContextVar token semantics).

    Takes precedence over an enclosing :func:`group_context`: while a
    per-rid context is active, fan-out is suppressed and the record lands
    only in this rid's buffer."""
    return _RolloutContext(rollout_id)


def group_context(rollout_ids: Iterable[str]) -> AbstractContextManager[None]:
    """Bind a group of rollout_ids to the current asyncio task.

    Used by env_service around :meth:`compute_group_reward` so logs from
    the body — which has no single rid to attribute to — fan out to every
    rid in the group by default. An exception escaping the body is logged
    via :func:`logger.exception` and likewise fans out to every rid's
    buffer (replacing the deprecated ``maybe_capture_group_errors`` helper).

    Env authors who want per-rid attribution inside the group method body
    wrap individual iterations in :func:`rollout_context` — the per-rid
    binding overrides this fan-out for the duration of that block."""
    return _GroupContext(rollout_ids)
