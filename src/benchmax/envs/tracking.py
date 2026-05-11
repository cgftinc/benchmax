from __future__ import annotations

import inspect
import logging
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional

LOGGER = logging.getLogger(__name__)

TRACKING_RUN_ID_KEY = "__benchmax_telemetry_run_id"
TRACKING_API_KEY_KEY = "__benchmax_telemetry_api_key"

_ACTIVE_TRACKER: ContextVar[Any | None] = ContextVar(
    "benchmax_active_telemetry_tracker", default=None
)
# Bounded LRU. Long-running pooled workers (one process, many run_ids) would
# otherwise accumulate trackers indefinitely.
_TRACKER_CACHE_MAX = 128
_TRACKER_CACHE: "OrderedDict[tuple[Optional[str], Optional[str]], Any | None]" = OrderedDict()


@dataclass(frozen=True)
class TrackingConfig:
    run_id: Optional[str] = None
    api_key: Optional[str] = None

    def resolved_run_id(self) -> Optional[str]:
        return self.run_id

    def is_enabled(self) -> bool:
        return bool(self.resolved_run_id())


def _build_tracker(config: TrackingConfig) -> Any | None:
    if not config.is_enabled():
        return None

    try:
        import job_telemetry
    except Exception as e:
        LOGGER.debug("job_telemetry import failed; env tracking disabled: %s", e)
        return None

    try:
        # api_key removed post-act-as-rotation: job_telemetry is tokenless;
        # the localhost otelcol sidecar's bearertokenauth attaches the
        # rotated act-as bearer on the outbound hop.
        job_telemetry.init(
            run_id=config.resolved_run_id(),
        )
    except Exception as e:
        LOGGER.debug("job_telemetry init failed; env tracking disabled: %s", e)
        return None

    return job_telemetry


def get_tracker(config: TrackingConfig | None) -> Any | None:
    if config is None:
        return None

    key = (config.resolved_run_id(), config.api_key)
    if key in _TRACKER_CACHE:
        # Touch for LRU recency.
        _TRACKER_CACHE.move_to_end(key)
        return _TRACKER_CACHE[key]

    tracker = _build_tracker(config)
    _TRACKER_CACHE[key] = tracker
    if len(_TRACKER_CACHE) > _TRACKER_CACHE_MAX:
        _TRACKER_CACHE.popitem(last=False)
    return tracker


@contextmanager
def tracking_context(config: TrackingConfig | None) -> Iterator[None]:
    token = _ACTIVE_TRACKER.set(get_tracker(config))
    try:
        yield
    finally:
        _ACTIVE_TRACKER.reset(token)


def log_env(rollout_id: str, message: str, commit: bool = False, flush_k: int = 30) -> None:
    tracker = _ACTIVE_TRACKER.get()
    if tracker is None:
        return

    try:
        tracker.log_environment(rollout_id, str(message), commit, flush_k)
    except Exception as e:
        LOGGER.debug("log_environment failed: %s", e)


def with_tracking(
    config_resolver: Callable[..., TrackingConfig | None],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function so calls run with an active env tracking context."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracking_context(config_resolver(*args, **kwargs)):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracking_context(config_resolver(*args, **kwargs)):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def to_tracking_payload(config: TrackingConfig | None) -> Dict[str, str]:
    if config is None:
        return {}

    payload: Dict[str, str] = {}
    resolved_run_id = config.resolved_run_id()
    if resolved_run_id:
        payload[TRACKING_RUN_ID_KEY] = resolved_run_id
    if config.api_key:
        payload[TRACKING_API_KEY_KEY] = config.api_key
    return payload


def pop_tracking_config(payload: Dict[str, Any]) -> TrackingConfig:
    run_id = payload.pop(TRACKING_RUN_ID_KEY, None)
    api_key = payload.pop(TRACKING_API_KEY_KEY, None)
    return TrackingConfig(run_id=run_id, api_key=api_key)
