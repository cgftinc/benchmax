from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional

LOGGER = logging.getLogger(__name__)

TRACKING_EXPERIMENT_ID_KEY = "__benchmax_expt_logger_experiment_id"
TRACKING_API_KEY_KEY = "__benchmax_expt_logger_api_key"

_ACTIVE_TRACKER: ContextVar[Any | None] = ContextVar(
    "benchmax_active_expt_logger_tracker", default=None
)
_TRACKER_CACHE: Dict[tuple[Optional[str], Optional[str]], Any | None] = {}


@dataclass(frozen=True)
class TrackingConfig:
    experiment_id: Optional[str] = None
    api_key: Optional[str] = None

    def resolved_experiment_id(self) -> Optional[str]:
        return self.experiment_id or os.getenv("EXPT_LOGGER_EXPERIMENT_ID")

    def is_enabled(self) -> bool:
        return bool(self.resolved_experiment_id())


def _build_tracker(config: TrackingConfig) -> Any | None:
    if not config.is_enabled():
        return None

    try:
        import expt_logger
    except Exception as e:
        LOGGER.debug("expt_logger import failed; env tracking disabled: %s", e)
        return None

    try:
        run = expt_logger.init(
            experiment_id=config.resolved_experiment_id(),
            api_key=config.api_key,
        )
    except Exception as e:
        LOGGER.debug("expt_logger init failed; env tracking disabled: %s", e)
        return None

    if hasattr(expt_logger, "log_environment"):
        return expt_logger
    if hasattr(run, "log_environment"):
        return run

    LOGGER.debug("expt_logger has no log_environment; env tracking disabled")
    return None


def get_tracker(config: TrackingConfig | None) -> Any | None:
    if config is None:
        return None

    key = (config.resolved_experiment_id(), config.api_key)
    if key not in _TRACKER_CACHE:
        _TRACKER_CACHE[key] = _build_tracker(config)
    return _TRACKER_CACHE[key]


@contextmanager
def tracking_context(config: TrackingConfig | None) -> Iterator[None]:
    token = _ACTIVE_TRACKER.set(get_tracker(config))
    try:
        yield
    finally:
        _ACTIVE_TRACKER.reset(token)


def log_env(rollout_id: str, message: str) -> None:
    tracker = _ACTIVE_TRACKER.get()
    if tracker is None:
        return

    try:
        tracker.log_environment(rollout_id, str(message))
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
    resolved_experiment_id = config.resolved_experiment_id()
    if resolved_experiment_id:
        payload[TRACKING_EXPERIMENT_ID_KEY] = resolved_experiment_id
    if config.api_key:
        payload[TRACKING_API_KEY_KEY] = config.api_key
    return payload


def pop_tracking_config(payload: Dict[str, Any]) -> TrackingConfig:
    experiment_id = payload.pop(TRACKING_EXPERIMENT_ID_KEY, None)
    api_key = payload.pop(TRACKING_API_KEY_KEY, None)
    return TrackingConfig(experiment_id=experiment_id, api_key=api_key)
