"""Adapter registry — maps provider names to TraceAdapter implementations."""

from __future__ import annotations

from benchmax.traces.adapter import TraceAdapter

_ADAPTERS: dict[str, type[TraceAdapter]] = {}


def _register_builtins() -> None:
    from benchmax.traces.braintrust.adapter import BraintrustTraceAdapter

    _ADAPTERS["braintrust"] = BraintrustTraceAdapter


def get_adapter_class(provider: str) -> type[TraceAdapter]:
    """Return the adapter class for *provider*.

    Callers are responsible for constructing the adapter with
    provider-specific credentials::

        AdapterClass = get_adapter_class("braintrust")
        adapter = AdapterClass(api_key="sk-...")

    Raises ``ValueError`` for unknown providers.
    """
    if not _ADAPTERS:
        _register_builtins()
    cls = _ADAPTERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown trace provider: {provider!r}. Available: {sorted(_ADAPTERS)}")
    return cls
