"""Data passed through the minimal model router."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Backend:
    name: str
    model: str
    provider: str
    estimated_cost_usd: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Prediction:
    backend: str
    success_probability: float
    input_token_band: str
    cache_read_token_band: str
    output_token_band: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutingRequest:
    request_id: str
    task: str
    backends: tuple[Backend, ...]
