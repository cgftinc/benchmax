"""Cheapest-adequate backend selection."""

from __future__ import annotations

from castform_router.types import Backend, Prediction


def select_backend(
    backends: tuple[Backend, ...], predictions: tuple[Prediction, ...], *, quality_threshold: float
) -> tuple[Backend, str]:
    if not 0 <= quality_threshold <= 1:
        raise ValueError("quality_threshold must be between 0 and 1")
    by_name = {backend.name: backend for backend in backends}
    scores = {prediction.backend: prediction for prediction in predictions}
    if not by_name or len(by_name) != len(backends):
        raise ValueError("backends must be non-empty and uniquely named")
    if set(scores) != set(by_name) or len(scores) != len(predictions):
        raise ValueError("scorer must return exactly one prediction per backend")
    adequate = [item for item in predictions if item.success_probability >= quality_threshold]
    if adequate:
        chosen = min(adequate, key=lambda item: by_name[item.backend].estimated_cost_usd)
        return by_name[chosen.backend], "cheapest_adequate"
    chosen = max(predictions, key=lambda item: item.success_probability)
    return by_name[chosen.backend], "highest_quality_fallback"
