from __future__ import annotations

import importlib


def require_harbor() -> None:
    """Fail at Harbor use, while keeping ordinary Benchmax imports lightweight."""

    try:
        importlib.import_module("harbor")
    except ModuleNotFoundError as exc:
        if exc.name != "harbor":
            raise
        raise ImportError(
            "HarborEnv requires the optional Harbor dependency; install "
            "`benchmax[harbor]` plus the selected sandbox provider extra"
        ) from None
