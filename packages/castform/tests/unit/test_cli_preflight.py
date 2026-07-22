"""Unit tests for doctor's optional-dependency preflight."""

from __future__ import annotations

import importlib.util

from castform.cli._preflight import extra_is_installed


def test_extra_is_installed_known_and_unknown():
    assert extra_is_installed("rag") is (
        importlib.util.find_spec("keybert") is not None
    )
    assert extra_is_installed("does-not-exist") is False
