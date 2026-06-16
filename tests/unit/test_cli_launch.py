"""Slice 1.5 offline: launcher-arg validation against the platform schema."""

from __future__ import annotations

import pytest

from benchmax.cli.launch import _build_launcher_args, _coerce_arg
from benchmax.platform.client import LaunchArgSpec


def _spec(name, type_, **kw):
    return LaunchArgSpec(
        name=name, label=name, type=type_, required=False, description="", **kw
    )


SPECS = [
    _spec("model", "string", enum=("qwen-4b", "qwen-35b")),
    _spec("max_rollout_len", "integer", min=128, max=8000, warn_above=4000),
    _spec("max_turns", "integer"),
]


def test_coerce_types():
    assert _coerce_arg(_spec("n", "integer"), "5") == 5
    assert _coerce_arg(_spec("x", "number"), "1.5") == 1.5
    assert _coerce_arg(_spec("b", "boolean"), "true") is True
    assert _coerce_arg(_spec("b", "boolean"), "no") is False
    assert _coerce_arg(_spec("s", "string"), "hi") == "hi"


def test_build_ok():
    out = _build_launcher_args(SPECS, ["model=qwen-4b", "max_rollout_len=2000"])
    assert out == {"model": "qwen-4b", "max_rollout_len": 2000}


def test_build_rejects_unknown_key():
    # max_response_len is the classic wrong knob — must be rejected, not silently sent.
    with pytest.raises(SystemExit, match="Unknown launch arg 'max_response_len'"):
        _build_launcher_args(SPECS, ["max_response_len=2000"])


def test_build_rejects_bad_enum():
    with pytest.raises(SystemExit, match="must be one of"):
        _build_launcher_args(SPECS, ["model=gpt-9"])


def test_build_rejects_out_of_range():
    with pytest.raises(SystemExit, match="above max"):
        _build_launcher_args(SPECS, ["max_rollout_len=99999"])
    with pytest.raises(SystemExit, match="below min"):
        _build_launcher_args(SPECS, ["max_rollout_len=1"])


def test_build_warns_above_soft_cap(capsys):
    _build_launcher_args(SPECS, ["max_rollout_len=6000"])
    assert "soft cap" in capsys.readouterr().err


def test_build_bad_pair():
    with pytest.raises(SystemExit, match="key=value"):
        _build_launcher_args(SPECS, ["model"])
