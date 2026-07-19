"""Unit tests for the ELO judge's pure helpers (no judge/network):
_make_slices (coverage), _rating_to_band (calibration), _bt_fit (recovers order)."""

from collections import Counter

import pytest

from telestich_env import (
    ACCEPTABLE_EDGE,
    ELO_GREAT_RATING,
    ELO_SLICE,
    GREAT_EDGE,
    _bt_fit,
    _make_slices,
    _rating_to_band,
)


@pytest.mark.parametrize("n", list(range(1, 13)))
def test_make_slices_cover_every_poem(n):
    slices = _make_slices(n)
    assert all(len(sl) <= ELO_SLICE for sl in slices)
    assert all(0 <= i < n for sl in slices for i in sl)
    cnt = Counter(i for sl in slices for i in sl)
    assert set(cnt) == set(range(n))  # every poem appears
    if n > ELO_SLICE:
        # past the single-slice case, every poem is cross-comparable (≥2 slices)
        assert min(cnt.values()) >= 2
        assert all(len(set(sl)) == len(sl) for sl in slices)  # no dup within a slice


def test_make_slices_small_group_single_call():
    assert _make_slices(ELO_SLICE) == [list(range(ELO_SLICE))]
    assert _make_slices(1) == [[0]]


def test_rating_to_band_calibrated_at_anchors():
    assert _rating_to_band(0.0) == pytest.approx(ACCEPTABLE_EDGE, abs=1e-6)
    assert _rating_to_band(ELO_GREAT_RATING) == pytest.approx(GREAT_EDGE, abs=1e-6)


def test_rating_to_band_monotonic_and_bounded():
    rs = [-8, -3, -1, 0, 0.6, ELO_GREAT_RATING, 3, 8]
    bands = [_rating_to_band(r) for r in rs]
    assert bands == sorted(bands)  # monotonic increasing
    assert all(0.0 < b < 1.0 for b in bands)  # never floors to 0 / saturates to 1
    # below the acceptable anchor → below the band edge; above great → above its edge
    assert _rating_to_band(-2.0) < ACCEPTABLE_EDGE
    assert _rating_to_band(3.0) > GREAT_EDGE


def test_bt_fit_pins_anchors_and_recovers_order():
    # players 0,1,2 = poems; 3 = acceptable (rating 0.0), 4 = great (rating ELO_GREAT_RATING)
    fixed = {3: 0.0, 4: ELO_GREAT_RATING}
    # poem0 beats everyone; poem1 beats acceptable but loses to great; poem2 loses to all
    pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (4, 1),
        (3, 2),
        (4, 2),
    ]
    R = _bt_fit(pairs * 3, fixed, n_players=5)
    assert R[3] == 0.0 and R[4] == ELO_GREAT_RATING  # anchors stay pinned
    assert R[0] > R[1] > R[2]  # poem order recovered
    assert R[1] > 0.0  # beats acceptable → above the baseline rating
    assert R[2] < 0.0  # below acceptable
