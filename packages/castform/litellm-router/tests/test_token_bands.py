from __future__ import annotations

import pytest
from castform_router.token_bands import (
    TokenClass,
    token_band_for_count,
    token_band_representative,
    token_band_representatives,
)


@pytest.mark.parametrize(
    ("value", "token_class", "expected"),
    [
        (0, "input", "zero"),
        (1, "input", "under_64k"),
        (65_535, "cache_read", "under_64k"),
        (65_536, "input", "64k_256k"),
        (1_048_576, "input", "1m_4m"),
        (4_194_304, "cache_read", "4m_plus"),
        (4_095, "output", "under_4k"),
        (4_096, "output", "4k_8k"),
        (32_768, "output", "32k_plus"),
    ],
)
def test_token_band_boundaries(
    value: int,
    token_class: TokenClass,
    expected: str,
) -> None:
    assert token_band_for_count(value, token_class) == expected


def test_token_band_representative_is_deterministic() -> None:
    assert token_band_representative("256k_1m", "input") == 524_288
    assert token_band_representative("8k_16k", "output") == 12_288
    assert token_band_representatives("cache_read")["1m_4m"] == 2_097_152


def test_unknown_token_band_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown input_token_band"):
        token_band_representative("huge", "input")
