"""Stable categorical token targets for the learned router contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TokenClass = Literal["input", "cache_read", "output"]


@dataclass(frozen=True, slots=True)
class TokenBand:
    name: str
    upper_exclusive: int | None
    representative: int


_CONTEXT_BANDS = (
    TokenBand("zero", 1, 0),
    TokenBand("under_64k", 65_536, 32_768),
    TokenBand("64k_256k", 262_144, 131_072),
    TokenBand("256k_1m", 1_048_576, 524_288),
    TokenBand("1m_4m", 4_194_304, 2_097_152),
    TokenBand("4m_plus", None, 6_291_456),
)

_OUTPUT_BANDS = (
    TokenBand("zero", 1, 0),
    TokenBand("under_4k", 4_096, 2_048),
    TokenBand("4k_8k", 8_192, 6_144),
    TokenBand("8k_16k", 16_384, 12_288),
    TokenBand("16k_32k", 32_768, 24_576),
    TokenBand("32k_plus", None, 49_152),
)


def token_band_names(token_class: TokenClass) -> tuple[str, ...]:
    return tuple(band.name for band in _bands(token_class))


def token_band_representatives(token_class: TokenClass) -> dict[str, int]:
    return {band.name: band.representative for band in _bands(token_class)}


def token_band_for_count(value: int, token_class: TokenClass) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token count must be a non-negative integer")
    for band in _bands(token_class):
        if band.upper_exclusive is None or value < band.upper_exclusive:
            return band.name
    raise AssertionError("token band table must end with an unbounded band")


def token_band_representative(name: object, token_class: TokenClass) -> int:
    if not isinstance(name, str):
        raise ValueError(f"{token_class}_token_band must be a string")
    for band in _bands(token_class):
        if name == band.name:
            return band.representative
    allowed = ", ".join(token_band_names(token_class))
    raise ValueError(
        f"unknown {token_class}_token_band {name!r}; expected one of: {allowed}"
    )


def _bands(token_class: TokenClass) -> tuple[TokenBand, ...]:
    if token_class in {"input", "cache_read"}:
        return _CONTEXT_BANDS
    if token_class == "output":
        return _OUTPUT_BANDS
    raise ValueError(f"unknown token class: {token_class!r}")
