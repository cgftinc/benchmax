"""Synthetic dominant-color examples for the multi-turn vision memory game."""

from __future__ import annotations

import base64
import io
import random
from collections.abc import Iterable, Mapping
from typing import Any

from benchmax.envs.base import JsonRow
from benchmax.envs.dataset import Dataset
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import Example

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 90, 220),
    "yellow": (235, 210, 50),
    "purple": (150, 60, 200),
    "orange": (240, 140, 30),
    "pink": (245, 150, 190),
    "brown": (140, 90, 45),
    "black": (25, 25, 25),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "cyan": (60, 200, 220),
    "magenta": (220, 60, 180),
    "teal": (30, 140, 140),
    "navy": (25, 35, 110),
    "olive": (128, 128, 40),
}


class DominantColorDataset(Dataset[JsonRow]):
    """Build prompt messages carrying the system rules and the first image."""

    def __init__(
        self,
        specs: Iterable[Mapping[str, Any]],
        *,
        system_prompt: str,
        image_size: int,
        tile_grid: int,
        dominant_fraction: float,
        noise_sigma: float,
    ) -> None:
        super().__init__(
            [
                _example(
                    dict(spec),
                    system_prompt=system_prompt,
                    image_size=image_size,
                    tile_grid=tile_grid,
                    dominant_fraction=dominant_fraction,
                    noise_sigma=noise_sigma,
                )
                for spec in specs
            ]
        )


def render_tile_image_uri(
    dominant: str,
    *,
    size: int,
    tile_grid: int,
    dominant_fraction: float,
    sigma: float,
    seed: str,
) -> str:
    """Checkered tile image whose dominant color is the answer, as a PNG data URI.

    ``dominant_fraction`` of the tiles carry the dominant color; the rest are
    random other palette colors. Seeded gaussian pixel noise keeps every PNG
    byte-unique (identical PNGs would collide in per-session hashed media
    loading downstream) and blurs the tile boundaries slightly.
    """

    from PIL import Image

    if size % tile_grid != 0:
        raise ValueError("image size must be divisible by tile_grid")
    rng = random.Random(seed)

    total_tiles = tile_grid * tile_grid
    dominant_count = max(1, round(dominant_fraction * total_tiles))
    others = [name for name in PALETTE if name != dominant]
    tile_names = [dominant] * dominant_count + [
        rng.choice(others) for _ in range(total_tiles - dominant_count)
    ]
    rng.shuffle(tile_names)
    tile_colors = [PALETTE[name] for name in tile_names]

    tile_px = size // tile_grid
    pixels = []
    for y in range(size):
        row_base = (y // tile_px) * tile_grid
        for x in range(size):
            base = tile_colors[row_base + (x // tile_px)]
            pixels.append(
                tuple(
                    min(255, max(0, round(channel + rng.gauss(0.0, sigma))))
                    for channel in base
                )
            )
    image = Image.new("RGB", (size, size))
    image.putdata(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _example(
    spec: JsonRow,
    *,
    system_prompt: str,
    image_size: int,
    tile_grid: int,
    dominant_fraction: float,
    noise_sigma: float,
) -> Example[JsonRow]:
    colors = spec.get("colors")
    noise_seed = spec.get("noise_seed")
    if (
        not isinstance(colors, list)
        or not colors
        or any(color not in PALETTE for color in colors)
    ):
        raise ValueError("dominant-color specs require a list of palette colors")
    if not isinstance(noise_seed, int):
        raise ValueError("dominant-color specs require an integer noise_seed")

    first_image = render_tile_image_uri(
        colors[0],
        size=image_size,
        tile_grid=tile_grid,
        dominant_fraction=dominant_fraction,
        sigma=noise_sigma,
        seed=f"{noise_seed}:0",
    )
    payload: JsonRow = {
        "prompt_messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": first_image}},
                    {"type": "text", "text": f"This is image 1 of {len(colors)}."},
                ],
            },
        ],
        "answer": ", ".join(colors),
        "colors": list(colors),
        "noise_seed": noise_seed,
    }
    return Example(id=canonical_example_id(payload), payload=payload)


__all__ = ["PALETTE", "DominantColorDataset", "render_tile_image_uri"]
