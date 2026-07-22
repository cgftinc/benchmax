from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from castform.rag.chunkers.models import Chunk


@dataclass
class AnchorBundle:
    """Selection intent used to drive multi-hop generation."""

    primary_chunk: Chunk
    secondary_chunks: list[Chunk]
    target_hop_count: int
    structural_hints: dict = field(default_factory=dict)
