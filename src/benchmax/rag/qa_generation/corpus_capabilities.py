from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CorpusCapabilities:
    """
    Describes which metadata fields are available in the corpus.
    Detected once at pipeline initialization and threaded through
    downstream components to enable graceful degradation.
    """

    has_document_ids: bool
    has_section_headers: bool
    has_sequential_links: bool
    has_dates: bool

    @property
    def can_detect_document_boundaries(self) -> bool:
        """Enables cross-document multi-hop targeting and co-location detection."""
        return self.has_document_ids

    @property
    def can_target_sequential_reasoning(self) -> bool:
        """Enables procedural/cause-effect chain questions."""
        return self.has_sequential_links

    @property
    def available_reasoning_modes(self) -> list[str]:
        """Return reasoning modes this corpus supports."""
        modes = ["factual", "inference"]
        if self.has_dates:
            modes.append("temporal")
        if self.has_sequential_links:
            modes.append("sequential")
        return modes

    @classmethod
    def detect(cls, sample_chunks: list[Any]) -> "CorpusCapabilities":
        """Auto-detect capabilities from a sample of chunks."""
        if not sample_chunks:
            return cls(
                has_document_ids=False,
                has_section_headers=False,
                has_sequential_links=False,
                has_dates=False,
            )

        metadata: dict[str, Any] = {}
        for chunk in sample_chunks:
            if hasattr(chunk, "metadata_dict"):
                md = chunk.metadata_dict
                if md:
                    metadata = md
                    break
            elif hasattr(chunk, "metadata") and chunk.metadata:
                metadata = chunk.metadata
                break
            elif isinstance(chunk, dict) and chunk.get("metadata"):
                metadata = chunk["metadata"]
                break

        has_document_ids = any(
            key in metadata for key in ("document_id", "file_name", "file", "source", "doc_id")
        )
        has_section_headers = any(
            key in metadata
            for key in ("section_header", "header", "h1", "h2", "h3", "title", "heading")
        )
        has_sequential_links = any(
            key in metadata for key in ("prev_chunk_id", "next_chunk_id", "prev_id", "next_id")
        ) or ("file" in metadata and "index" in metadata)
        has_dates = any(key in metadata for key in ("date_start", "date_end", "date", "timestamp"))

        return cls(
            has_document_ids=has_document_ids,
            has_section_headers=has_section_headers,
            has_sequential_links=has_sequential_links,
            has_dates=has_dates,
        )

    def describe(self) -> str:
        """Human-readable summary for logging."""
        modes = self.available_reasoning_modes
        return (
            f"CorpusCapabilities("
            f"doc_ids={self.has_document_ids}, "
            f"headers={self.has_section_headers}, "
            f"sequential={self.has_sequential_links}, "
            f"dates={self.has_dates}) "
            f"→ reasoning modes: {modes}"
        )
