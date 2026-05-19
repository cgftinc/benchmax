"""Data models for corpus API responses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Corpus:
    """Represents a corpus in the API."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "Corpus":
        """Create from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00")),
        )


@dataclass
class CorpusChunk:
    """A chunk as returned from the API search."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None


@dataclass
class UploadResult:
    """Result from uploading chunks."""

    success: bool
    inserted_count: int
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def failed_count(self) -> int:
        return len(self.chunk_ids) - self.inserted_count if self.chunk_ids else 0


@dataclass
class SearchResult:
    """Result from a search query."""

    results: list[CorpusChunk]
    total: int
    query: str

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)
