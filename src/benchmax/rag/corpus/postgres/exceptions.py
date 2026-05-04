"""Exceptions for corpus API client."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Corpus


class CorpusAPIError(Exception):
    """Base exception for corpus API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(CorpusAPIError):
    """API key is invalid or missing."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, 401)


class CorpusLimitError(CorpusAPIError):
    """Maximum corpus limit (5) reached."""

    def __init__(self, existing_corpora: list[Corpus] | None = None):
        super().__init__("Maximum of 5 corpora per user reached", 400)
        self.existing_corpora = existing_corpora or []


class ChunkLimitError(CorpusAPIError):
    """Maximum chunk limit (10,000) per corpus reached."""

    def __init__(self, corpus_id: str, current_count: int, attempted_add: int):
        super().__init__(
            f"Would exceed 10,000 chunk limit. Current: {current_count}, "
            f"Attempted to add: {attempted_add}",
            400,
        )
        self.corpus_id = corpus_id
        self.current_count = current_count
        self.attempted_add = attempted_add


class CorpusNotFoundError(CorpusAPIError):
    """Corpus with given ID not found."""

    def __init__(self, corpus_id: str):
        super().__init__(f"Corpus not found: {corpus_id}", 404)
        self.corpus_id = corpus_id
