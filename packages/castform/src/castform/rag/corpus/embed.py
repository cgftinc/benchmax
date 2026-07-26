"""OpenAI-compatible query/document embedding for async RAG search clients.

The rollout-facing provider search clients (Turbopuffer / Pinecone / Chroma)
accept an optional async ``embed_fn``. Wiring one in makes vector / hybrid
retrieval work regardless of how the user's index was built — turbopuffer
vector/hybrid (no server-side embed), a pinecone index NOT on the hosted model,
or a non-cloud chroma collection.

``qa-gen`` does NOT need this (it reads chunks directly); it's only for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmax.auth import ModelAuth
from castform.model_auth import create_async_openai_client

# The served, non-hidden embeddings model in the llm-proxy catalog.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


@dataclass(frozen=True, slots=True)
class OpenAIEmbedder:
    """Pickle-safe embedding callable with explicit call-time model auth.

    ``auth`` follows the same contract as :class:`benchmax.rewards.Judge`.
    Managed environments use ``InjectedAuth("embedding")``; customer-owned
    endpoints use ``StaticBearerAuth``. No platform or SDK environment
    credential is inferred.
    """

    model: str
    base_url: str
    auth: ModelAuth
    timeout: float | None = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.auth, ModelAuth):
            raise TypeError("embedding auth must implement ModelAuth")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("embedding model must be non-empty")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("embedding base_url must be non-empty")
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or self.timeout <= 0
        ):
            raise ValueError("embedding timeout must be positive or None")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("embedding max_retries must be non-negative")

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) for text in texts
        ):
            raise TypeError("embedding input must be a list of strings")
        if not texts:
            return []

        client = create_async_openai_client(
            model=self.model,
            base_url=self.base_url,
            auth=self.auth,
            request_id="rag-embedding",
            max_retries=self.max_retries,
        )
        try:
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
                timeout=self.timeout,
            )
            return [item.embedding for item in response.data]
        finally:
            await client.close()
