"""Training-time environment for the Neon RAG example.

This module intentionally imports only Benchmax and the example-local runtime
adapter. Developer-side data preparation and Castform orchestration stay in
``data.py`` and ``main.py`` so they are not captured in the environment bundle.
"""

from __future__ import annotations

from benchmax.envs import InjectedAuth
from benchmax.rag.embed import OpenAIEmbedder
from benchmax.rag.env import RagEnv
from neon_backend.config import CORPUS_NAME
from neon_backend.search import NeonSearch

JUDGE_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_SEARCH_CALLS = 8


class NeonRagEnv(RagEnv):
    """Answer grounded questions with hybrid search over a versioned Neon corpus."""

    system_prompt = RagEnv.render_system_prompt(
        corpus_description="the documents indexed in the Neon RAG example corpus",
        max_search_calls=MAX_SEARCH_CALLS,
    )

    def __init__(
        self,
        *,
        judge_base_url: str,
        embedding_base_url: str,
        search_database_url: str,
        judge_model: str = JUDGE_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        embedder = OpenAIEmbedder(
            model=embedding_model,
            base_url=embedding_base_url,
            auth=InjectedAuth("embedding"),
        )
        super().__init__(
            search=NeonSearch(
                CORPUS_NAME,
                embed_fn=embedder,
                database_url=search_database_url,
            ),
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            judge_auth=InjectedAuth("judge"),
            max_search_calls=MAX_SEARCH_CALLS,
        )


__all__ = ["NeonRagEnv"]
