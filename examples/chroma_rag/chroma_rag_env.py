"""Training-time environment for the Chroma RAG example."""

from __future__ import annotations

from benchmax.envs import InjectedAuth
from benchmax.rag.embed import OpenAIEmbedder
from benchmax.rag.env import RagEnv
from search import ChromaSearch

COLLECTION_NAME = "benchmax-rag"
MAX_SEARCH_CALLS = 8


class ChromaRagEnv(RagEnv):
    system_prompt = RagEnv.render_system_prompt(
        corpus_description="the documents indexed in the Chroma RAG example collection",
        max_search_calls=MAX_SEARCH_CALLS,
    )

    def __init__(
        self,
        *,
        judge_base_url: str,
        embedding_base_url: str,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
    ) -> None:
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            base_url=embedding_base_url,
            auth=InjectedAuth("embedding"),
        )
        super().__init__(
            search=ChromaSearch(
                COLLECTION_NAME,
                embed_fn=embedder,
                tenant=tenant,
                database=database,
                host=host,
                port=port,
                ssl=ssl,
                api_key=api_key,
            ),
            judge_base_url=judge_base_url,
            judge_model="gpt-5.4-mini",
            judge_auth=InjectedAuth("judge"),
            max_search_calls=MAX_SEARCH_CALLS,
        )


__all__ = ["ChromaRagEnv"]
