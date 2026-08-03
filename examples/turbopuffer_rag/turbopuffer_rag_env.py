"""Training-time environment for the TurboPuffer RAG example."""

from __future__ import annotations

from benchmax.envs import InjectedAuth
from benchmax.rag.embed import OpenAIEmbedder
from benchmax.rag.env import RagEnv
from search import TurbopufferSearch

NAMESPACE = "benchmax-rag"
REGION = "aws-us-east-1"
MAX_SEARCH_CALLS = 8


class TurbopufferRagEnv(RagEnv):
    system_prompt = RagEnv.render_system_prompt(
        corpus_description="the documents indexed in the TurboPuffer RAG example namespace",
        max_search_calls=MAX_SEARCH_CALLS,
    )

    def __init__(
        self,
        *,
        judge_base_url: str,
        embedding_base_url: str,
        api_key: str,
    ) -> None:
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            base_url=embedding_base_url,
            auth=InjectedAuth("embedding"),
        )
        super().__init__(
            search=TurbopufferSearch(
                NAMESPACE,
                region=REGION,
                embed_fn=embedder,
                api_key=api_key,
            ),
            judge_base_url=judge_base_url,
            judge_model="gpt-5.4-mini",
            judge_auth=InjectedAuth("judge"),
            max_search_calls=MAX_SEARCH_CALLS,
        )


__all__ = ["TurbopufferRagEnv"]
