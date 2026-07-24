"""Concrete neon RAG training env over the gitlab_handbook HYBRID corpus.

Wires the live `gitlab_handbook_neon` corpus into the post-harbor
`SearchEnv(BaseEnv)` search seam via constructor DI, running HYBRID retrieval
(bm25 + vector, fused by RRF):

    SearchEnv(search=NeonSearch(<logical table>, dsn_provider=<baked ro dsn str>,
                                embed_fn=PlatformEmbedFn(...)), ...)

Neon does NOT become a HarborEnv — harbor generalized the launch path but left
the search seam untouched. `NeonSearch` plugs in as a `SearchClient`.

Credential transport (two seams, both call-time, no live socket pickled):
  1. the RO DSN is baked as a `str` (never `None`) so `resolve_read_dsn_provider`
     captures it in a closure — a `None` would resolve `NEON_CORPUS_DSN_RO` from
     the environment at unpickle, and the trainer's Ray actor has no such env
     var, so the run would lose its only retrieval backend. The DSN string is an
     accepted at-rest tradeoff (RO grant, SELECT-only, never logged).
  2. the query embedder is a `PlatformEmbedFn` — pickle-safe (its live OpenAI
     client is dropped by `__getstate__` and rebuilt lazily post-unpickle). Its
     credential is resolved at CALL time via the platform seam (`api_key=""` →
     `ACT_AS_TOKEN_PATH` / `PLATFORM_API_KEY`), so NO key literal lands in the
     pickle. Only its non-secret `base_url` (the judge's llm-proxy) is baked.

`neon_env_constructor_args` returns exactly the `constructor_args` dict that
`benchmax.bundle` pickles alongside `NeonRagEnv` as `(env_class, constructor_args)`;
`build_env` is the convenience wrapper the smoke drives.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# `postgres-search/main.py` is a library env shipped as an un-packaged module
# (py-modules = []); import it by path the same way its own test-suite does.
_POSTGRES_SEARCH = Path(__file__).resolve().parents[1] / "postgres-search"
if str(_POSTGRES_SEARCH) not in sys.path:
    sys.path.insert(0, str(_POSTGRES_SEARCH))

from main import SearchEnv, TOOL_OUTPUT_TRUNCATION_SUFFIX  # noqa: E402

from castform.rag.corpus.embed import PlatformEmbedFn  # noqa: E402
from castform.rag.corpus.neon.search import NeonSearch  # noqa: E402

# Logical corpus name (NeonSearch resolves the versioned physical table); see
# examples/gitlab_handbook_bm25_neon/handbook_corpus.py:LOGICAL_NAME.
CORPUS_TABLE = "gitlab_handbook_neon"
CORPUS_DESCRIPTION = "the GitLab company handbook"

# Judge / policy endpoint. llm.castform.dev serves gpt-5.4* + grok; gpt-5.4-mini
# is tool-call capable and cheap, used for both the rollout policy and the judge.
# The OpenAI-compatible API lives under /v1 (the SDK appends /chat/completions).
# The query embedder targets the SAME endpoint (text-embedding-3-large) so it
# authenticates the same in-actor way the judge does.
JUDGE_BASE_URL = "https://llm.castform.dev/v1"
JUDGE_MODEL = "gpt-5.4-mini"

MAX_SEARCH_CALLS = 6

# Full top-10 hybrid render must never be truncated (the base cap is 10k). A top-10
# handbook render is ~18.6k chars typical / ~21.2k max, so 22k keeps it whole.
MAX_TOOL_OUTPUT_CHARS = 22000


def _build_embed_fn() -> PlatformEmbedFn:
    """Query embedder for the vector leg — pickle-safe, no baked secret.

    ``base_url`` is pinned to the judge's llm-proxy (a non-secret) so the vector
    leg hits the same reachable endpoint the judge uses; ``api_key`` is left empty
    so the bearer resolves at CALL time through the platform credential seam
    (``ACT_AS_TOKEN_PATH`` in-actor / ``PLATFORM_API_KEY`` self-serve) — nothing
    is baked into the pickle.
    """
    return PlatformEmbedFn(base_url=JUDGE_BASE_URL)


class NeonRagEnv(SearchEnv):
    """SearchEnv bound to the live gitlab_handbook_neon corpus in HYBRID mode.

    A named, self-contained env class in THIS example so the trainer's bundle is
    self-describing and does not top-level on postgres-search's ``SearchEnv``.
    The judge endpoint, model, search budget and system prompt are baked as class
    config; the only constructor arg is the ``NeonSearch`` client (which carries
    the baked RO DSN and the query embedder). Pickled BY VALUE alongside the
    ``main`` module (see run.py).

    With an ``embed_fn`` present, ``NeonSearch.available_modes`` is
    ``[hybrid, lexical, vector]`` and ``SearchEnv`` selects ``hybrid`` as the
    default search mode, so each search fuses bm25 + vector via RRF.
    """

    system_prompt = SearchEnv.render_system_prompt(
        corpus_description=CORPUS_DESCRIPTION, max_search_calls=MAX_SEARCH_CALLS
    )

    def __init__(self, search: Any, **overrides: Any) -> None:
        overrides.setdefault("judge_base_url", JUDGE_BASE_URL)
        overrides.setdefault("judge_model", JUDGE_MODEL)
        overrides.setdefault("max_search_calls", MAX_SEARCH_CALLS)
        overrides.setdefault("system_prompt", self.system_prompt)
        super().__init__(search, **overrides)

    @staticmethod
    def _truncate_tool_output(
        text: str,
        max_chars: int = MAX_TOOL_OUTPUT_CHARS,
        suffix: str = TOOL_OUTPUT_TRUNCATION_SUFFIX,
    ) -> str:
        # Raise the base 10k cap to 22k so a full top-10 hybrid render survives.
        return SearchEnv._truncate_tool_output(text, max_chars=max_chars, suffix=suffix)


def neon_env_constructor_args(dsn: str) -> dict[str, Any]:
    """`constructor_args` for :class:`NeonRagEnv` — the baked-DSN HYBRID search client.

    Judge / prompt / budget are baked into the class, so the bundle's
    ``constructor_args`` is a single ``NeonSearch`` carrying the RO DSN string and
    a ``PlatformEmbedFn`` query embedder (which enables the vector leg → hybrid).
    """
    if not isinstance(dsn, str) or not dsn:
        raise ValueError("dsn must be a non-empty read-only dsn string")
    return {
        "search": NeonSearch(
            CORPUS_TABLE, dsn_provider=dsn, embed_fn=_build_embed_fn()
        )
    }


def neon_search_constructor_args(dsn: str) -> dict[str, Any]:
    """Build the `SearchEnv` `constructor_args` with the RO DSN baked as a str.

    This dict is exactly what `benchmax.bundle` cloudpickles alongside the env
    class. `dsn` MUST be a resolved RO DSN string, not `None` — the baked closure
    is the whole point of the credential-transport test. The ``PlatformEmbedFn``
    enables the vector leg so ``hybrid`` mode is available.
    """
    if not isinstance(dsn, str) or not dsn:
        raise ValueError("dsn must be a non-empty read-only dsn string")
    return {
        "search": NeonSearch(
            CORPUS_TABLE, dsn_provider=dsn, embed_fn=_build_embed_fn()
        ),
        "judge_base_url": JUDGE_BASE_URL,
        "judge_model": JUDGE_MODEL,
        "max_search_calls": MAX_SEARCH_CALLS,
        "system_prompt": SearchEnv.render_system_prompt(
            corpus_description=CORPUS_DESCRIPTION,
            max_search_calls=MAX_SEARCH_CALLS,
        ),
    }


def build_env(dsn: str) -> SearchEnv:
    """Construct a `SearchEnv` over the neon hybrid corpus with a baked RO DSN."""
    return SearchEnv(**neon_search_constructor_args(dsn))


# Two QA rows in SearchEnv's row contract (question / answer / reference_chunks),
# reshaped from examples/gitlab_handbook_bm25_neon/datasets/verdicts_v2.jsonl
# natural-language keeps. `reference_chunks` carry the gold source file (metadata
# key `file`) so the UNGATED retrieval_hit term can score when the model cites it.
GITLAB_SMOKE_ROWS: list[dict[str, Any]] = [
    {
        "question": (
            "Where can I find the GitLab Security dashboard enablement material, "
            "and what is the enablement link?"
        ),
        "answer": (
            "GitLab publishes a 'GitLab Security dashboard enablement' resource, "
            "linked as an enablement video, alongside its product training and "
            "sales enablement materials."
        ),
        "reference_chunks": [
            {
                "metadata": {
                    "file": "marketing/brand-and-product-marketing/"
                    "product-and-solution-marketing/reseller-kit.md"
                }
            }
        ],
    },
    {
        "question": (
            "In GitLab's Jobs-To-Be-Done (JTBD) canvas, how is a 'Job Performer' "
            "defined?"
        ),
        "answer": (
            "A Job Performer is the person executing a specific job, distinct "
            "from their job title."
        ),
        "reference_chunks": [
            {"metadata": {"file": "product/ux/jobs-to-be-done/jtbd-playbook.md"}}
        ],
    },
]
