"""Concrete neon RAG training env over the gitlab_handbook bm25 corpus.

Wires the live `gitlab_handbook_neon` bm25 corpus into the post-harbor
`SearchEnv(BaseEnv)` search seam via constructor DI:

    SearchEnv(search=NeonSearch(<logical table>, dsn_provider=<baked ro dsn str>,
                                embed_fn=None), ...)

Neon does NOT become a HarborEnv — harbor generalized the launch path but left
the search seam untouched. `NeonSearch` plugs in as a `SearchClient`.

Two authoring traps handled here:
  1. the RO DSN is baked as a `str` (never `None`) so `resolve_read_dsn_provider`
     captures it in a closure — a `None` would resolve `NEON_CORPUS_DSN_RO` from
     the environment at unpickle, defeating the baked-credential transport test.
  2. the lexical/bm25 path is used so `embed_fn=None` — no live embedder closure
     needs to pickle.

`neon_search_constructor_args` returns exactly the `constructor_args` dict that
`benchmax.bundle` pickles alongside `SearchEnv` as `(env_class, constructor_args)`;
`build_env` is the convenience wrapper the smoke drives; `build_bundle` is the
one seam uploaders should call, so the artifact stays self-contained (see
`NEON_BUNDLE_LOCAL_MODULES`).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmax.bundle import Bundle, dump_bundle

# `postgres-search/main.py` is a library env shipped as an un-packaged module
# (py-modules = []); import it by path the same way its own test-suite does.
_POSTGRES_SEARCH = Path(__file__).resolve().parents[1] / "postgres-search"
if str(_POSTGRES_SEARCH) not in sys.path:
    sys.path.insert(0, str(_POSTGRES_SEARCH))

from main import SearchEnv  # noqa: E402

from castform.rag.corpus.neon.search import NeonSearch  # noqa: E402

# Logical corpus name (NeonSearch resolves the versioned physical table); see
# examples/gitlab_handbook_bm25_neon/handbook_corpus.py:LOGICAL_NAME.
CORPUS_TABLE = "gitlab_handbook_neon"
CORPUS_DESCRIPTION = "the GitLab company handbook"

# Judge / policy endpoint. llm.castform.dev serves gpt-5.4* + grok; gpt-5.4-mini
# is tool-call capable and cheap, used for both the rollout policy and the judge.
# The OpenAI-compatible API lives under /v1 (the SDK appends /chat/completions).
JUDGE_BASE_URL = "https://llm.castform.dev/v1"
JUDGE_MODEL = "gpt-5.4-mini"

MAX_SEARCH_CALLS = 6

# The neon provider lives in the `castform` distribution, whose PUBLISHED wheels
# carry no `rag/corpus/neon/` package — an eval image that only `pip install
# castform` cannot unpickle a by-reference NeonSearch. Naming the entry module
# here makes `dump_bundle` cloudpickle it BY VALUE; `auto_local_modules` then
# walks the rest of the referenced castform source in (credentials, query,
# client, schema), so the artifact needs no castform source at all.
NEON_BUNDLE_LOCAL_MODULES: tuple[str, ...] = ("castform.rag.corpus.neon.search",)

# Third-party imports the captured neon source still performs lazily at query
# time. `castform` itself is deliberately absent: nothing of it survives by
# reference, and the published version would be the wrong one anyway.
NEON_BUNDLE_PIP_DEPENDENCIES: tuple[str, ...] = (
    "psycopg[binary]>=3.2.0",
    "pgvector>=0.3.0",
)


def neon_bundle_local_modules() -> list[ModuleType]:
    """Import and return the modules `dump_bundle` must capture by value."""
    return [importlib.import_module(name) for name in NEON_BUNDLE_LOCAL_MODULES]


def neon_search_constructor_args(dsn: str) -> dict[str, Any]:
    """Build the `SearchEnv` `constructor_args` with the RO DSN baked as a str.

    This dict is exactly what `benchmax.bundle` cloudpickles alongside the env
    class. `dsn` MUST be a resolved RO DSN string, not `None` — the baked closure
    is the whole point of the credential-transport test.
    """
    if not isinstance(dsn, str) or not dsn:
        raise ValueError("dsn must be a non-empty read-only dsn string")
    return {
        "search": NeonSearch(CORPUS_TABLE, dsn_provider=dsn, embed_fn=None),
        "judge_base_url": JUDGE_BASE_URL,
        "judge_model": JUDGE_MODEL,
        "max_search_calls": MAX_SEARCH_CALLS,
        "system_prompt": SearchEnv.render_system_prompt(
            corpus_description=CORPUS_DESCRIPTION,
            max_search_calls=MAX_SEARCH_CALLS,
        ),
    }


def build_env(dsn: str) -> SearchEnv:
    """Construct a `SearchEnv` over the neon bm25 corpus with a baked RO DSN."""
    return SearchEnv(**neon_search_constructor_args(dsn))


def build_bundle(
    dsn: str, *, pip_dependencies: tuple[str, ...] | None = None
) -> Bundle:
    """Dump the self-contained deployable bundle for this env.

    The only supported way to build this env's artifact: it captures the neon
    provider source by value so the bundle loads on an image that has BenchMax
    but no `castform` neon source. `dsn` is a resolved RO DSN string, baked into
    the pickle. `pip_dependencies` overrides `NEON_BUNDLE_PIP_DEPENDENCIES` for
    callers that must pin differently; the captured-module set is not tunable.
    """
    return dump_bundle(
        SearchEnv,
        constructor_args=neon_search_constructor_args(dsn),
        pip_dependencies=(
            NEON_BUNDLE_PIP_DEPENDENCIES
            if pip_dependencies is None
            else pip_dependencies
        ),
        local_modules=neon_bundle_local_modules(),
    )


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
