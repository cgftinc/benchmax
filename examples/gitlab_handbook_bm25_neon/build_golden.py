"""Author the BLIND all-capability golden retrieval eval for the Neon corpus.

Reuses the SAME re-chunked handbook corpus the golden is validated against, but
produces a NEW, different dataset (not the lexical-only BM25 golden). Two halves:

* **qa-gen (LEXICAL + VECTOR rows)** — the Slice-6 DI seam runs the real qa-gen
  pipeline against the shared corpus. Two passes differ only in query STYLE: a
  keyword pass (obfuscation off) yields lexically-trivial questions (the
  ``lexical`` rows), a paraphrase pass (natural/expert style, obfuscation on)
  yields semantic questions BM25 should miss (the ``vector`` rows). Gold ids are
  the EXACT source-chunk hashes qa-gen already carries in ``reference_chunks[].id``
  — never re-derived from a retrieval result. The retrieval-coupled
  ``retrieval_too_easy_llm`` filter is dropped so gold stays independent of what
  Neon returns (F12/B8).
* **hand-curated (FILTER + HYBRID rows)** — see :mod:`curated_rows`.

Non-circularity guards baked in: gold is authored before any Neon retrieval runs;
gold ids are exact chunk hashes verified to exist in the in-memory collection;
vector questions pass a LOCAL lexical-hardness filter (low question/chunk token
overlap) so they are not keyword-solvable — the live gate then CONFIRMS this via a
Neon BM25 ablation. Output is a FROZEN JSONL committed to the repo and never
regenerated in CI, plus a provenance manifest (NB4).

Run (after ingest, from the workspace root, creds sourced)::

    uv run --extra neon python examples/gitlab_handbook_bm25_neon/build_golden.py \\
        --work-dir /tmp/handbook_repo --out-dir examples/gitlab_handbook_bm25_neon/datasets
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.embed import DEFAULT_EMBED_MODEL, platform_embed_fn
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord
from castform.rag.corpus.neon.source import NeonChunkSource
from castform.rag.qa_generation.generators import direct_llm
from castform.rag.qa_generation.neon_entrypoint import neon_llm_url
from castform.rag.qa_generation.pipeline import auto_tune, run_pipeline
from castform.rag.qa_generation.pipeline_config import (
    CorpusConfig,
    CorpusContextConfig,
    FilteringConfig,
    GenerationConfig,
    LLMDirectGenerationConfig,
    MicroBatchConfig,
    OutputConfig,
    PipelineConfig,
    PlatformConfig,
    RefinementConfig,
    TargetsConfig,
)

from curated_rows import build_curated_rows
from multi_gold import MultiGoldExpander
from handbook_corpus import (
    HANDBOOK_COMMIT,
    HANDBOOK_REPO_URL,
    HANDBOOK_SUBDIR,
    LOGICAL_NAME,
    ChunkerParams,
    build_collection,
    sparse_checkout,
)

# Generator / judge model ids (qa-gen defaults, recorded in the manifest).
GENERATOR_MODEL = "gpt-5.4"
JUDGE_MODEL = "gpt-5.4-mini"

# BLIND authoring: drop retrieval_too_easy_llm (the only Neon-retrieval-coupled
# filter). The rest are judge-LLM/heuristic only.
BLIND_FILTERS = ["quality_gate", "grounding_llm", "hop_count_validity"]

# All-keyword vs all-paraphrase style, forcing the two qa-gen passes.
_STYLE_KEYWORD = {"keyword": 1.0, "natural": 0.0, "expert": 0.0}
_STYLE_PARAPHRASE = {"keyword": 0.0, "natural": 0.6, "expert": 0.4}

_LEXICAL_SYSTEM_PROMPT = (
    "You are an expert QA dataset author for a retrieval benchmark. Write a "
    "single-focus question that asks for one specific fact from the GitLab "
    "handbook, reusing the handbook's own distinctive terminology (page title, "
    "team, tool, policy, or product name) so the source chunk is easy to locate. "
    "Write a concise answer under 60 words with no preamble."
)
_VECTOR_SYSTEM_PROMPT = (
    "You are an expert QA dataset author for a semantic-retrieval benchmark. Write "
    "a single-focus question that asks for one specific fact from the GitLab "
    "handbook. PARAPHRASE it: replace the handbook's distinctive surface words with "
    "everyday synonyms, but KEEP the specific concept, entity, quantity, or step "
    "the fact is about, so the question stays clearly about the same thing and a "
    "semantic search can still find the source while a plain keyword search cannot. "
    "Do not make it vague or generic. Write a concise answer under 60 words with no "
    "preamble."
)

_HANDBOOK_DESCRIPTION = (
    "The GitLab handbook is a public company handbook covering engineering, "
    "product, security, sales, marketing, finance, people operations, legal, and "
    "internal ways of working."
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"[a-z]{4,}")
_STOP = frozenset(
    {"what", "when", "which", "where", "does", "the", "and", "for", "how", "that",
     "with", "from", "this", "into", "about", "gitlab", "handbook"}
)

# Local lexical-hardness ceiling: a vector question is kept only if at most this
# fraction of its content words also appear in its gold chunk (blind of Neon).
# Moderate (not aggressive) — enough to stay not-keyword-solvable while leaving the
# question semantically recoverable by the vector leg.
_VECTOR_MAX_OVERLAP = 0.55


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def _lexical_overlap(question: str, gold_content: str) -> float:
    q = _content_words(question)
    if not q:
        return 1.0
    return len(q & _content_words(gold_content)) / len(q)


def bind_source(collection: ChunkCollection, base_domain: str) -> NeonChunkSource:
    """Bind a Neon source to the active corpus with the in-memory collection set.

    Setting ``collection`` guarantees qa-gen samples the same chunks whose hashes
    were ingested (hash alignment) and skips the pipeline's materialization path.
    The ``embed_fn`` is pinned to the platform embeddings host.
    """
    source = NeonChunkSource(
        LOGICAL_NAME, embed_fn=platform_embed_fn(base_url=neon_llm_url(base_domain))
    )
    source.collection = collection
    return source


def _pipeline_config(
    *, system_prompt: str, n_samples: int, out_dir: Path, seed: int
) -> PipelineConfig:
    return PipelineConfig(
        platform=PlatformConfig(),
        corpus=CorpusConfig(corpus_name=LOGICAL_NAME, corpus_id="", min_chunk_chars=400),
        corpus_context=CorpusContextConfig(
            enabled=False,
            description=_HANDBOOK_DESCRIPTION,
            generate_entity_patterns=False,
        ),
        targets=TargetsConfig(
            total_samples=n_samples,
            primary_type_distribution={"lookup": 1.0, "multi_hop": 0.0},
            reasoning_mode_distribution={
                "factual": 1.0,
                "temporal": 0.0,
                "inference": 0.0,
                "sequential": 0.0,
            },
            hop_distribution={1: 1.0, 2: 0.0},
        ),
        generation=GenerationConfig(
            llm_direct=LLMDirectGenerationConfig(
                system_prompt=system_prompt, max_concurrent=8
            )
        ),
        filtering=FilteringConfig(filters=list(BLIND_FILTERS)),
        refinement=RefinementConfig(),
        micro_batch=MicroBatchConfig(batch_size=0, max_parallel_batches=0),
        output=OutputConfig(
            dir=str(out_dir),
            train_jsonl="train_dataset.jsonl",
            eval_jsonl="eval_dataset.jsonl",
        ),
        random_seed=seed,
    )


def _read_qa_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("train_dataset.jsonl", "eval_dataset.jsonl"):
        path = out_dir / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_qa_pass(
    source: NeonChunkSource,
    collection_hashes: set[str],
    *,
    search_mode: str,
    style_dist: dict[str, float],
    system_prompt: str,
    n_samples: int,
    out_dir: Path,
    base_domain: str,
    seed: int,
    max_overlap: float | None,
) -> list[NeonEvalRecord]:
    """Run one qa-gen pass forcing ``style_dist`` and map rows to eval records.

    ``max_overlap`` (vector pass only) drops questions whose content words overlap
    the gold chunk above the ceiling — a LOCAL, retrieval-free hardness filter.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _pipeline_config(
        system_prompt=system_prompt, n_samples=n_samples, out_dir=out_dir, seed=seed
    )
    cfg.pin_llm_base_url(neon_llm_url(base_domain))

    orig_style = direct_llm.get_style_distribution
    orig_auto = auto_tune
    direct_llm.get_style_distribution = lambda _qa_type: dict(style_dist)
    import castform.rag.qa_generation.pipeline as _pipeline_mod

    _pipeline_mod.auto_tune = lambda *a, **k: {}  # keep the requested style mix
    try:
        run_pipeline(cfg, source_factory=lambda _c: source)
    finally:
        direct_llm.get_style_distribution = orig_style
        _pipeline_mod.auto_tune = orig_auto

    records: list[NeonEvalRecord] = []
    for row in _read_qa_rows(out_dir):
        question = str(row.get("question") or row.get("prompt") or "").strip()
        refs = row.get("reference_chunks") or []
        gold = [
            str(rc.get("id"))
            for rc in refs
            if _HEX64.match(str(rc.get("id", ""))) and str(rc.get("id")) in collection_hashes
        ]
        if not question or not gold:
            continue
        if max_overlap is not None:
            gold_content = "\n".join(str(rc.get("content") or "") for rc in refs)
            if _lexical_overlap(question, gold_content) > max_overlap:
                continue
        records.append(
            NeonEvalRecord(
                capability=f"{search_mode}_lookup",
                search_mode=search_mode,  # type: ignore[arg-type]
                query=question,
                gold_chunk_hashes=gold,
            )
        )
    return records


def _dedup(records: list[NeonEvalRecord]) -> list[NeonEvalRecord]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out: list[NeonEvalRecord] = []
    for r in records:
        key = (r.query, tuple(sorted(r.gold_chunk_hashes)))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build_golden(
    *,
    work_dir: Path,
    out_dir: Path,
    base_domain: str = "castform.dev",
    lexical_samples: int = 60,
    vector_samples: int = 90,
    n_filter: int = 10,
    n_hybrid: int = 8,
    seed: int = 42,
    build_timestamp: str,
) -> dict[str, Any]:
    """Author the full golden set and return the provenance manifest."""
    params = ChunkerParams()
    docs_dir = sparse_checkout(work_dir, commit=HANDBOOK_COMMIT)
    collection = build_collection(docs_dir, params=params)
    collection_hashes = {c.hash for c in collection}
    source = bind_source(collection, base_domain)

    lexical = run_qa_pass(
        source,
        collection_hashes,
        search_mode="lexical",
        style_dist=_STYLE_KEYWORD,
        system_prompt=_LEXICAL_SYSTEM_PROMPT,
        n_samples=lexical_samples,
        out_dir=out_dir / "qa_lexical",
        base_domain=base_domain,
        seed=seed,
        max_overlap=None,
    )
    vector = run_qa_pass(
        source,
        collection_hashes,
        search_mode="vector",
        style_dist=_STYLE_PARAPHRASE,
        system_prompt=_VECTOR_SYSTEM_PROMPT,
        n_samples=vector_samples,
        out_dir=out_dir / "qa_vector",
        base_domain=base_domain,
        seed=seed + 1,
        max_overlap=_VECTOR_MAX_OVERLAP,
    )
    curated = build_curated_rows(collection, n_filter=n_filter, n_hybrid=n_hybrid)

    # Judge-confirmed multi-gold: credit same-file chunks that also answer the
    # question (blind of Neon retrieval), so an equally-correct chunk is not a miss.
    expander = MultiGoldExpander(collection, base_url=neon_llm_url(base_domain))
    expanded = [
        r.model_copy(
            update={"gold_chunk_hashes": expander.expand(r.query, r.gold_chunk_hashes)}
        )
        for r in lexical + vector + curated
    ]

    records = _dedup(expanded)
    # Deterministic on-disk order: by mode then query.
    records.sort(key=lambda r: (r.search_mode, r.query))

    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = out_dir / "gitlab_handbook_neon_golden.jsonl"
    with frozen.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    counts: dict[str, int] = {}
    for r in records:
        counts[r.search_mode] = counts.get(r.search_mode, 0) + 1
    counts["curated_filter_hybrid"] = len(curated)

    return {
        "dataset": frozen.name,
        "build_timestamp": build_timestamp,
        "corpus_source": {
            "repo_url": HANDBOOK_REPO_URL,
            "subdir": HANDBOOK_SUBDIR,
            "commit": HANDBOOK_COMMIT,
        },
        "logical_name": LOGICAL_NAME,
        "chunker": params.as_dict(),
        "chunk_count": len(collection_hashes),
        "embedder": {"model": DEFAULT_EMBED_MODEL, "dim": 3072},
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "blind_filters": list(BLIND_FILTERS),
        "vector_max_lexical_overlap": _VECTOR_MAX_OVERLAP,
        "multi_gold": {
            "method": "judge-confirmed same-file supporting chunks (blind)",
            "judge_model": JUDGE_MODEL,
        },
        "gold_per_row_mean": round(
            sum(len(r.gold_chunk_hashes) for r in records) / len(records), 3
        ),
        "row_counts": counts,
        "total_rows": len(records),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--base-domain", default="castform.dev")
    p.add_argument("--lexical-samples", type=int, default=60)
    p.add_argument("--vector-samples", type=int, default=90)
    p.add_argument("--n-filter", type=int, default=10)
    p.add_argument("--n-hybrid", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--build-timestamp",
        required=True,
        help="ISO-8601 build timestamp recorded in the manifest",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_golden(
        work_dir=args.work_dir,
        out_dir=args.out_dir,
        base_domain=args.base_domain,
        lexical_samples=args.lexical_samples,
        vector_samples=args.vector_samples,
        n_filter=args.n_filter,
        n_hybrid=args.n_hybrid,
        seed=args.seed,
        build_timestamp=args.build_timestamp,
    )
    manifest_path = args.out_dir / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
