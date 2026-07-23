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
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.credentials import resolve_read_dsn_provider
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG, NeonTableSpec
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
from equivalence import build_equivalence_sets
from equivalence import params as equivalence_params
from multi_gold import MultiGoldExpander
from qa_validity import NATURAL_CAPABILITIES, filter_records, load_verdict_cache
from handbook_corpus import (
    HANDBOOK_COMMIT,
    HANDBOOK_REPO_URL,
    HANDBOOK_SUBDIR,
    LOGICAL_NAME,
    ChunkerParams,
    build_collection,
    git_tracked_docs,
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
    "handbook. PARAPHRASE the wording — prefer everyday synonyms over the handbook's "
    "phrasing — BUT you MUST keep exactly one disambiguating ANCHOR from the source "
    "verbatim: a proper noun, a person or team name, a product, tool, policy, or "
    "config-key name, or a specific number, so the question points at THIS one chunk "
    "and not a dozen similar ones. NEVER replace that anchor with a generic "
    "placeholder like 'this contact', 'a certain tool', or 'the team'. Keep the "
    "specific concept, entity, quantity, or step intact; do not make it vague. Write "
    "a concise answer under 60 words with no preamble."
)

_HANDBOOK_DESCRIPTION = (
    "The GitLab handbook is a public company handbook covering engineering, "
    "product, security, sales, marketing, finance, people operations, legal, and "
    "internal ways of working."
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"[a-z]{4,}")
_STOP = frozenset(
    {
        "what",
        "when",
        "which",
        "where",
        "does",
        "the",
        "and",
        "for",
        "how",
        "that",
        "with",
        "from",
        "this",
        "into",
        "about",
        "gitlab",
        "handbook",
    }
)

# Local lexical-hardness ceiling: a vector question is kept only if at most this
# fraction of its content words also appear in its gold chunk (blind of Neon). Loose
# on purpose: the anchored prompt deliberately shares one distinctive token with the
# gold, so a tight ceiling would pressure the model to strip the anchor (the exact
# defect the fix-round removes). This only rejects near-verbatim copies; the real
# not-keyword-solvable check is the live BM25-ablation margin in the gate.
_VECTOR_MAX_OVERLAP = 0.75


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
        corpus=CorpusConfig(
            corpus_name=LOGICAL_NAME, corpus_id="", min_chunk_chars=400
        ),
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
    reuse: bool = False,
) -> list[NeonEvalRecord]:
    """Run one qa-gen pass forcing ``style_dist`` and map rows to eval records.

    ``max_overlap`` (vector pass only) drops questions whose content words overlap
    the gold chunk above the ceiling — a LOCAL, retrieval-free hardness filter.
    ``reuse`` reads the prior pass's raw output instead of regenerating (so curated
    / metric iterations do not re-spend on qa-gen).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = any(
        (out_dir / n).exists() and (out_dir / n).stat().st_size > 0
        for n in ("train_dataset.jsonl", "eval_dataset.jsonl")
    )
    if not (reuse and cached):
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
            if _HEX64.match(str(rc.get("id", "")))
            and str(rc.get("id")) in collection_hashes
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


def _current_spec() -> NeonTableSpec:
    """Resolve the active corpus version to a spec (for equivalence neighbour reads)."""
    ro = NeonClient(resolve_read_dsn_provider(None))
    rows = ro.execute(ro.read_ledger_sql(), {"logical": LOGICAL_NAME})
    for version, _state, is_current in rows:
        if is_current:
            return NeonTableSpec(
                LOGICAL_NAME, version, text_search_config=DEFAULT_TEXT_SEARCH_CONFIG
            )
    raise LookupError(f"no current published version for {LOGICAL_NAME!r}")


def _expand_gold(
    records: list[NeonEvalRecord], mapping: dict[str, list[str]]
) -> list[NeonEvalRecord]:
    out = []
    for r in records:
        gold: list[str] = []
        for h in r.gold_chunk_hashes:
            gold.extend(mapping.get(h, [h]))
        out.append(
            r.model_copy(update={"gold_chunk_hashes": list(dict.fromkeys(gold))})
        )
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
    reuse_qa: bool = False,
    verdicts_path: Path | None = None,
    build_timestamp: str,
) -> dict[str, Any]:
    """Author the full cleaned golden set and return the provenance manifest.

    Pipeline: generate keyword (lexical) + anchored-paraphrase (vector) qa rows and
    the curated filter/hybrid probes; screen the NATURAL-LANGUAGE rows through the
    blind answerability filter (:mod:`qa_validity`), dropping only defective pairs
    with a recorded reason; then expand every kept row's gold with judge-confirmed
    same-file supporting chunks (:mod:`multi_gold`) and templated near-duplicates
    (:mod:`equivalence`). The frozen JSONL + manifest record the full drop audit and
    all authoring parameters.
    """
    llm_url = neon_llm_url(base_domain)
    params = ChunkerParams()
    docs_dir = sparse_checkout(work_dir, commit=HANDBOOK_COMMIT)
    tracked = git_tracked_docs(work_dir, HANDBOOK_SUBDIR, params.file_extensions)
    collection = build_collection(docs_dir, params=params, files=tracked)
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
        reuse=reuse_qa,
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
        reuse=reuse_qa,
    )
    curated = build_curated_rows(collection, n_filter=n_filter, n_hybrid=n_hybrid)

    # R1: drop DEFECTIVE natural-language pairs (blind of retrieval); curated probes
    # are answerable-by-construction and exempt (see qa_validity). Verdicts are
    # reused from ``verdicts_path`` when supplied so the frozen-set drops reconcile
    # exactly with the standalone audit and the judge is not re-spent.
    natural = lexical + vector
    cache = load_verdict_cache(verdicts_path) if verdicts_path else None
    kept_natural, dropped, vjudge, verdicts = filter_records(
        natural, collection, base_url=llm_url, cache=cache
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "verdicts_v2.jsonl").open("w", encoding="utf-8") as vf:
        for row in verdicts:
            vf.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = kept_natural + curated

    # Expand gold: judge-confirmed same-file chunks, then templated near-duplicates.
    expander = MultiGoldExpander(collection, base_url=llm_url)
    rows = [
        r.model_copy(
            update={"gold_chunk_hashes": expander.expand(r.query, r.gold_chunk_hashes)}
        )
        for r in rows
    ]
    all_gold = sorted({h for r in rows for h in r.gold_chunk_hashes})
    equiv = build_equivalence_sets(
        NeonClient(resolve_read_dsn_provider(None)), _current_spec(), all_gold
    )
    rows = _expand_gold(rows, equiv)

    records = _dedup(rows)
    records.sort(key=lambda r: (r.search_mode, r.query))

    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = out_dir / "gitlab_handbook_neon_golden.jsonl"
    with frozen.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    mode_counts: dict[str, int] = {}
    cap_counts: dict[str, int] = {}
    for r in records:
        mode_counts[r.search_mode] = mode_counts.get(r.search_mode, 0) + 1
        cap_counts[r.capability] = cap_counts.get(r.capability, 0) + 1
    dropped_by_reason: dict[str, int] = {}
    for d in dropped:
        dropped_by_reason[d["reason"]] = dropped_by_reason.get(d["reason"], 0) + 1
    verdict_tally: dict[str, int] = {}
    for v in verdicts:
        if v["natural"]:
            verdict_tally[v["verdict"]] = verdict_tally.get(v["verdict"], 0) + 1

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
        "models": {
            "generator_requested": GENERATOR_MODEL,
            "judge_requested": JUDGE_MODEL,
            "judge_resolved": vjudge.resolved_model,
        },
        "authoring_params": {
            "lexical_samples": lexical_samples,
            "vector_samples": vector_samples,
            "n_filter": n_filter,
            "n_hybrid": n_hybrid,
            "seed": seed,
            "blind_filters": list(BLIND_FILTERS),
            "vector_max_lexical_overlap": _VECTOR_MAX_OVERLAP,
            "vector_prompt_keeps_anchor": True,
        },
        "validity_filter": {
            "scope_capabilities": sorted(NATURAL_CAPABILITIES),
            "natural_generated": len(natural),
            "natural_kept": len(kept_natural),
            "dropped": len(dropped),
            "defective_rate": round(len(dropped) / max(len(natural), 1), 4),
            "verdict_tally": verdict_tally,
            "dropped_by_reason": dropped_by_reason,
            "dropped_sample": dropped[:25],
            "verdicts_file": "verdicts_v2.jsonl",
            "note": (
                "curated single-token/bag-of-words filter+hybrid probes are "
                "answerable-by-construction and exempt from the answerability judge"
            ),
        },
        "multi_gold": {
            "method": "judge-confirmed same-file supporting chunks (blind)",
            "judge_model": JUDGE_MODEL,
        },
        "equivalence_set": equivalence_params(),
        "gold_per_row_mean": round(
            sum(len(r.gold_chunk_hashes) for r in records) / max(len(records), 1), 3
        ),
        "decoy_row_count": sum(1 for r in records if r.decoy_chunk_hashes),
        "decoy_total": sum(len(r.decoy_chunk_hashes) for r in records),
        "row_counts_by_mode": mode_counts,
        "row_counts_by_capability": cap_counts,
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
        "--reuse-qa",
        action="store_true",
        help="reuse the prior qa-gen output instead of regenerating (no re-spend)",
    )
    p.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="reuse a verdicts_v2.jsonl audit as the validity-judge cache (no re-spend)",
    )
    p.add_argument(
        "--build-timestamp",
        required=True,
        help="iso-8601 build timestamp recorded in the manifest",
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
        reuse_qa=args.reuse_qa,
        verdicts_path=args.verdicts,
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
