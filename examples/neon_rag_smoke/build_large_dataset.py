"""Build a training-sized, validity-filtered QA dataset over the live neon corpus.

Enlarges the 28-row smoke set into a few hundred DISTINCT, high-quality (question,
answer, gold-file) rows for a GPU RL training run against the already-ingested
``gitlab_handbook_neon`` bm25 corpus. Reuses the slice-6/7 qa-generation DI path
(no re-embed, no re-ingest): a single :class:`NeonChunkSource` reads chunks from
the live DB via the RO DSN, two qa-gen passes (keyword + paraphrase) over-generate
raw pairs, then the slice-7 answerability judge (:mod:`qa_validity`, blind of
retrieval) drops only defective pairs. Output rows match the env contract the
``SearchEnv`` unpickles and scores: ``{"question", "answer", "reference_chunks":
[{"metadata": {"file": <real gold source file>}}]}``.

Death-tolerant: each qa-gen pass caches its raw output on disk (``--reuse`` skips
regeneration); the answerability judge caches verdicts to ``verdicts_large.jsonl``
and is not re-spent on resume. A heartbeat log records liveness for a resuming
worker.

Run (root uv venv already has castform[rag,neon])::

    source .venv/bin/activate
    python examples/neon_rag_smoke/build_large_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# --- config --------------------------------------------------------------

LOGICAL_NAME = "gitlab_handbook_neon"
BASE_DOMAIN = "castform.dev"
GENERATOR_MODEL = "gpt-5.4"
JUDGE_MODEL = "gpt-5.4-mini"

CREDS = Path.home() / ".config" / "neon-benchmax.env"
HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "gen"
OUT_DIR = HERE / "datasets"

# Structural-only in-pipeline filter. The answerability judge (qa_validity) is the
# honest generator-quality gate downstream, so its drop rate measures the generator
# rather than being pre-masked by an in-pipeline grounding judge (which would also
# double the judge spend). quality_gate is heuristic (length / dedup / format).
STRUCTURAL_FILTERS = ["quality_gate"]

# Two passes differ only in query STYLE (slice-7 discipline): a keyword pass yields
# lexically-trivial questions, a paraphrase pass yields semantic questions. A broad
# training spread, not a curated eval golden — so no hardness pruning.
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


def load_creds() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in CREDS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def heartbeat(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    with (GEN_DIR / "heartbeat.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


# --- qa-gen passes -------------------------------------------------------


def _pipeline_config(system_prompt: str, n_samples: int, out_dir: Path, seed: int):
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
        filtering=FilteringConfig(filters=list(STRUCTURAL_FILTERS)),
        # No refinement loop: over-generation covers yield; keeps the run bounded.
        refinement=RefinementConfig(enabled=False),
        micro_batch=MicroBatchConfig(batch_size=0, max_parallel_batches=0),
        output=OutputConfig(
            dir=str(out_dir),
            train_jsonl="train_dataset.jsonl",
            eval_jsonl="eval_dataset.jsonl",
        ),
        random_seed=seed,
    )


def run_qa_pass(
    source: Any,
    collection_hashes: set[str],
    *,
    style_dist: dict[str, float],
    system_prompt: str,
    n_samples: int,
    out_dir: Path,
    seed: int,
    reuse: bool,
) -> list[dict[str, Any]]:
    """Run one qa-gen pass forcing ``style_dist``; return raw rows with a real gold.

    Rows keep ``question`` / ``answer`` and the gold source ``file`` pulled from the
    seed chunk's own metadata (the ``reference_chunks[0].id`` hash resolved against
    the in-memory corpus). Only rows whose gold hash is a real corpus chunk with a
    non-empty ``file`` survive.
    """
    from castform.rag.qa_generation.generators import direct_llm
    from castform.rag.qa_generation.neon_entrypoint import neon_llm_url
    from castform.rag.qa_generation.pipeline import run_pipeline
    import castform.rag.qa_generation.pipeline as pipeline_mod

    out_dir.mkdir(parents=True, exist_ok=True)
    cached = any(
        (out_dir / n).exists() and (out_dir / n).stat().st_size > 0
        for n in ("train_dataset.jsonl", "eval_dataset.jsonl")
    )
    if not (reuse and cached):
        cfg = _pipeline_config(system_prompt, n_samples, out_dir, seed)
        cfg.pin_llm_base_url(neon_llm_url(BASE_DOMAIN))
        orig_style = direct_llm.get_style_distribution
        orig_auto = pipeline_mod.auto_tune
        direct_llm.get_style_distribution = lambda _t: dict(style_dist)
        pipeline_mod.auto_tune = lambda *a, **k: {}  # keep the requested style mix
        try:
            heartbeat(f"pass start: n={n_samples} style={style_dist} -> {out_dir.name}")
            run_pipeline(cfg, source_factory=lambda _c: source)
            heartbeat(f"pass done: {out_dir.name}")
        finally:
            direct_llm.get_style_distribution = orig_style
            pipeline_mod.auto_tune = orig_auto
    else:
        heartbeat(f"pass reuse: {out_dir.name} (cached raw on disk)")

    by_hash = {c.hash: c for c in source.collection.chunks}
    rows: list[dict[str, Any]] = []
    for name in ("train_dataset.jsonl", "eval_dataset.jsonl"):
        path = out_dir / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            question = str(raw.get("question") or "").strip()
            answer = str(raw.get("answer") or "").strip()
            refs = raw.get("reference_chunks") or []
            if not question or not answer or not refs:
                continue
            gold_hash = str((refs[0] or {}).get("id", ""))
            if not _HEX64.match(gold_hash) or gold_hash not in collection_hashes:
                continue
            gold_chunk = by_hash[gold_hash]
            gold_file = str(gold_chunk.get_metadata("file") or "").strip()
            if not gold_file:
                continue
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "gold_hash": gold_hash,
                    "gold_file": gold_file,
                }
            )
    return rows


# --- validity filter -----------------------------------------------------


def judge_all(
    rows: list[dict[str, Any]],
    by_hash: dict[str, Any],
    *,
    base_url: str,
    verdicts_path: Path,
) -> list[dict[str, Any]]:
    """Answerability-judge every row (blind of retrieval) with an on-disk cache.

    Reuses the slice-7 :class:`qa_validity.QaValidityJudge` and its verdict-row
    shape. A committed ``verdicts_large.jsonl`` caches per-pair verdicts keyed by
    (question, gold-hash), so a resumed run does not re-spend the judge. Threaded
    (8-way) with a progress heartbeat since one pair = one judge call.
    """
    import sys

    from openai import OpenAI
    from castform.platform.credentials import resolve_judge_key

    # qa_validity ships as an un-packaged module in the sibling slice-7 example;
    # import it by path (same discipline the smoke uses for postgres-search).
    _slice7 = HERE.parent / "gitlab_handbook_bm25_neon"
    if str(_slice7) not in sys.path:
        sys.path.insert(0, str(_slice7))
    from qa_validity import QaValidityJudge  # noqa: E402

    cache: dict[str, dict[str, Any]] = {}
    if verdicts_path.exists():
        for line in verdicts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["_key"]] = r

    judge = QaValidityJudge(base_url=base_url, model=JUDGE_MODEL)
    judge._client = OpenAI(base_url=base_url, api_key=resolve_judge_key("", base_url))

    def key_of(row: dict[str, Any]) -> str:
        return normalize_question(row["question"]) + "\x00" + row["gold_hash"]

    todo = [r for r in rows if key_of(r) not in cache]
    heartbeat(f"validity judge: {len(todo)} to judge, {len(cache)} cached")

    done = 0
    lock_out: list[dict[str, Any]] = []

    def judge_one(row: dict[str, Any]) -> dict[str, Any]:
        gold = by_hash[row["gold_hash"]]
        v = judge.judge(row["question"], [gold])
        return {
            "_key": key_of(row),
            "question": row["question"],
            "answer": row["answer"],
            "gold_hash": row["gold_hash"],
            "gold_file": row["gold_file"],
            "verdict": v.verdict,
            "reason": v.reason,
            "citation": v.citation,
            "confidence": v.confidence,
            "gold_preview": gold.content[:280],
        }

    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(judge_one, r): r for r in todo}
            for fut in as_completed(futs):
                res = fut.result()
                cache[res["_key"]] = res
                lock_out.append(res)
                done += 1
                if done % 25 == 0:
                    heartbeat(f"  judged {done}/{len(todo)}")
                    _flush_verdicts(verdicts_path, cache)
        _flush_verdicts(verdicts_path, cache)

    # Return verdict rows in input order.
    return [cache[key_of(r)] for r in rows]


def _flush_verdicts(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in cache.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --- main ----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyword-samples", type=int, default=340)
    p.add_argument("--paraphrase-samples", type=int, default=320)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse", action="store_true", help="reuse cached raw qa-gen output")
    p.add_argument("--defect-cap", type=float, default=0.35)
    args = p.parse_args()

    creds = load_creds()
    dsn = creds["NEON_CORPUS_DSN_RO"]
    os.environ["NEON_CORPUS_DSN_RO"] = dsn
    os.environ.setdefault("CASTFORM_API_KEY", creds["CASTFORM_API_KEY"])
    os.environ.setdefault("PLATFORM_API_KEY", creds["PLATFORM_API_KEY"])

    from castform.rag.chunkers.models import ChunkCollection
    from castform.rag.corpus.neon.source import NeonChunkSource
    from castform.rag.qa_generation.neon_entrypoint import neon_llm_url

    llm_url = neon_llm_url(BASE_DOMAIN)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # One source over the LIVE corpus. Materialize the in-memory collection once
    # (RO snapshot scan, no re-embed) so both passes reuse it and hashes align.
    source = NeonChunkSource(LOGICAL_NAME, embed_fn=None, read_dsn_provider=dsn)
    heartbeat("scanning live corpus into memory (no re-embed)...")
    collection = ChunkCollection(list(source.scan_chunks()))
    source.collection = collection
    collection_hashes = {c.hash for c in collection.chunks}
    by_hash = {c.hash: c for c in collection.chunks}
    heartbeat(f"corpus in memory: {len(collection_hashes)} chunks")

    keyword = run_qa_pass(
        source,
        collection_hashes,
        style_dist=_STYLE_KEYWORD,
        system_prompt=_LEXICAL_SYSTEM_PROMPT,
        n_samples=args.keyword_samples,
        out_dir=GEN_DIR / "qa_keyword",
        seed=args.seed,
        reuse=args.reuse,
    )
    paraphrase = run_qa_pass(
        source,
        collection_hashes,
        style_dist=_STYLE_PARAPHRASE,
        system_prompt=_VECTOR_SYSTEM_PROMPT,
        n_samples=args.paraphrase_samples,
        out_dir=GEN_DIR / "qa_paraphrase",
        seed=args.seed + 1,
        reuse=args.reuse,
    )
    raw = keyword + paraphrase
    heartbeat(f"raw rows with real gold: {len(raw)} "
              f"(keyword={len(keyword)}, paraphrase={len(paraphrase)})")

    # Dedup by normalized question BEFORE judging (don't re-spend on dupes).
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in raw:
        nq = normalize_question(r["question"])
        if nq in seen:
            continue
        seen.add(nq)
        deduped.append(r)
    heartbeat(f"distinct questions (pre-judge): {len(deduped)} "
              f"(dropped {len(raw) - len(deduped)} exact/near dupes)")

    verdicts_path = OUT_DIR / "verdicts_large.jsonl"
    verdict_rows = judge_all(deduped, by_hash, base_url=llm_url, verdicts_path=verdicts_path)

    kept = [v for v in verdict_rows if v["verdict"] != "drop"]
    dropped = [v for v in verdict_rows if v["verdict"] == "drop"]
    total = len(verdict_rows)
    defect_rate = len(dropped) / max(total, 1)
    heartbeat(
        f"validity: generated={total} kept={len(kept)} dropped={len(dropped)} "
        f"defect_rate={defect_rate:.3f}"
    )

    if defect_rate > args.defect_cap:
        heartbeat(
            f"STOP: defect_rate {defect_rate:.3f} > cap {args.defect_cap}. "
            "Generator indicted; not shipping a defective training set."
        )
        _write_summary(args, keyword, paraphrase, deduped, kept, dropped, defect_rate,
                       by_hash, stopped=True)
        return 2

    # Emit env rows. Dedup once more by question (kept order is judge-completion
    # order; make it deterministic by sorting on question for a stable split).
    kept_sorted = sorted(kept, key=lambda r: r["question"])
    final_seen: set[str] = set()
    env_rows: list[dict[str, Any]] = []
    for r in kept_sorted:
        nq = normalize_question(r["question"])
        if nq in final_seen:
            continue
        final_seen.add(nq)
        gold_file = r["gold_file"].strip()
        if not gold_file:
            continue  # guarded already, but never emit an empty gold
        env_rows.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "reference_chunks": [{"metadata": {"file": gold_file}}],
            }
        )

    need = args.n_train + args.n_eval
    if len(env_rows) < need:
        heartbeat(
            f"WARNING: only {len(env_rows)} distinct kept rows; wanted {need}. "
            "Emitting all; coordinator should decide whether to top up generation."
        )

    # Deterministic shuffle then split so train/eval don't share a question.
    import random

    rng = random.Random(args.seed)
    rng.shuffle(env_rows)
    n_eval = min(args.n_eval, max(0, len(env_rows) - 1))
    n_train = min(args.n_train, len(env_rows) - n_eval)
    eval_rows = env_rows[:n_eval]
    train_rows = env_rows[n_eval : n_eval + n_train]

    train_path = OUT_DIR / "train_large.jsonl"
    eval_path = OUT_DIR / "eval_large.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    _assert_dataset(train_rows, eval_rows)
    heartbeat(
        f"WROTE train={len(train_rows)} -> {train_path} | eval={len(eval_rows)} -> {eval_path}"
    )
    _write_summary(args, keyword, paraphrase, deduped, kept, dropped, defect_rate,
                   by_hash, stopped=False, train=train_rows, eval_=eval_rows)
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _assert_dataset(train: list[dict], eval_: list[dict]) -> None:
    tq = [normalize_question(r["question"]) for r in train]
    eq = [normalize_question(r["question"]) for r in eval_]
    assert len(tq) == len(set(tq)), "duplicate questions in train"
    assert len(eq) == len(set(eq)), "duplicate questions in eval"
    assert not (set(tq) & set(eq)), "train/eval question overlap"
    for r in train + eval_:
        f = r["reference_chunks"][0]["metadata"]["file"]
        assert isinstance(f, str) and f.strip(), "empty gold file"


def _section(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else path


def _write_summary(args, keyword, paraphrase, deduped, kept, dropped, defect_rate,
                   by_hash, *, stopped: bool, train=None, eval_=None) -> None:
    from collections import Counter

    sections = Counter(_section(r["gold_file"]) for r in kept)
    drop_reasons = Counter(d["reason"] for d in dropped)
    summary = {
        "stopped": stopped,
        "keyword_samples": args.keyword_samples,
        "paraphrase_samples": args.paraphrase_samples,
        "raw_keyword_rows": len(keyword),
        "raw_paraphrase_rows": len(paraphrase),
        "distinct_pre_judge": len(deduped),
        "generated_judged": len(kept) + len(dropped),
        "kept": len(kept),
        "dropped": len(dropped),
        "defect_rate": round(defect_rate, 4),
        "defect_cap": args.defect_cap,
        "drop_reasons": dict(drop_reasons),
        "kept_section_coverage": dict(sections.most_common()),
        "train_rows": len(train) if train is not None else 0,
        "eval_rows": len(eval_) if eval_ is not None else 0,
    }
    (GEN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    heartbeat("summary: " + json.dumps(summary))


if __name__ == "__main__":
    raise SystemExit(main())
