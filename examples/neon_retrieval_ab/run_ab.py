"""Run the paired offline retrieval A/B against the live Neon corpus.

Sends EVERY query through the production Neon query layer
(:func:`castform.rag.corpus.neon.query.run_query`) once per mode with identical
settings — same corpus, same ``top_k``, no metadata filter, same
``text_search_config`` — so ``mode`` is the only thing that differs between arms.
Nothing here is tuned per mode; there is deliberately no per-mode knob to tune.

The query embedding is computed ONCE per question (batched) and reused by both the
vector and the hybrid arm, so the whole run costs one embedding pass over the 430
questions plus 4 retrieval legs per question (lexical 1, vector 1, hybrid 2).

Each returned chunk is mapped to its source file via ``metadata["file"]`` — the
same field the dataset's ``reference_chunks[].metadata.file`` gold uses — and the
per-chunk list is deduped to a ranked FILE list preserving first-occurrence rank.
Metrics downstream are computed over that ranked file list.

Writes one JSON object per query to ``results/raw_results.jsonl`` plus a
``results/run_manifest.json`` describing the run, so the analysis is re-runnable
without touching Neon. No credential is ever written to either file.

Usage::

    source ~/.config/neon-benchmax.env
    uv run python examples/neon_retrieval_ab/run_ab.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from castform.rag.corpus.embed import DEFAULT_EMBED_MODEL, platform_embed_fn
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA
from castform.rag.corpus.neon.query import NeonQueryRequest, run_query
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_proxy import classify  # noqa: E402

LOGICAL_NAME = "gitlab_handbook_neon"
LLM_URL = "https://llm.castform.dev/v1"

# Identical across all three arms. `mode` is the ONLY per-arm difference.
TOP_K = 10
MODES = ("lexical", "vector", "hybrid")

EMBED_BATCH = 64
DEFAULT_CONCURRENCY = 4

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
DATASETS = (
    _REPO / "examples/neon_rag_smoke/datasets/train_large.jsonl",
    _REPO / "examples/neon_rag_smoke/datasets/eval_large.jsonl",
)
RESULTS_DIR = _HERE / "results"
ENV_FILE = Path.home() / ".config" / "neon-benchmax.env"

_LOCAL = threading.local()


def load_dsn() -> str:
    """Return the read-only corpus DSN, sourcing ``~/.config/neon-benchmax.env`` if needed.

    Only ``NEON_CORPUS_DSN_RO`` is ever read; the read-write DSN is never touched.
    The value is returned, never logged.
    """
    if not os.environ.get("NEON_CORPUS_DSN_RO") and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            match = re.match(r'^(?:export\s+)?([A-Z_]+)="?([^"]*)"?$', line.strip())
            if match and match.group(1) not in os.environ:
                os.environ[match.group(1)] = match.group(2)
    dsn = os.environ.get("NEON_CORPUS_DSN_RO")
    if not dsn:
        raise SystemExit("NEON_CORPUS_DSN_RO not set")
    return dsn


def load_rows() -> list[dict[str, Any]]:
    """Load all A/B rows, attaching provenance, gold file set, and derived style."""
    rows: list[dict[str, Any]] = []
    for path in DATASETS:
        split = path.stem.replace("_large", "")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            gold = sorted(
                {
                    (chunk.get("metadata") or {}).get("file")
                    for chunk in raw.get("reference_chunks") or []
                }
                - {None, ""}
            )
            rows.append(
                {
                    "row_id": len(rows),
                    "split": split,
                    "question": raw["question"],
                    "gold_files": gold,
                    "style": classify(raw["question"]),
                }
            )
    return rows


def embed_queries(questions: list[str], *, base_url: str) -> list[list[float]]:
    """Embed every question ONCE, in batches, reusing the platform embeddings client."""
    embed = platform_embed_fn(base_url=base_url)
    vectors: list[list[float]] = []
    for start in range(0, len(questions), EMBED_BATCH):
        vectors.extend(embed(questions[start : start + EMBED_BATCH]))
    return vectors


def _client(dsn: str) -> NeonClient:
    """Return this thread's ``NeonClient``.

    ``NeonClient`` caches a single psycopg connection and is not thread-safe, so
    each worker owns one. Every query still runs as one single-version snapshot
    read under the module's shared advisory lock (``read_in_snapshot``).
    """
    client = getattr(_LOCAL, "client", None)
    if client is None:
        client = NeonClient(lambda: dsn)
        _LOCAL.client = client
    return client


def retrieve(dsn: str, question: str, vector: list[float] | None, mode: str) -> dict[str, Any]:
    """Run one query in one mode and return its chunk-level and file-level ranking."""
    request = NeonQueryRequest(
        mode=mode,  # type: ignore[arg-type]
        top_k=TOP_K,
        text=question if mode in ("lexical", "hybrid") else None,
        vector=tuple(vector) if mode in ("vector", "hybrid") else None,
        filter=None,
    )
    rows = run_query(
        _client(dsn),
        request,
        logical_name=LOGICAL_NAME,
        schema=CORPUS_SCHEMA,
        text_search_config=DEFAULT_TEXT_SEARCH_CONFIG,
    )
    chunk_files = [(row.metadata or {}).get("file") or row.source_file for row in rows]
    ranked_files: list[str] = []
    for name in chunk_files:
        if name not in ranked_files:
            ranked_files.append(name)
    return {
        "chunk_ids": [row.chunk_id for row in rows],
        "chunk_files": chunk_files,
        "ranked_files": ranked_files,
    }


def run(concurrency: int, base_url: str) -> Path:
    """Execute the full three-arm run and write the raw results + manifest."""
    dsn = load_dsn()
    rows = load_rows()
    started = time.time()

    print(f"embedding {len(rows)} queries once ({DEFAULT_EMBED_MODEL})", flush=True)
    vectors = embed_queries([r["question"] for r in rows], base_url=base_url)
    if len(vectors) != len(rows):
        raise SystemExit(f"embedding count {len(vectors)} != row count {len(rows)}")

    results: dict[int, dict[str, Any]] = {r["row_id"]: {} for r in rows}
    failures: list[dict[str, Any]] = []
    collect = threading.Lock()

    def task(args: tuple[dict[str, Any], list[float], str]) -> None:
        row, vector, mode = args
        try:
            hits = retrieve(dsn, row["question"], vector, mode)
        except Exception as exc:  # recorded, never silently filled in
            with collect:
                failures.append(
                    {
                        "row_id": row["row_id"],
                        "mode": mode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            return
        with collect:
            results[row["row_id"]][mode] = hits

    jobs = [
        (row, vectors[row["row_id"]], mode) for mode in MODES for row in rows
    ]
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in pool.map(task, jobs):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} queries", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "modes": results[row["row_id"]]}) + "\n")

    manifest = {
        "corpus": LOGICAL_NAME,
        "schema": CORPUS_SCHEMA,
        "text_search_config": DEFAULT_TEXT_SEARCH_CONFIG,
        "modes": list(MODES),
        "top_k": TOP_K,
        "filter": None,
        "embed_model": DEFAULT_EMBED_MODEL,
        "embed_base_url": base_url,
        "embed_calls": (len(rows) + EMBED_BATCH - 1) // EMBED_BATCH,
        "rows": len(rows),
        "rows_with_gold": sum(1 for r in rows if r["gold_files"]),
        "retrieval_calls": len(jobs),
        "concurrency": concurrency,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 1),
        "datasets": [str(p.relative_to(_REPO)) for p in DATASETS],
    }
    (RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "failures"}, indent=2))
    if failures:
        print(f"FAILURES: {len(failures)} (see run_manifest.json)")
    return raw_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--base-url", default=LLM_URL)
    args = parser.parse_args()
    run(args.concurrency, args.base_url)


if __name__ == "__main__":
    main()
