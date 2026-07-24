"""Build train/eval datasets for the neon RAG gitlab smoke, in SearchEnv's row
contract (question / answer / reference_chunks).

Source rows are the natural, keep/fix verdicts in
``examples/gitlab_handbook_bm25_neon/datasets/verdicts_v2.jsonl``. Each verdict
names its gold chunk by content hash (``gold_chunk_hashes``); we resolve that hash
to the chunk's real ``source_file`` via a read-only lookup against the live
``gitlab_handbook_neon`` corpus view, so every emitted row carries a REAL gold
``file`` and the UNGATED retrieval_hit reward term can score. The verdict ``query``
becomes the question and its ``citation`` the reference answer.

Run inside the neon_rag_smoke uv env (needs the ``neon`` extra for psycopg):
    uv run --project examples/neon_rag_smoke python examples/neon_rag_smoke/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from castform.rag.corpus.neon.provision import CORPUS_SCHEMA
from castform.rag.corpus.neon.schema import view_name

CREDS = Path.home() / ".config" / "neon-benchmax.env"
VERDICTS = (
    Path(__file__).resolve().parents[1]
    / "gitlab_handbook_bm25_neon"
    / "datasets"
    / "verdicts_v2.jsonl"
)
OUT_DIR = Path(__file__).resolve().parent / "datasets"
CORPUS_TABLE = "gitlab_handbook_neon"

N_TRAIN = 28
N_EVAL = 4


def load_creds() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in CREDS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def read_verdicts() -> list[dict]:
    """Natural keep/fix verdicts with a gold chunk hash and a citation answer."""
    rows: list[dict] = []
    seen_hashes: set[str] = set()
    for line in VERDICTS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        if not v.get("natural"):
            continue
        if v.get("verdict") not in ("keep", "fix"):
            continue
        hashes = v.get("gold_chunk_hashes") or []
        query = (v.get("query") or "").strip()
        citation = (v.get("citation") or "").strip()
        if not hashes or not query or not citation:
            continue
        h = hashes[0]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        rows.append({"query": query, "citation": citation, "hash": h})
    return rows


def resolve_source_files(dsn: str, hashes: list[str]) -> dict[str, str]:
    """Map each chunk hash (the view's ``id``) to its ``source_file``, RO."""
    import psycopg
    from psycopg import sql

    view = sql.Identifier(CORPUS_SCHEMA, view_name(CORPUS_TABLE))
    out: dict[str, str] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id, source_file FROM {} WHERE id = ANY(%s)").format(view),
                (hashes,),
            )
            for chunk_id, source_file in cur.fetchall():
                if source_file:
                    out[chunk_id] = source_file
    return out


def main() -> None:
    dsn = load_creds()["NEON_CORPUS_DSN_RO"]
    verdicts = read_verdicts()
    file_by_hash = resolve_source_files(dsn, [v["hash"] for v in verdicts])

    rows: list[dict] = []
    for v in verdicts:
        source_file = file_by_hash.get(v["hash"])
        if not source_file:
            continue  # gold chunk not present in the active corpus version; skip
        rows.append(
            {
                "question": v["query"],
                "answer": v["citation"],
                "reference_chunks": [{"metadata": {"file": source_file}}],
            }
        )

    need = N_TRAIN + N_EVAL
    if len(rows) < need:
        raise SystemExit(
            f"only {len(rows)} rows have a resolvable gold file; need {need}. "
            "Corpus/verdicts drift — inspect before shipping."
        )
    rows = rows[:need]
    train, eval_ = rows[:N_TRAIN], rows[N_TRAIN:need]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("eval.jsonl", eval_)):
        path = OUT_DIR / name
        with path.open("w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(split):3d} rows -> {path}")

    print("\nsample train row:")
    print(json.dumps(train[0], indent=2))


if __name__ == "__main__":
    main()
