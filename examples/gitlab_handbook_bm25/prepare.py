#!/usr/bin/env -S uv run --isolated --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "castform[rag] @ git+https://github.com/castform-ai/benchmax.git@c19b4addb767a745bc8f75e7167afd3958d4dfa3#subdirectory=packages/castform",
# ]
# ///
"""Prepare the one corpus and one dataset shared by both A/B arms."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from castform import config
from castform.rag.corpus.postgres.client import CorpusClient
from castform.rag.corpus.postgres.source import PostgresChunkSource

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "experiment.json"
SHARED = ROOT / "shared"
MANIFEST_PATH = SHARED / "manifest.json"


def _spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def _find_corpus(name: str) -> dict[str, Any] | None:
    async with CorpusClient(base_url=config.platform_url()) as client:
        # The public Corpus dataclass currently omits chunkCount, but the A/B
        # guard needs it. Read the same authenticated endpoint without dropping
        # fields while keeping credential resolution inside CorpusClient.
        response = await client._request("GET", "/v1/corpora")
        client._handle_response_errors(response)
        matches = [row for row in response.json() if row.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"multiple corpora named {name!r}; refusing an ambiguous A/B input")
    if not matches:
        return None
    row = matches[0]
    return {
        "id": row["id"],
        "name": row["name"],
        "chunk_count": int(row["chunkCount"]),
    }


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)


def _checkout_source(corpus_spec: dict[str, Any]) -> Path:
    source_root = ROOT / "artifacts" / "source"
    if source_root.exists():
        raise RuntimeError(
            f"{source_root} already exists; move it aside before an explicit re-ingest"
        )
    source_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            corpus_spec["source_repo"],
            str(source_root),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", corpus_spec["source_subdir"]],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", corpus_spec["source_commit"]],
        cwd=source_root,
        check=True,
    )
    return source_root / corpus_spec["source_subdir"]


async def _ingest(corpus_spec: dict[str, Any]) -> Any:
    existing = await _find_corpus(corpus_spec["name"])
    if existing is not None:
        raise RuntimeError(
            f"corpus {corpus_spec['name']!r} already exists; reuse it instead of appending chunks"
        )
    docs = _checkout_source(corpus_spec)
    source = PostgresChunkSource(corpus_name=corpus_spec["name"])
    chunking = corpus_spec["chunking"]
    await source.populate_from_folder(
        str(docs),
        min_chars=chunking["min_chars"],
        max_chars=chunking["max_chars"],
        overlap_chars=chunking["overlap_chars"],
        file_extensions=[".md", ".mdx"],
        batch_size=100,
        show_summary=True,
        on_limit="error",
    )
    return await _find_corpus(corpus_spec["name"])


async def prepare(*, ingest: bool, refresh_datasets: bool) -> dict[str, Any]:
    spec = _spec()
    corpus_spec = spec["corpus"]
    corpus = await (_ingest(corpus_spec) if ingest else _find_corpus(corpus_spec["name"]))
    if corpus is None:
        raise RuntimeError(
            f"corpus {corpus_spec['name']!r} was not found; rerun with --ingest to create it once"
        )
    if corpus["chunk_count"] != corpus_spec["expected_chunk_count"]:
        raise RuntimeError(
            f"corpus chunk count changed: expected {corpus_spec['expected_chunk_count']}, "
            f"found {corpus['chunk_count']}"
        )

    SHARED.mkdir(parents=True, exist_ok=True)
    dataset_spec = spec["dataset"]
    dataset_files: dict[str, dict[str, Any]] = {}
    for split in ("train", "eval"):
        destination = SHARED / f"{split}.jsonl"
        source_url = dataset_spec["base_url"] + f"{split}_dataset.jsonl"
        if refresh_datasets or not destination.exists():
            _download(source_url, destination)
        rows = sum(1 for line in destination.read_text(encoding="utf-8").splitlines() if line)
        digest = _sha256(destination)
        expected = dataset_spec["files"][split]
        if rows != expected["rows"] or digest != expected["sha256"]:
            raise RuntimeError(
                f"{split} dataset drifted: expected rows={expected['rows']} "
                f"sha256={expected['sha256']}, found rows={rows} sha256={digest}"
            )
        dataset_files[split] = {
            "path": str(destination.relative_to(ROOT)),
            "rows": rows,
            "sha256": digest,
            "source": source_url,
        }

    manifest = {
        "corpus": {
            "id": corpus["id"],
            "name": corpus["name"],
            "chunk_count": corpus["chunk_count"],
            "source_commit": corpus_spec["source_commit"],
            "chunking": corpus_spec["chunking"],
        },
        "dataset": {
            "revision": dataset_spec["revision"],
            "files": dataset_files,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingest",
        action="store_true",
        help=(
            "Create the corpus only when it does not already exist; "
            "never appends to an existing corpus."
        ),
    )
    parser.add_argument(
        "--refresh-datasets",
        action="store_true",
        help="Download the pinned public train/eval files even when local copies exist.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                prepare(
                    ingest=args.ingest,
                    refresh_datasets=args.refresh_datasets,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
