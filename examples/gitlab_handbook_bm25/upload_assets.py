#!/usr/bin/env -S uv run --isolated --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "castform @ git+https://github.com/castform-ai/benchmax.git@c19b4addb767a745bc8f75e7167afd3958d4dfa3#subdirectory=packages/castform",
# ]
# ///
"""Upload both versioned bundles and the single shared dataset prefix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from castform.platform.client import StorageClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run prepare.py and both build_bundle.py scripts")
    return path


def main() -> None:
    train = _required(ROOT / "shared" / "train.jsonl")
    evaluation = _required(ROOT / "shared" / "eval.jsonl")
    dataset_digest = hashlib.sha256(
        (f"train.jsonl:{_sha256(train)}\neval.jsonl:{_sha256(evaluation)}\n").encode()
    ).hexdigest()[:16]
    dataset_prefix = f"datasets/gitlab-bm25-gateway-ab/{dataset_digest}"

    result: dict[str, Any] = {"arms": {}}
    with StorageClient() as client:
        train_upload = client.upload_local_file(f"{dataset_prefix}/train.jsonl", train)
        eval_upload = client.upload_local_file(f"{dataset_prefix}/eval.jsonl", evaluation)
        train_path = train_upload["blobPath"]
        eval_path = eval_upload["blobPath"]
        uploaded_prefix = train_path.rsplit("/", 1)[0]
        if eval_path.rsplit("/", 1)[0] != uploaded_prefix:
            raise RuntimeError("train and eval datasets uploaded under different prefixes")
        result["dataset"] = {
            "prefix": uploaded_prefix,
            "train_path": train_path,
            "eval_path": eval_path,
            "train_sha256": _sha256(train),
            "eval_sha256": _sha256(evaluation),
        }
        for arm in ("pre_harbor", "post_harbor"):
            cls_path = _required(ROOT / arm / "artifacts" / "env-cls.pkl")
            metadata_path = _required(ROOT / arm / "artifacts" / "env-metadata.json")
            env_digest = hashlib.sha256(
                (f"{_sha256(cls_path)}\n{_sha256(metadata_path)}\n").encode()
            ).hexdigest()[:16]
            env_prefix = f"envs/gitlab-bm25-gateway-ab/{arm}/{env_digest}"
            env_cls_path = f"{env_prefix}/env-cls.pkl"
            env_metadata_path = f"{env_prefix}/env-metadata.json"
            cls_upload = client.upload_local_file(env_cls_path, cls_path)
            metadata_upload = client.upload_local_file(env_metadata_path, metadata_path)
            result["arms"][arm] = {
                "env_cls_path": cls_upload["blobPath"],
                "env_metadata_path": metadata_upload["blobPath"],
                "env_cls_sha256": _sha256(cls_path),
                "env_metadata_sha256": _sha256(metadata_path),
            }

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "uploaded_assets.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
