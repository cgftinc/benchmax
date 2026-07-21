"""High-level helper for uploading a prepared training run.

Uploads a completed environment bundle and any caller-supplied datasets in a
single call. The returned dataclass spreads into
``TrainerClient.launch_training_run``.

Bundling remains a BenchMax concern. This helper deliberately accepts a
``Bundle`` instead of environment construction inputs so Castform cannot
silently rebuild or reinterpret the artifact selected by the caller.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmax.bundle import Bundle
from castform import config

from .client import StorageClient


@dataclass(frozen=True)
class UploadedTrainingRun:
    """Blob paths for a training run's uploaded assets.

    Field names match ``TrainerClient.launch_training_run`` kwargs so the
    result spreads directly into the launch call::

        uploaded = upload_training_run(...)
        run_id = trainer.launch_training_run(
            **dataclasses.asdict(uploaded),
        )
    """

    env_cls_path: str
    env_metadata_path: str
    train_dataset_path: str | None = None
    eval_dataset_path: str | None = None


# Mirrors the platform's blob-path guard (isSafeBlobPath): the upload endpoint
# rejects keys whose segments fall outside this charset, since a stray
# `?`/`#`/space breaks the trainer's SAS-URL parsing downstream. Fail loud here
# with an actionable message instead of letting the user hit an opaque 400.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_blob_path(path: str, *, source: str) -> None:
    for seg in path.split("/"):
        if not seg or seg in (".", "..") or not _SAFE_SEGMENT.fullmatch(seg):
            raise ValueError(
                f"Invalid storage path segment {seg!r} in {path!r} (from {source}). "
                f"Path segments may contain only letters, digits, '.', '_', '-'. "
                f"The platform rejects others (e.g. '?', '#', spaces) because they "
                f"break blob-URL parsing. Use a run_name without such characters."
            )


def upload_training_run(
    *,
    bundle: Bundle,
    train_dataset: list[dict[str, Any]] | None = None,
    eval_dataset: list[dict[str, Any]] | None = None,
    run_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    env_prefix: str | None = None,
    dataset_prefix: str | None = None,
    storage_client: StorageClient | None = None,
) -> UploadedTrainingRun:
    """Upload a completed environment bundle and optional datasets.

    Default layout:
        envs/<run_name>/<env_hash>/{env-cls.pkl, env-metadata.json}
        datasets/<run_name>/<dataset_hash>/{train.jsonl, eval.jsonl}

    Hashes are sha256 of the pickled bundle (envs) and supplied JSONL bytes
    (datasets), truncated to 16 / 8 hex chars. When no dataset is supplied,
    Castform uploads only the bundle and returns ``None`` for both dataset paths.

    Args:
        bundle: Completed BenchMax environment bundle. Its pickle and metadata
            are uploaded exactly as supplied.
        train_dataset: Optional training examples. ``None`` means this split is
            managed by the environment/runtime and is not uploaded. An empty
            list is still an explicitly supplied dataset and uploads an empty
            JSONL file.
        eval_dataset: Optional evaluation examples, with the same semantics.
        run_name: Training run identifier; used as the storage path segment.
        api_key: Platform API key. Optional — when omitted (and no
            ``storage_client`` is passed) the bearer resolves per request via
            the credential seam (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY``).
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        env_prefix: Override the default env directory. When set, env files
            land at ``<env_prefix>/{env-cls.pkl, env-metadata.json}``.
        dataset_prefix: Override the default dataset directory. Valid only when
            at least one dataset is supplied; supplied JSONL files land below it.
        storage_client: BYOC. Pass an existing client to reuse its connection
            pool, custom timeouts, or test fakes. Otherwise constructed from
            ``api_key``/``base_url``.

    Returns:
        UploadedTrainingRun containing bundle paths and optional dataset paths.
    """
    has_datasets = train_dataset is not None or eval_dataset is not None
    if dataset_prefix is not None and not has_datasets:
        raise ValueError("dataset_prefix requires at least one supplied dataset")

    train_jsonl = (
        "\n".join(json.dumps(row) for row in train_dataset) + "\n"
        if train_dataset is not None
        else None
    )
    eval_jsonl = (
        "\n".join(json.dumps(row) for row in eval_dataset) + "\n"
        if eval_dataset is not None
        else None
    )

    if storage_client is None:
        # api_key optional: StorageClient resolves the bearer per request via
        # the credential seam (ACT_AS_TOKEN_PATH / PLATFORM_API_KEY) when unset.
        storage_client = StorageClient(
            api_key=api_key,
            base_url=base_url or config.platform_url(),
        )

    if env_prefix is None:
        env_hash = hashlib.sha256(bundle.pickled).hexdigest()[:16]
        env_prefix = f"envs/{run_name}/{env_hash}"

    if dataset_prefix is None and has_datasets:
        if train_jsonl is not None and eval_jsonl is not None:
            dataset_hash_input = train_jsonl.encode() + eval_jsonl.encode()
        elif train_jsonl is not None:
            dataset_hash_input = b"train\0" + train_jsonl.encode()
        else:
            assert eval_jsonl is not None
            dataset_hash_input = b"eval\0" + eval_jsonl.encode()
        dataset_hash = hashlib.sha256(dataset_hash_input).hexdigest()[:8]
        dataset_prefix = f"datasets/{run_name}/{dataset_hash}"

    # Reject unsafe keys before uploading — covers both the run_name-derived
    # defaults and any caller-supplied env_prefix/dataset_prefix override.
    _validate_blob_path(env_prefix, source="env_prefix/run_name")
    if dataset_prefix is not None:
        _validate_blob_path(dataset_prefix, source="dataset_prefix/run_name")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cls_local = tmpdir / "env-cls.pkl"
        meta_local = tmpdir / "env-metadata.json"
        cls_local.write_bytes(bundle.pickled)
        meta_local.write_bytes(bundle.metadata.to_json_bytes())
        env_cls_path = storage_client.upload_local_file(
            f"{env_prefix}/env-cls.pkl", cls_local
        )["blobPath"]
        env_metadata_path = storage_client.upload_local_file(
            f"{env_prefix}/env-metadata.json", meta_local
        )["blobPath"]

        train_dataset_path = None
        eval_dataset_path = None
        if train_jsonl is not None:
            assert dataset_prefix is not None
            train_local = tmpdir / "train.jsonl"
            train_local.write_text(train_jsonl)
            train_dataset_path = storage_client.upload_local_file(
                f"{dataset_prefix}/train.jsonl", train_local
            )["blobPath"]
        if eval_jsonl is not None:
            assert dataset_prefix is not None
            eval_local = tmpdir / "eval.jsonl"
            eval_local.write_text(eval_jsonl)
            eval_dataset_path = storage_client.upload_local_file(
                f"{dataset_prefix}/eval.jsonl", eval_local
            )["blobPath"]

        return UploadedTrainingRun(
            env_cls_path=env_cls_path,
            env_metadata_path=env_metadata_path,
            train_dataset_path=train_dataset_path,
            eval_dataset_path=eval_dataset_path,
        )
