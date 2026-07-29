"""High-level helper for uploading environment assets.

Uploads a completed environment bundle and any caller-supplied datasets in a
single call. The returned dataclass spreads into
``TrainerClient.launch_training_run``.

Bundling remains a benchmax concern. This helper deliberately accepts a
``Bundle`` instead of environment construction inputs so Castform cannot
silently rebuild or reinterpret the artifact selected by the caller.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmax.bundle import Bundle, bundle_digest

from castform import config

from .client import StorageClient


@dataclass(frozen=True)
class UploadedEnvironmentAssets:
    """Blob paths for an uploaded environment bundle and optional datasets.

    Field names match ``TrainerClient.launch_training_run`` kwargs so the
    result spreads directly into the launch call::

        uploaded = upload_assets(...)
        run_id = trainer.launch_training_run(
            **dataclasses.asdict(uploaded),
        )

    ``dataset_path`` is the blob prefix holding every uploaded dataset file.
    The trainer mirrors that prefix to the machine and hands it to the
    environment as its dataset base directory.
    """

    env_cls_path: str
    env_metadata_path: str
    dataset_path: str | None = None


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


def upload_assets(
    *,
    bundle: Bundle,
    train_dataset: list[dict[str, Any]] | None = None,
    eval_dataset: list[dict[str, Any]] | None = None,
    dataset_files: Mapping[str, bytes | str | Path] | None = None,
    run_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    env_prefix: str | None = None,
    dataset_prefix: str | None = None,
    storage_client: StorageClient | None = None,
) -> UploadedEnvironmentAssets:
    """Upload a completed environment bundle and optional dataset files.

    Default layout:
        envs/<run_name>/<env_hash>/{env-cls.pkl, env-metadata.json}
        datasets/<run_name>/<dataset_hash>/<every dataset file>

    Environment paths use benchmax's complete-artifact digest, covering both
    the pickle and canonical metadata. The dataset prefix hashes every
    uploaded file's name and bytes. The hashes are truncated to 16 / 8 hex
    chars. When no dataset content is supplied, Castform uploads only the
    bundle and returns ``None`` for ``dataset_path``.

    Args:
        bundle: Completed benchmax environment bundle. Its pickle and metadata
            are uploaded exactly as supplied.
        train_dataset: Optional training examples, written as ``train.jsonl``
            under the dataset prefix. ``None`` means the environment manages
            this split at runtime. An empty list is still an explicitly
            supplied dataset and uploads an empty JSONL file.
        eval_dataset: Optional evaluation examples, written as ``eval.jsonl``,
            with the same semantics.
        dataset_files: Arbitrary extra dataset content: relative file name
            (subdirectories allowed) mapped to bytes, text, or a local Path to
            read. Names must not collide with the generated JSONL splits.
        run_name: Asset namespace used as the storage path segment.
        api_key: Platform API key. Optional — when omitted (and no
            ``storage_client`` is passed) the bearer resolves per request via
            the credential seam (``ACT_AS_TOKEN_PATH`` / ``CASTFORM_API_KEY``).
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        env_prefix: Override the default env directory. When set, env files
            land at ``<env_prefix>/{env-cls.pkl, env-metadata.json}``.
        dataset_prefix: Override the default dataset directory. Valid only when
            at least one dataset file is supplied.
        storage_client: BYOC. Pass an existing client to reuse its connection
            pool, custom timeouts, or test fakes. Otherwise constructed from
            ``api_key``/``base_url``.

    Returns:
        UploadedEnvironmentAssets with the bundle paths and the optional dataset
        prefix (``dataset_path``).
    """
    files = _collect_dataset_files(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_files=dataset_files,
    )
    if dataset_prefix is not None and not files:
        raise ValueError("dataset_prefix requires at least one supplied dataset file")

    if storage_client is None:
        # api_key optional: StorageClient resolves the bearer per request via
        # the credential seam (ACT_AS_TOKEN_PATH / CASTFORM_API_KEY) when unset.
        storage_client = StorageClient(
            api_key=api_key,
            base_url=base_url or config.platform_url(),
        )

    if env_prefix is None:
        env_hash = bundle_digest(bundle)[:16]
        env_prefix = f"envs/{run_name}/{env_hash}"

    if dataset_prefix is None and files:
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(files[name])
        dataset_prefix = f"datasets/{run_name}/{digest.hexdigest()[:8]}"

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
        env_cls_path = storage_client.upload_local_file(f"{env_prefix}/env-cls.pkl", cls_local)[
            "blobPath"
        ]
        env_metadata_path = storage_client.upload_local_file(
            f"{env_prefix}/env-metadata.json", meta_local
        )["blobPath"]

        for name, content in files.items():
            assert dataset_prefix is not None
            local = tmpdir / "dataset" / name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
            storage_client.upload_local_file(f"{dataset_prefix}/{name}", local)

        return UploadedEnvironmentAssets(
            env_cls_path=env_cls_path,
            env_metadata_path=env_metadata_path,
            dataset_path=dataset_prefix,
        )


def _collect_dataset_files(
    *,
    train_dataset: list[dict[str, Any]] | None,
    eval_dataset: list[dict[str, Any]] | None,
    dataset_files: Mapping[str, bytes | str | Path] | None,
) -> dict[str, bytes]:
    """Normalize every dataset input into relative-name -> bytes."""

    files: dict[str, bytes] = {}
    if train_dataset is not None:
        files["train.jsonl"] = _jsonl_bytes(train_dataset)
    if eval_dataset is not None:
        files["eval.jsonl"] = _jsonl_bytes(eval_dataset)
    for name, content in (dataset_files or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("dataset_files names must be non-empty strings")
        _validate_blob_path(name, source="dataset_files")
        if name in files:
            raise ValueError(f"dataset_files name {name!r} collides with a generated JSONL split")
        if isinstance(content, Path):
            files[name] = content.read_bytes()
        elif isinstance(content, str):
            files[name] = content.encode("utf-8")
        elif isinstance(content, bytes):
            files[name] = content
        else:
            raise TypeError(
                f"dataset_files[{name!r}] must be bytes, str, or Path, got {type(content).__name__}"
            )
    return files


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")
