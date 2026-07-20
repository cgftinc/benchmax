"""High-level helper for preparing a training run.

Bundles an env class and uploads everything required to launch in a single
call. The returned dataclass spreads into ``TrainerClient.launch_training_run``
and ``RolloutClient.validate_examples``.

Composition (not collapse): the underlying primitives — ``dump_bundle``,
``StorageClient.upload_local_file`` — remain independently usable. This
helper exists for the common case where the user has an env class + two
in-memory datasets and wants the four blob paths needed to launch.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmax import config
from benchmax.bundle import dump_bundle
from benchmax.sft.dataset import SftDataset, canonical_jsonl

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
    train_dataset_path: str
    eval_dataset_path: str


@dataclass(frozen=True)
class UploadedSftRun:
    """Blob paths for an SFT run's uploaded datasets.

    ``eval_dataset_path`` is ``None`` when no eval dataset was supplied to
    :func:`upload_sft_run` — nothing is uploaded for eval in that case.
    """

    train_dataset_path: str
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
    env_class: type,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    run_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    constructor_args: dict[str, Any] | None = None,
    pip_dependencies: list[str] | None = None,
    local_modules: list[ModuleType] | None = None,
    env_prefix: str | None = None,
    dataset_prefix: str | None = None,
    storage_client: StorageClient | None = None,
) -> UploadedTrainingRun:
    """Bundle the env class and upload it + datasets to platform storage.

    Default layout:
        envs/<run_name>/<env_hash>/{env-cls.pkl, env-metadata.json}
        datasets/<run_name>/<dataset_hash>/{train.jsonl, eval.jsonl}

    Hashes are sha256 of the pickled bundle (envs) and of the concatenated
    train+eval JSONL bytes (datasets), truncated to 16 / 8 hex chars.

    Args:
        env_class: BaseEnv subclass to bundle.
        train_dataset: Training examples (list of dicts).
        eval_dataset: Eval examples (list of dicts).
        run_name: Training run identifier; used as the storage path segment.
        api_key: Platform API key. Optional — when omitted (and no
            ``storage_client`` is passed) the bearer resolves per request via
            the credential seam (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY``).
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        constructor_args: Optional kwargs to bake into the env bundle.
        pip_dependencies: Pip deps to install on the trainer before unpickling.
        local_modules: Module objects to pickle by value (for envs that import
            from local .py files). See ``dump_bundle`` docs.
        env_prefix: Override the default env directory. When set, env files
            land at ``<env_prefix>/{env-cls.pkl, env-metadata.json}``.
        dataset_prefix: Override the default dataset directory. When set,
            JSONL files land at ``<dataset_prefix>/{train.jsonl, eval.jsonl}``.
        storage_client: BYOC. Pass an existing client to reuse its connection
            pool, custom timeouts, or test fakes. Otherwise constructed from
            ``api_key``/``base_url``.

    Returns:
        UploadedTrainingRun containing the four blob paths.
    """
    if storage_client is None:
        # api_key optional: StorageClient resolves the bearer per request via
        # the credential seam (ACT_AS_TOKEN_PATH / PLATFORM_API_KEY) when unset.
        storage_client = StorageClient(
            api_key=api_key,
            base_url=base_url or config.platform_url(),
        )

    bundle = dump_bundle(
        env_class,
        constructor_args=constructor_args,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    )

    train_jsonl = ("\n".join(json.dumps(r) for r in train_dataset) + "\n").encode()
    eval_jsonl = ("\n".join(json.dumps(r) for r in eval_dataset) + "\n").encode()

    if env_prefix is None:
        env_hash = hashlib.sha256(bundle.pickled).hexdigest()[:16]
        env_prefix = f"envs/{run_name}/{env_hash}"

    # Reject unsafe keys before uploading — covers both the run_name-derived
    # defaults and any caller-supplied env_prefix override.
    _validate_blob_path(env_prefix, source="env_prefix/run_name")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cls_local = tmpdir / "env-cls.pkl"
        meta_local = tmpdir / "env-metadata.json"

        cls_local.write_bytes(bundle.pickled)
        meta_local.write_bytes(bundle.metadata.to_json_bytes())

        train_path, eval_path = _upload_dataset_pair(
            storage_client=storage_client,
            train_jsonl=train_jsonl,
            eval_jsonl=eval_jsonl,
            run_name=run_name,
            dataset_prefix=dataset_prefix,
        )

        return UploadedTrainingRun(
            env_cls_path=storage_client.upload_local_file(
                f"{env_prefix}/env-cls.pkl", cls_local
            )["blobPath"],
            env_metadata_path=storage_client.upload_local_file(
                f"{env_prefix}/env-metadata.json", meta_local
            )["blobPath"],
            train_dataset_path=train_path,
            # eval_jsonl is always provided here (never None), so eval_path is
            # always a str — the SFT path is the only caller that sees None.
            eval_dataset_path=eval_path,  # type: ignore[arg-type]
        )


def _upload_dataset_pair(
    *,
    storage_client: StorageClient,
    train_jsonl: bytes,
    eval_jsonl: bytes | None,
    run_name: str,
    dataset_prefix: str | None,
) -> tuple[str, str | None]:
    """Upload a train (+ optional eval) JSONL pair to ``datasets/<run>/<hash>/``.

    The reusable dataset-upload half shared by :func:`upload_training_run`
    (eval always present) and :func:`upload_sft_run` (eval optional).
    ``eval_jsonl=None`` uploads nothing for eval and returns ``None`` for its
    path. The prefix hash covers whatever bytes are actually uploaded, so
    train-only and train+eval runs land in different hash buckets.
    """
    if dataset_prefix is None:
        hashed = train_jsonl + (eval_jsonl or b"")
        dataset_hash = hashlib.sha256(hashed).hexdigest()[:8]
        dataset_prefix = f"datasets/{run_name}/{dataset_hash}"

    _validate_blob_path(dataset_prefix, source="dataset_prefix/run_name")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        train_local = tmpdir / "train.jsonl"
        train_local.write_bytes(train_jsonl)
        train_path = storage_client.upload_local_file(
            f"{dataset_prefix}/train.jsonl", train_local
        )["blobPath"]

        eval_path: str | None = None
        if eval_jsonl is not None:
            eval_local = tmpdir / "eval.jsonl"
            eval_local.write_bytes(eval_jsonl)
            eval_path = storage_client.upload_local_file(
                f"{dataset_prefix}/eval.jsonl", eval_local
            )["blobPath"]

    return train_path, eval_path


def upload_sft_run(
    *,
    train: SftDataset,
    eval: SftDataset | None = None,
    run_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    dataset_prefix: str | None = None,
    storage_client: StorageClient | None = None,
) -> UploadedSftRun:
    """Serialize an SFT train (+ optional eval) dataset and upload it.

    Serializes rows via :func:`benchmax.sft.dataset.canonical_jsonl` only —
    the canonicalization boundary's upload side; there is no raw-rows entry
    point. ``eval=None`` uploads nothing for eval and ``eval_dataset_path``
    on the result is ``None``.

    Default layout: ``datasets/<run_name>/<dataset_hash>/{train.jsonl,
    eval.jsonl}`` (hash is sha256 of the uploaded bytes, truncated to 8 hex
    chars — omits eval bytes when there's no eval dataset).

    Args:
        train: Canonicalized training dataset (see ``sft.load_sft_dataset``).
        eval: Canonicalized eval dataset, or ``None`` to skip eval entirely.
        run_name: Training run identifier; used as the storage path segment.
        api_key: Platform API key. Optional — when omitted (and no
            ``storage_client`` is passed) the bearer resolves per request via
            the credential seam (``ACT_AS_TOKEN_PATH`` / ``PLATFORM_API_KEY``).
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        dataset_prefix: Override the default dataset directory. When set,
            JSONL files land at ``<dataset_prefix>/{train.jsonl, eval.jsonl}``.
        storage_client: BYOC. Pass an existing client to reuse its connection
            pool, custom timeouts, or test fakes. Otherwise constructed from
            ``api_key``/``base_url``.

    Returns:
        UploadedSftRun with the train path and the optional eval path.
    """
    if storage_client is None:
        storage_client = StorageClient(
            api_key=api_key,
            base_url=base_url or config.platform_url(),
        )

    train_jsonl = canonical_jsonl(train)
    eval_jsonl = canonical_jsonl(eval) if eval is not None else None

    train_path, eval_path = _upload_dataset_pair(
        storage_client=storage_client,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        run_name=run_name,
        dataset_prefix=dataset_prefix,
    )

    return UploadedSftRun(train_dataset_path=train_path, eval_dataset_path=eval_path)
