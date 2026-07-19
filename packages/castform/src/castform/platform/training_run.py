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

from castform import config
from benchmax.bundle import dump_bundle

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
        env_class: Environment implementation to bundle.
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

    train_jsonl = "\n".join(json.dumps(r) for r in train_dataset) + "\n"
    eval_jsonl = "\n".join(json.dumps(r) for r in eval_dataset) + "\n"

    if env_prefix is None:
        env_hash = hashlib.sha256(bundle.pickled).hexdigest()[:16]
        env_prefix = f"envs/{run_name}/{env_hash}"

    if dataset_prefix is None:
        dataset_hash = hashlib.sha256(
            train_jsonl.encode() + eval_jsonl.encode()
        ).hexdigest()[:8]
        dataset_prefix = f"datasets/{run_name}/{dataset_hash}"

    # Reject unsafe keys before uploading — covers both the run_name-derived
    # defaults and any caller-supplied env_prefix/dataset_prefix override.
    _validate_blob_path(env_prefix, source="env_prefix/run_name")
    _validate_blob_path(dataset_prefix, source="dataset_prefix/run_name")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cls_local = tmpdir / "env-cls.pkl"
        meta_local = tmpdir / "env-metadata.json"
        train_local = tmpdir / "train.jsonl"
        eval_local = tmpdir / "eval.jsonl"

        cls_local.write_bytes(bundle.pickled)
        meta_local.write_bytes(bundle.metadata.to_json_bytes())
        train_local.write_text(train_jsonl)
        eval_local.write_text(eval_jsonl)

        return UploadedTrainingRun(
            env_cls_path=storage_client.upload_local_file(
                f"{env_prefix}/env-cls.pkl", cls_local
            )["blobPath"],
            env_metadata_path=storage_client.upload_local_file(
                f"{env_prefix}/env-metadata.json", meta_local
            )["blobPath"],
            train_dataset_path=storage_client.upload_local_file(
                f"{dataset_prefix}/train.jsonl", train_local
            )["blobPath"],
            eval_dataset_path=storage_client.upload_local_file(
                f"{dataset_prefix}/eval.jsonl", eval_local
            )["blobPath"],
        )
