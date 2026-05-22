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

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmax import config
from benchmax.bundle import dump_bundle

from .client import StorageClient


@dataclass(frozen=True)
class UploadedTrainingRun:
    """Blob paths for a training run's uploaded assets.

    Field names match ``TrainerClient.launch_training_run`` kwargs so the
    result spreads directly into the launch call::

        uploaded = upload_training_run(...)
        run_id = trainer.launch_training_run(
            training_run_type="simple",
            **dataclasses.asdict(uploaded),
        )
    """

    env_cls_path: str
    env_metadata_path: str
    train_dataset_path: str
    eval_dataset_path: str


def upload_training_run(
    *,
    env_class: type,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    constructor_args: dict[str, Any] | None = None,
    pip_dependencies: list[str] | None = None,
    local_modules: list[ModuleType] | None = None,
    storage_prefix: str = "training-runs",
    storage_client: StorageClient | None = None,
) -> UploadedTrainingRun:
    """Bundle the env class and upload it + datasets to platform storage.

    Args:
        env_class: BaseEnv subclass to bundle.
        train_dataset: Training examples (list of dicts).
        eval_dataset: Eval examples (list of dicts).
        name: Training run name; used as the storage path segment.
        api_key: Platform API key. Required if ``storage_client`` not provided.
        base_url: Platform base URL. Defaults to ``config.platform_url()``.
        constructor_args: Optional kwargs to bake into the env bundle.
        pip_dependencies: Pip deps to install on the trainer before unpickling.
        local_modules: Module objects to pickle by value (for envs that import
            from local .py files). See ``dump_bundle`` docs.
        storage_prefix: Storage path prefix. Files land at
            ``{storage_prefix}/{name}/{file}``. Default: ``"training-runs"``.
        storage_client: BYOC. Pass an existing client to reuse its connection
            pool, custom timeouts, or test fakes. Otherwise constructed from
            ``api_key``/``base_url``.

    Returns:
        UploadedTrainingRun containing the four blob paths.
    """
    if storage_client is None:
        if api_key is None:
            raise ValueError("Provide either api_key or storage_client")
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

    prefix = f"{storage_prefix}/{name}"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cls_local = tmpdir / "env-cls.pkl"
        meta_local = tmpdir / "env-metadata.json"
        train_local = tmpdir / "train.jsonl"
        eval_local = tmpdir / "eval.jsonl"

        cls_local.write_bytes(bundle.pickled)
        meta_local.write_bytes(bundle.metadata.to_json_bytes())
        train_local.write_text(
            "\n".join(json.dumps(r) for r in train_dataset) + "\n"
        )
        eval_local.write_text(
            "\n".join(json.dumps(r) for r in eval_dataset) + "\n"
        )

        return UploadedTrainingRun(
            env_cls_path=storage_client.upload_local_file(
                f"{prefix}/env-cls.pkl", cls_local
            )["blobPath"],
            env_metadata_path=storage_client.upload_local_file(
                f"{prefix}/env-metadata.json", meta_local
            )["blobPath"],
            train_dataset_path=storage_client.upload_local_file(
                f"{prefix}/train.jsonl", train_local
            )["blobPath"],
            eval_dataset_path=storage_client.upload_local_file(
                f"{prefix}/eval.jsonl", eval_local
            )["blobPath"],
        )
