from __future__ import annotations

import asyncio
import logging
import math
import re
import shutil
from builtins import ExceptionGroup
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from benchmax.envs.dataset import Dataset, validate_max_examples
from benchmax.envs.harbor.dep_check import require_harbor
from benchmax.envs.shared_types import Example

if TYPE_CHECKING:
    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import TaskConfig

__all__ = ["HarborDataset"]

logger = logging.getLogger(__name__)

_TASK_DOWNLOAD_MAX_ATTEMPTS = 3
_TASK_DOWNLOAD_RETRY_BASE_DELAY_SECS = 1.0
_CONTENT_HASH = re.compile(r"^[0-9a-f]{64}$")


class HarborDataset(Dataset[Any]):
    """Fixed, content-addressed snapshot of one Harbor dataset."""

    @classmethod
    async def create(
        cls,
        config: DatasetConfig,
        *,
        base_dir: Path,
        disable_verification: bool = False,
        max_examples: int | None = None,
        split: str | None = None,
        eval_ratio: float | None = None,
    ) -> HarborDataset:
        """Resolve and freeze a deterministic, optionally limited task snapshot.

        Package manifests expose the same content digest used for ``Example.id``.
        When every selected task has such a digest, ordering, ratio splitting,
        and limiting happen before task contents are downloaded.
        """

        require_harbor()
        from harbor.models.job.config import DatasetConfig
        from harbor.models.task.id import PackageTaskId
        from harbor.models.trial.config import TaskConfig
        from harbor.publisher.packager import Packager
        from harbor.tasks.client import TaskClient

        validate_max_examples(max_examples)
        if split not in (None, "train", "eval"):
            raise ValueError(f"unknown Harbor dataset split: {split!r}")
        if split is not None:
            if eval_ratio is None:
                raise ValueError("eval_ratio is required when selecting a Harbor split")
            _validate_eval_ratio(eval_ratio)
        elif eval_ratio is not None:
            raise ValueError("eval_ratio requires a Harbor split")

        if not isinstance(config, DatasetConfig):
            raise TypeError(f"Harbor dataset must be DatasetConfig, got {type(config).__name__}")

        snapshot_root = Path(base_dir).expanduser().resolve()
        downloads_dir = snapshot_root / "downloads"
        tasks_dir = snapshot_root / "tasks"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir.mkdir(parents=True, exist_ok=True)

        resolved_config = config.model_copy(
            deep=True,
            update={"download_dir": downloads_dir},
        )
        task_configs = await resolved_config.get_task_configs(
            disable_verification=disable_verification
        )
        if not task_configs:
            raise ValueError("Harbor dataset resolved to no tasks")

        manifest_hashes = [
            _package_content_hash(task_config.get_task_id(), PackageTaskId)
            for task_config in task_configs
        ]
        selected_configs = task_configs
        selected_hashes = manifest_hashes
        selected_from_manifest = False
        if all(content_hash is not None for content_hash in manifest_hashes):
            ordered = sorted(
                zip(task_configs, manifest_hashes, strict=True),
                key=lambda pair: cast(str, pair[1]),
            )
            if split is not None:
                ordered = _select_split_pairs(
                    ordered,
                    split=split,
                    eval_ratio=cast(float, eval_ratio),
                )
            if max_examples is not None:
                ordered = ordered[:max_examples]
            selected_configs = [pair[0] for pair in ordered]
            selected_hashes = [pair[1] for pair in ordered]
            selected_from_manifest = True

        task_ids = [task_config.get_task_id() for task_config in selected_configs]
        downloads = await _download_tasks_with_retries(
            TaskClient(),
            task_ids,
            overwrite=resolved_config.overwrite,
            output_dir=downloads_dir,
        )

        examples: list[Example[TaskConfig]] = []
        seen_ids: set[str] = set()
        for task_config, download in zip(
            selected_configs,
            downloads.results,
            strict=True,
        ):
            content_hash, _ = await asyncio.to_thread(
                Packager.compute_content_hash,
                download.path,
            )
            example_id = content_hash
            if example_id in seen_ids:
                raise ValueError(f"Harbor dataset contains duplicate task content: {example_id}")
            seen_ids.add(example_id)

            snapshot_path = tasks_dir / content_hash
            await asyncio.to_thread(
                _copy_content_addressed_snapshot,
                download.path,
                snapshot_path,
                content_hash,
                Packager,
            )
            examples.append(
                Example(
                    id=example_id,
                    payload=TaskConfig(
                        path=snapshot_path,
                        source=task_config.source,
                    ),
                )
            )

        examples.sort(key=lambda example: example.id)
        if selected_from_manifest:
            expected_hashes = {cast(str, value) for value in selected_hashes}
            actual_hashes = {example.id for example in examples}
            if actual_hashes != expected_hashes:
                raise RuntimeError(
                    "Harbor task content did not match the package manifest: "
                    f"expected={sorted(expected_hashes)} actual={sorted(actual_hashes)}"
                )
        else:
            if split is not None:
                examples = list(
                    _select_split_examples(
                        examples,
                        split=split,
                        eval_ratio=cast(float, eval_ratio),
                    )
                )
            if max_examples is not None:
                examples = examples[:max_examples]
        return cls(examples)

    def train_eval_split(
        self,
        eval_ratio: float,
    ) -> tuple[HarborDataset, HarborDataset]:
        """Return deterministic, complementary views over this snapshot."""

        _validate_eval_ratio(eval_ratio)

        examples = tuple(sorted(self, key=lambda example: example.id))
        if eval_ratio == 0 or len(examples) < 2:
            return HarborDataset(examples), HarborDataset(())

        eval_count = min(len(examples) - 1, max(1, math.ceil(len(examples) * eval_ratio)))
        return HarborDataset(examples[eval_count:]), HarborDataset(examples[:eval_count])


def _package_content_hash(task_id: Any, package_task_type: type[Any]) -> str | None:
    if not isinstance(task_id, package_task_type) or task_id.ref is None:
        return None
    content_hash = str(task_id.ref).removeprefix("sha256:").lower()
    return content_hash if _CONTENT_HASH.fullmatch(content_hash) else None


def _validate_eval_ratio(eval_ratio: float) -> None:
    if (
        isinstance(eval_ratio, bool)
        or not isinstance(eval_ratio, (int, float))
        or not math.isfinite(eval_ratio)
        or not 0 <= eval_ratio < 1
    ):
        raise ValueError("eval_ratio must satisfy 0 <= eval_ratio < 1")


def _eval_count(size: int, eval_ratio: float) -> int:
    if eval_ratio == 0 or size < 2:
        return 0
    return min(size - 1, max(1, math.ceil(size * eval_ratio)))


def _select_split_pairs(
    ordered: list[tuple[Any, str | None]],
    *,
    split: str,
    eval_ratio: float,
) -> list[tuple[Any, str | None]]:
    count = _eval_count(len(ordered), eval_ratio)
    if split == "eval":
        if count == 0:
            if eval_ratio == 0:
                raise ValueError("HarborEnv automatic eval is disabled by eval_ratio=0")
            raise ValueError("HarborEnv automatic eval requires at least two dataset examples")
        return ordered[:count]
    return ordered[count:]


def _select_split_examples(
    examples: list[Example[Any]],
    *,
    split: str,
    eval_ratio: float,
) -> list[Example[Any]]:
    count = _eval_count(len(examples), eval_ratio)
    if split == "eval":
        if count == 0:
            if eval_ratio == 0:
                raise ValueError("HarborEnv automatic eval is disabled by eval_ratio=0")
            raise ValueError("HarborEnv automatic eval requires at least two dataset examples")
        return examples[:count]
    return examples[count:]


async def _download_tasks_with_retries(
    client: Any,
    task_ids: list[Any],
    *,
    overwrite: bool,
    output_dir: Path,
) -> Any:
    """Retry failed Harbor download batches while preserving completed task caches."""

    for attempt in range(1, _TASK_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            return await client.download_tasks(
                task_ids,
                overwrite=overwrite,
                output_dir=output_dir,
            )
        except ExceptionGroup:
            if attempt == _TASK_DOWNLOAD_MAX_ATTEMPTS:
                raise
            delay = _TASK_DOWNLOAD_RETRY_BASE_DELAY_SECS * 2 ** (attempt - 1)
            logger.warning(
                "Harbor task download batch failed; retrying cached batch in %.1fs (attempt %d/%d)",
                delay,
                attempt + 1,
                _TASK_DOWNLOAD_MAX_ATTEMPTS,
                exc_info=True,
            )
            await asyncio.sleep(delay)

    raise AssertionError("unreachable")


def _copy_content_addressed_snapshot(
    source: Path,
    destination: Path,
    expected_hash: str,
    packager: Any,
) -> None:
    """Copy once, then verify an existing or newly-created content snapshot."""

    if not destination.exists():
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        try:
            shutil.copytree(source, temporary)
            try:
                temporary.rename(destination)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    actual_hash, _ = packager.compute_content_hash(destination)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Harbor task snapshot content changed while it was being created: "
            f"expected sha256:{expected_hash}, got sha256:{actual_hash}"
        )
