from __future__ import annotations

import asyncio
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from benchmax.envs.dataset import Dataset
from benchmax.envs.harbor.dep_check import require_harbor
from benchmax.envs.shared_types import Example

if TYPE_CHECKING:
    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import TaskConfig

__all__ = ["HarborDataset"]


class HarborDataset(Dataset[Any]):
    """Fixed, content-addressed snapshot of one Harbor dataset."""

    @classmethod
    async def create(
        cls,
        config: DatasetConfig,
        *,
        base_dir: Path,
        disable_verification: bool = False,
    ) -> HarborDataset:
        """Resolve Harbor's dataset config and freeze every task below ``base_dir``."""

        require_harbor()
        from harbor.models.job.config import DatasetConfig
        from harbor.models.trial.config import TaskConfig
        from harbor.publisher.packager import Packager
        from harbor.tasks.client import TaskClient

        if not isinstance(config, DatasetConfig):
            raise TypeError(
                f"Harbor dataset must be DatasetConfig, got {type(config).__name__}"
            )

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

        task_ids = [task_config.get_task_id() for task_config in task_configs]
        downloads = await TaskClient().download_tasks(
            task_ids,
            overwrite=resolved_config.overwrite,
            output_dir=downloads_dir,
        )

        examples: list[Example[TaskConfig]] = []
        seen_ids: set[str] = set()
        for task_config, download in zip(
            task_configs,
            downloads.results,
            strict=True,
        ):
            content_hash, _ = await asyncio.to_thread(
                Packager.compute_content_hash,
                download.path,
            )
            example_id = content_hash
            if example_id in seen_ids:
                raise ValueError(
                    f"Harbor dataset contains duplicate task content: {example_id}"
                )
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
        return cls(examples)

    def train_eval_split(
        self,
        eval_ratio: float,
    ) -> tuple[HarborDataset, HarborDataset]:
        """Return deterministic, complementary views over this snapshot."""

        if (
            isinstance(eval_ratio, bool)
            or not isinstance(eval_ratio, (int, float))
            or not math.isfinite(eval_ratio)
            or not 0 <= eval_ratio < 1
        ):
            raise ValueError("eval_ratio must satisfy 0 <= eval_ratio < 1")

        examples = tuple(sorted(self, key=lambda example: example.id))
        if eval_ratio == 0 or len(examples) < 2:
            return HarborDataset(examples), HarborDataset(())

        eval_count = min(
            len(examples) - 1, max(1, math.ceil(len(examples) * eval_ratio))
        )
        return HarborDataset(examples[eval_count:]), HarborDataset(
            examples[:eval_count]
        )


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
