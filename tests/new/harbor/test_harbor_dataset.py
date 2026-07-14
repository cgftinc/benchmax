from __future__ import annotations

from pathlib import Path

import pytest
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)
from harbor.models.trial.config import TaskConfig

from benchmax.envs import Dataset
from benchmax.envs.harbor import HarborDataset, HarborEnv, HarborTrialTemplate


@pytest.mark.asyncio
async def test_harbor_dataset_identity_is_content_addressed_and_snapshotted(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "machine-a" / "dataset"
    second_source = tmp_path / "machine-b" / "dataset"
    _write_task(first_source / "task", instruction="Solve the task.")
    _write_task(second_source / "task", instruction="Solve the task.")

    first = await HarborDataset.create(
        DatasetConfig(path=first_source),
        base_dir=tmp_path / "cache-a",
    )
    second = await HarborDataset.create(
        DatasetConfig(path=second_source),
        base_dir=tmp_path / "cache-b",
    )

    assert len(first) == len(second) == 1
    assert first[0].id == second[0].id
    assert first[0].id.startswith("sha256:")
    assert first[0].payload.path != second[0].payload.path
    assert first[0].payload.path.is_relative_to(tmp_path / "cache-a")
    assert second[0].payload.path.is_relative_to(tmp_path / "cache-b")

    (first_source / "task" / "instruction.md").write_text("Changed later.")
    assert first[0].payload.path.joinpath("instruction.md").read_text() == (
        "Solve the task."
    )

    _write_task(second_source / "task", instruction="A different task.")
    changed = await HarborDataset.create(
        DatasetConfig(path=second_source),
        base_dir=tmp_path / "cache-c",
    )
    assert changed[0].id != first[0].id


@pytest.mark.asyncio
async def test_harbor_env_selects_lowest_canonical_ids_for_eval_ratio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    for index in range(10):
        _write_task(source / f"task-{index}", instruction=f"Task {index}")

    default_env = _make_env(DatasetConfig(path=source))
    default_train = await default_env.create_dataset("train", tmp_path / "default")
    default_eval = await default_env.create_dataset("eval", tmp_path / "default")

    assert len(default_train) == 9
    assert len(default_eval) == 1
    _assert_complementary(default_train, default_eval)
    assert (
        len(
            {
                example.payload.path.parent
                for example in (*default_train, *default_eval)
                if example.payload.path is not None
            }
        )
        == 1
    )

    custom_env = _make_env(DatasetConfig(path=source), eval_ratio=0.3)
    custom_train = await custom_env.create_dataset("train", tmp_path / "custom")
    custom_eval = await custom_env.create_dataset("eval", tmp_path / "custom")

    assert len(custom_train) == 7
    assert len(custom_eval) == 3
    _assert_complementary(custom_train, custom_eval)
    assert [example.id for example in custom_eval] == sorted(
        example.id for example in (*custom_train, *custom_eval)
    )[:3]


@pytest.mark.asyncio
async def test_explicit_eval_dataset_overrides_ratio_splitting(tmp_path: Path) -> None:
    train_source = tmp_path / "train-source"
    eval_source = tmp_path / "eval-source"
    for index in range(3):
        _write_task(train_source / f"task-{index}", instruction=f"Train {index}")
    for index in range(2):
        _write_task(eval_source / f"task-{index}", instruction=f"Eval {index}")

    env = _make_env(
        DatasetConfig(path=train_source),
        eval_dataset=DatasetConfig(path=eval_source),
        eval_ratio=0.9,
    )
    train = await env.create_dataset("train", tmp_path / "cache")
    eval_ = await env.create_dataset("eval", tmp_path / "cache")

    assert len(train) == 3
    assert len(eval_) == 2
    assert _instructions(train) == {f"Train {index}" for index in range(3)}
    assert _instructions(eval_) == {f"Eval {index}" for index in range(2)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("example_count", "eval_ratio", "message"),
    [
        (2, 0.0, "automatic eval is disabled"),
        (1, 0.1, "requires at least two dataset examples"),
    ],
)
async def test_harbor_env_rejects_unavailable_automatic_eval(
    tmp_path: Path,
    example_count: int,
    eval_ratio: float,
    message: str,
) -> None:
    source = tmp_path / "dataset"
    for index in range(example_count):
        _write_task(source / f"task-{index}", instruction=f"Task {index}")

    env = _make_env(DatasetConfig(path=source), eval_ratio=eval_ratio)

    with pytest.raises(ValueError, match=message):
        await env.create_dataset("eval", tmp_path / "cache")


def _make_env(
    dataset: DatasetConfig,
    *,
    eval_dataset: DatasetConfig | None = None,
    eval_ratio: float = 0.1,
) -> HarborEnv:
    return HarborEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        eval_ratio=eval_ratio,
        trial=HarborTrialTemplate(
            agent=TrialAgentConfig(name="mini-swe-agent"),
            environment=TrialEnvironmentConfig(type=EnvironmentType.DOCKER),
            verifier=TrialVerifierConfig(),
        ),
    )


def _assert_complementary(
    train: Dataset[TaskConfig],
    eval_: Dataset[TaskConfig],
) -> None:
    train_ids = {example.id for example in train}
    eval_ids = {example.id for example in eval_}
    assert train_ids.isdisjoint(eval_ids)
    assert len(train_ids | eval_ids) == len(train) + len(eval_)


def _instructions(dataset: Dataset[TaskConfig]) -> set[str]:
    return {
        example.payload.path.joinpath("instruction.md").read_text()
        for example in dataset
        if example.payload.path is not None
    }


def _write_task(task_dir: Path, *, instruction: str) -> None:
    task_dir.joinpath("environment").mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("tests").mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("task.toml").write_text(
        """\
version = "1.0"

[task]
name = "tests/example"
authors = []
keywords = []

[verifier]
timeout_sec = 30.0

[agent]
timeout_sec = 30.0

[environment]
build_timeout_sec = 30.0
cpus = 1
memory_mb = 512
storage_mb = 1024
gpus = 0
"""
    )
    task_dir.joinpath("instruction.md").write_text(instruction)
    task_dir.joinpath("environment", "Dockerfile").write_text("FROM ubuntu:24.04\n")
    task_dir.joinpath("tests", "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
