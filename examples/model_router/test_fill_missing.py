import json
from pathlib import Path

from fill_missing import coverage_state, materialize_fill_set


def make_task(farm: Path, task_id: str) -> None:
    task = farm / task_id
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "payload.txt").write_text(task_id)


def make_result(
    runs: Path,
    jobs_dir: str,
    task_id: str,
    *,
    reward: int | float | None,
    exception_type: str | None = None,
) -> None:
    trial = runs / jobs_dir / f"{task_id}__trial"
    trial.mkdir(parents=True)
    result = {"verifier_result": {"rewards": {"reward": reward}}}
    if exception_type:
        result["exception_info"] = {"exception_type": exception_type}
    (trial / "result.json").write_text(json.dumps(result))


def test_coverage_scans_suffixed_runs_and_filters_infrastructure(
    tmp_path: Path,
) -> None:
    farm = tmp_path / "farm"
    runs = tmp_path / "runs"
    for task_id in ("clean", "infra", "timeout", "unseen"):
        make_task(farm, task_id)

    make_result(runs, "farm-model-wave2", "clean", reward=1)
    make_result(
        runs,
        "farm-model-retry",
        "infra",
        reward=0,
        exception_type="NetworkConnectionError",
    )
    make_result(
        runs,
        "farm-model",
        "timeout",
        reward=0,
        exception_type="AgentTimeoutError",
    )
    # A similarly named model must not leak into coverage.
    make_result(runs, "farm-model2", "unseen", reward=1)

    missing, done, retired, attempts = coverage_state(
        "model", farm=farm, runs_dir=runs, max_attempts=3
    )

    assert missing == ["infra", "unseen"]
    assert done == {"clean", "timeout"}
    assert retired == set()
    assert attempts == {"clean": 1, "infra": 1, "timeout": 1}


def test_coverage_retires_repeated_infrastructure_failures(tmp_path: Path) -> None:
    farm = tmp_path / "farm"
    runs = tmp_path / "runs"
    make_task(farm, "task")
    make_result(
        runs,
        "farm-model",
        "task",
        reward=0,
        exception_type="AuthenticationError",
    )

    missing, done, retired, attempts = coverage_state(
        "model", farm=farm, runs_dir=runs, max_attempts=1
    )

    assert missing == []
    assert done == set()
    assert retired == {"task"}
    assert attempts == {"task": 1}


def test_materialize_fill_set_replaces_target_with_exact_tasks(
    tmp_path: Path,
) -> None:
    farm = tmp_path / "farm"
    for task_id in ("a", "b", "c"):
        make_task(farm, task_id)
    target = tmp_path / "fill"
    make_task(target, "stale")

    materialize_fill_set(farm, target, ["a", "c"])

    assert sorted(path.name for path in target.iterdir()) == ["a", "c"]
    assert (target / "a" / "payload.txt").read_text() == "a"
    assert (target / "c" / "payload.txt").read_text() == "c"
