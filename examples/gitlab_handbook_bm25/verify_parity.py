#!/usr/bin/env -S uv run --isolated --script
# /// script
# requires-python = "==3.12.*"
# dependencies = ["pyyaml>=6.0.2"]
# ///
"""Static checks for the intentionally equal parts of the two A/B arms."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CASTFORM_ROOT = ROOT.parents[3]
ARMS = ("pre_harbor", "post_harbor")
TRAINER_TASK_PATH = "core/trainer/src/trainer/customer/self_serve/simple.yaml"
CONSTANTS = (
    "CORPUS_NAME",
    "JUDGE_MODEL",
    "MAX_SEARCH_CALLS",
    "W_CORRECTNESS",
    "W_CONCISENESS",
    "W_CITATION_RECALL",
    "W_CITATION_GROUNDING",
    "W_RETRIEVAL_HIT",
    "FULL_CONTENT_RANKS",
    "FULL_CHUNK_CHARS",
    "SNIPPET_CHARS",
    "TOOL_OUTPUT_CHARS",
    "REWARD_KEYS",
)
COMMON_ENVS = {
    "ARG_MODEL": "Qwen/Qwen3.5-4B",
    "ARG_LR": "4e-6",
    "ARG_NUM_EPOCH": "5",
    "ARG_N_SAMPLES_PER_PROMPT": "9",
    "ARG_N_SAMPLES_PER_EVAL_PROMPT": "9",
    "ARG_ROLLOUT_MAX_CONTEXT_LEN": "16384",
    "ARG_EVAL_MAX_CONTEXT_LEN": "16384",
    "ARG_EVAL_INTERVAL": "5",
    "ARG_SAVE_INTERVAL": "5",
    "ARG_KL_LOSS_COEF": "0",
    "ARG_LORA_RANK": "128",
    "ARG_LORA_ALPHA": "256",
}
TASK_NAMES = {
    "pre_harbor": "gitlab-bm25-ab-pre-harbor-no-gateway",
    "post_harbor": "gitlab-bm25-ab-post-harbor-gateway-pre-harbor-effective-3-tools",
}


def _assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        value = node.value
        if isinstance(target, ast.Name) and value is not None:
            try:
                values[target.id] = ast.literal_eval(value)
            except (TypeError, ValueError):
                pass
    return values


def _literal_dict_fields(path: Path, assignment_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name) or target.id != assignment_name:
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError(f"{path}: {assignment_name} must be a dict literal")
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values, strict=True):
            try:
                key = ast.literal_eval(key_node)
                value = ast.literal_eval(value_node)
            except (TypeError, ValueError):
                continue
            if isinstance(key, str):
                result[key] = value
        return result
    raise AssertionError(f"{path}: missing {assignment_name}")


def _task(arm: str) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / arm / "task.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{arm}: task.yaml must contain one mapping")
    return value


def _source_task(ref: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "show", f"{ref}:{TRAINER_TASK_PATH}"],
        cwd=CASTFORM_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = yaml.safe_load(result.stdout)
    if not isinstance(value, dict):
        raise AssertionError(f"{ref}:{TRAINER_TASK_PATH} must contain one mapping")
    return value


def _benchmax_ref(ref: str) -> str:
    result = subprocess.run(
        ["git", "ls-tree", ref, "core/benchmax"],
        cwd=CASTFORM_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.split()
    if len(fields) != 4 or fields[1] != "commit":
        raise AssertionError(f"{ref}: missing core/benchmax gitlink")
    return fields[2]


def main() -> None:
    spec = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
    assignments = {arm: _assignments(ROOT / arm / "environment.py") for arm in ARMS}
    launch_configs = {
        arm: _literal_dict_fields(ROOT / arm / "environment.py", "LAUNCH_CONFIG") for arm in ARMS
    }
    for name in CONSTANTS:
        left = assignments["pre_harbor"].get(name)
        right = assignments["post_harbor"].get(name)
        assert left == right, f"environment constant drift: {name}: {left!r} != {right!r}"

    assert launch_configs["pre_harbor"]["max_context_len"] == 16384
    assert launch_configs["post_harbor"]["max_context_tokens"] == 16384
    assert launch_configs["pre_harbor"]["num_epochs"] == 5
    assert launch_configs["post_harbor"]["num_epochs"] == 5

    tasks = {arm: _task(arm) for arm in ARMS}
    for arm, task in tasks.items():
        assert task["name"] == TASK_NAMES[arm], f"{arm}: task name drifted"
        envs = task["envs"]
        for name, expected in COMMON_ENVS.items():
            assert str(envs.get(name)) == expected, f"{arm}: {name} drifted"
        assert all(value is not None for value in envs.values()), f"{arm}: null env value"
        assert all(isinstance(value, str) for value in task["secrets"].values()), (
            f"{arm}: secrets must be string placeholders"
        )
        assert not any(
            key in task["envs"] and task["envs"][key] for key in ("WANDB_API_KEY", "GIT_TOKEN")
        ), f"{arm}: secret value leaked into envs"
        expected_ref = spec["castform_refs"][arm]
        assert task["workdir"]["ref"] == expected_ref, f"{arm}: workdir ref drifted"
        source_task = _source_task(expected_ref)
        for field in ("resources", "setup", "run"):
            assert task[field] == source_task[field], (
                f"{arm}: {field} drifted from {expected_ref}:{TRAINER_TASK_PATH}"
            )
        assert task["workdir"]["url"] == source_task["workdir"]["url"], (
            f"{arm}: workdir URL drifted from source task"
        )
        assert _benchmax_ref(expected_ref) == spec["benchmax_refs"][arm], (
            f"{arm}: Benchmax ref does not match its Castform gitlink"
        )

    pre_envs = tasks["pre_harbor"]["envs"]
    assert pre_envs["ARG_MAX_TURNS"] == "5"
    assert pre_envs["ARG_MAX_TOOL_CALLS"] == "4"
    assert "ARG_MAX_TURNS" not in tasks["post_harbor"]["envs"]
    assert assignments["post_harbor"]["MAX_SEARCH_CALLS"] + 1 == 5
    assert assignments["post_harbor"]["EFFECTIVE_MAX_TOOL_CALLS"] == 3

    pre_run = tasks["pre_harbor"]["run"]
    post_run = tasks["post_harbor"]["run"]
    assert "ensure_platform_job_env.py" not in pre_run
    assert "ensure_platform_job_env.py" in post_run
    assert "USER_DATA_ARG_TRAIN_DATASET_PATH" in pre_run
    assert "USER_DATA_ARG_EVAL_DATASET_PATH" in pre_run
    assert "USER_DATA_ARG_DATASET_PATH" in post_run
    assert "trainer.tools.run_training" in pre_run
    assert "trainer.tools.run_training" in post_run

    contracts = [ROOT / arm / "artifacts" / "contract.json" for arm in ARMS]
    if all(path.is_file() for path in contracts):
        pre_contract = json.loads(contracts[0].read_text(encoding="utf-8"))
        post_contract = json.loads(contracts[1].read_text(encoding="utf-8"))
        assert pre_contract == post_contract, "bundled prompt/tool/format contract drifted"

    print("parity checks passed")


if __name__ == "__main__":
    main()
