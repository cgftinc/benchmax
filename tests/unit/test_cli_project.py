"""`_project.py` loader behavior — the import-safe `main.py` entrypoint."""

from __future__ import annotations

import pytest

# A minimal env whose module body ends in a `__main__` guard that would raise if
# it ran on import. The loader execs the file under its stem ("main"), never
# "__main__", so the guard body is skipped — this is what makes `python main.py`
# a safe entrypoint that the CLI can still import.
_GUARDED_MAIN = """\
from typing import Any

from benchmax.envs.base_env import BaseEnv


class MyEnv(BaseEnv):
    async def list_tools(self):
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return ""

    async def compute_reward(self, rollout_id, messages, task, **kwargs):
        return {"reward": 1.0}


if __name__ == "__main__":
    raise SystemExit("main.py body must not run on loader import")
"""


def test_loader_skips_main_guard_import_safe(tmp_path):
    """A `main.py` with an `if __name__ == "__main__":` block loads via the loader
    WITHOUT running that block. Regression guard for the run.py→main.py rename: the
    exec stem drives sys.modules + pickle name resolution, and a fired guard would
    raise SystemExit (BaseException — not caught by the loader's `except Exception`),
    crashing load_project."""
    from benchmax.cli._project import load_project

    (tmp_path / "main.py").write_text(_GUARDED_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text(
        '{"prompt": "q", "ground_truth": "a"}\n'
    )

    project = load_project(directory=str(tmp_path))  # SystemExit here if guard fired
    assert project.env_class.__name__ == "MyEnv"
    # Exec'd under the file stem — this "main" (not "__main__") is why the guard is
    # skipped, and it's the module name pickled by value into the sandbox bundle.
    assert project.module.__name__ == "main"


def test_loader_discovers_env_via_load_module(tmp_path):
    """`_load_module_from_file` + `discover_env_class` — the same path the scaffold
    tests use — resolves the single BaseEnv subclass under the new stem."""
    from benchmax.cli._project import _load_module_from_file, discover_env_class

    (tmp_path / "main.py").write_text(_GUARDED_MAIN)
    mod = _load_module_from_file(tmp_path / "main.py")
    assert mod.__name__ == "main"
    assert discover_env_class(mod).__name__ == "MyEnv"


# --- TRAINING_MODE contract (slice 5) -------------------------------------

_ENV_MAIN = """\
from benchmax.envs.base_env import BaseEnv


class MyEnv(BaseEnv):
    async def list_tools(self):
        return []

    async def run_tool(self, rollout_id, tool_name, **tool_args):
        return ""

    async def compute_reward(self, rollout_id, messages, task, **kwargs):
        return {"reward": 1.0}
"""

_NO_ENV_MAIN = "x = 1\n"


def _write_sft_project(tmp_path, *, main_body=_NO_ENV_MAIN, train="{}\n", eval=None):
    (tmp_path / "main.py").write_text('TRAINING_MODE = "sft"\n\n' + main_body)
    (tmp_path / "train_dataset.jsonl").write_text(train)
    if eval is not None:
        (tmp_path / "eval_dataset.jsonl").write_text(eval)
    return tmp_path


def test_no_marker_defaults_to_rl_env_required(tmp_path):
    """No TRAINING_MODE marker -> rl, and today's error-on-missing-env is
    preserved (env still required)."""
    from benchmax.cli._project import ProjectError, load_project

    (tmp_path / "main.py").write_text(_NO_ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    with pytest.raises(ProjectError, match="No BaseEnv"):
        load_project(directory=str(tmp_path))


def test_no_marker_with_env_loads_as_rl(tmp_path):
    from benchmax.cli._project import load_project

    (tmp_path / "main.py").write_text(_ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    project = load_project(directory=str(tmp_path))
    assert project.training_mode == "rl"
    assert project.env_class.__name__ == "MyEnv"


def test_explicit_rl_marker_same_as_no_marker(tmp_path):
    from benchmax.cli._project import load_project

    (tmp_path / "main.py").write_text('TRAINING_MODE = "rl"\n\n' + _ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    project = load_project(directory=str(tmp_path))
    assert project.training_mode == "rl"
    assert project.env_class.__name__ == "MyEnv"


def test_unknown_training_mode_marker_raises(tmp_path):
    from benchmax.cli._project import ProjectError, load_project

    (tmp_path / "main.py").write_text('TRAINING_MODE = "supervised"\n\n' + _NO_ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    with pytest.raises(ProjectError, match="invalid TRAINING_MODE 'supervised'"):
        load_project(directory=str(tmp_path))


def test_training_mode_none_raises_not_treated_as_absent(tmp_path):
    """A present TRAINING_MODE = None must NOT be treated the same as an absent
    marker (which defaults to rl) -- None is present-but-invalid."""
    from benchmax.cli._project import ProjectError, load_project

    (tmp_path / "main.py").write_text("TRAINING_MODE = None\n\n" + _NO_ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    with pytest.raises(ProjectError) as exc_info:
        load_project(directory=str(tmp_path))
    assert "invalid TRAINING_MODE" in str(exc_info.value)


def test_training_mode_unhashable_list_raises_project_error_not_type_error(tmp_path):
    """An unhashable TRAINING_MODE value must raise ProjectError, not a raw
    TypeError from a hash/equality check against the allowed set."""
    from benchmax.cli._project import ProjectError, load_project

    (tmp_path / "main.py").write_text("TRAINING_MODE = []\n\n" + _NO_ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    with pytest.raises(ProjectError) as exc_info:
        load_project(directory=str(tmp_path))
    assert "invalid TRAINING_MODE" in str(exc_info.value)


def test_training_mode_unhashable_dict_raises_project_error_not_type_error(tmp_path):
    from benchmax.cli._project import ProjectError, load_project

    (tmp_path / "main.py").write_text("TRAINING_MODE = {}\n\n" + _NO_ENV_MAIN)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    with pytest.raises(ProjectError) as exc_info:
        load_project(directory=str(tmp_path))
    assert "invalid TRAINING_MODE" in str(exc_info.value)


def test_training_mode_unhashable_str_subclass_raises_project_error_not_type_error(
    tmp_path,
):
    """A str SUBCLASS instance (even one spelling "sft") must still raise
    ProjectError, not TypeError, when it's unhashable -- an exact-str check is
    required, not isinstance, since the frozenset `in` check below would
    otherwise try to hash it."""
    from benchmax.cli._project import ProjectError, load_project

    main_body = (
        "class BadMode(str):\n"
        "    __hash__ = None\n\n"
        'TRAINING_MODE = BadMode("sft")\n\n' + _NO_ENV_MAIN
    )
    (tmp_path / "main.py").write_text(main_body)
    (tmp_path / "train_dataset.jsonl").write_text('{"prompt": "q"}\n')
    with pytest.raises(ProjectError) as exc_info:
        load_project(directory=str(tmp_path))
    assert "invalid TRAINING_MODE" in str(exc_info.value)


def test_sft_marker_skips_env_discovery(tmp_path):
    from benchmax.cli._project import load_project

    _write_sft_project(tmp_path, train='{"messages": [{"role": "user", "content": "hi"}]}\n')
    project = load_project(directory=str(tmp_path))
    assert project.training_mode == "sft"
    assert project.env_class is None
    assert project.sft_train_path == tmp_path / "train_dataset.jsonl"
    assert project.sft_eval_path is None


def test_sft_marker_and_env_class_is_ambiguous(tmp_path):
    """TRAINING_MODE = "sft" AND a BaseEnv subclass in the same main.py is an
    ambiguous project — both signals present."""
    from benchmax.cli._project import ProjectError, load_project

    _write_sft_project(tmp_path, main_body=_ENV_MAIN)
    with pytest.raises(ProjectError, match="ambiguous project"):
        load_project(directory=str(tmp_path))


def test_sft_marker_and_imported_env_class_is_ambiguous(tmp_path):
    """An env class merely IMPORTED into main.py's namespace (not locally
    defined) still counts as "present" for the sft-ambiguity guard -- only RL
    auto-discovery restricts itself to classes defined in the file itself."""
    from benchmax.cli._project import ProjectError, load_project

    _write_sft_project(
        tmp_path,
        main_body="from benchmax.envs.sft_demonstration_env import SftDemonstrationEnv\n",
        train='{"messages": []}\n',
    )
    with pytest.raises(ProjectError, match="ambiguous project"):
        load_project(directory=str(tmp_path))


def test_env_class_flag_combined_with_sft_mode_raises(tmp_path):
    from benchmax.cli._project import ProjectError, load_project

    _write_sft_project(tmp_path, train='{"messages": []}\n')
    with pytest.raises(ProjectError, match="--env-class can't be combined"):
        load_project(directory=str(tmp_path), env_class_name="Whatever")


def test_sft_mode_never_parses_dataset_malformed_json_does_not_raise(tmp_path):
    """`_load_jsonl` must NOT be invoked for sft data — a malformed JSON line
    in the dataset must not raise ProjectError; it's a report issue instead
    (surfaced later by sft.dataset.load_sft_dataset / validate_sft_dataset)."""
    from benchmax.cli._project import load_project

    _write_sft_project(tmp_path, train="not valid json at all\n")
    project = load_project(directory=str(tmp_path))  # must not raise
    assert project.training_mode == "sft"
    assert project.train_dataset == []  # never parsed by the loader
    assert project.sft_train_path == tmp_path / "train_dataset.jsonl"


def test_sft_mode_missing_train_dataset_raises(tmp_path):
    from benchmax.cli._project import ProjectError, load_project

    (tmp_path / "main.py").write_text('TRAINING_MODE = "sft"\n\n' + _NO_ENV_MAIN)
    with pytest.raises(ProjectError, match="dataset not found"):
        load_project(directory=str(tmp_path))


def test_sft_mode_eval_optional_by_default(tmp_path):
    from benchmax.cli._project import load_project

    _write_sft_project(tmp_path, train="{}\n")
    project = load_project(directory=str(tmp_path))
    assert project.sft_eval_path is None


def test_sft_mode_eval_path_recorded_when_present(tmp_path):
    from benchmax.cli._project import load_project

    _write_sft_project(tmp_path, train="{}\n", eval="{}\n")
    project = load_project(directory=str(tmp_path))
    assert project.sft_eval_path == tmp_path / "eval_dataset.jsonl"


def test_sft_mode_require_eval_raises_when_missing(tmp_path):
    from benchmax.cli._project import ProjectError, load_project

    _write_sft_project(tmp_path, train="{}\n")
    with pytest.raises(ProjectError, match="eval dataset required"):
        load_project(directory=str(tmp_path), require_eval=True)
