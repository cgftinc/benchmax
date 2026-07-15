"""`_project.py` loader behavior — the import-safe `main.py` entrypoint."""

from __future__ import annotations

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
