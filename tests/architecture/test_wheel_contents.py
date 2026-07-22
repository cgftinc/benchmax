from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("package_dir", "module_name", "retired_paths"),
    [
        (
            "packages/benchmax",
            "benchmax",
            {
                "benchmax/envs/reward_helpers.py",
                "benchmax/prompts/__init__.py",
                "benchmax/rubrics/__init__.py",
            },
        ),
        (
            "packages/castform",
            "castform",
            {
                "castform/cli/corpus.py",
                "castform/cli/data.py",
                "castform/cli/dataview.py",
                "castform/cli/launch.py",
                "castform/cli/templates/viewer.html",
            },
        ),
    ],
)
def test_direct_wheel_build_excludes_stale_build_tree(
    tmp_path: Path,
    package_dir: str,
    module_name: str,
    retired_paths: set[str],
) -> None:
    """Deleted modules cannot leak from a persistent build cache."""

    workspace = Path(__file__).resolve().parents[2]
    project = tmp_path / module_name
    shutil.copytree(
        workspace / package_dir,
        project,
        ignore=shutil.ignore_patterns(
            "build",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
        ),
    )
    sentinel = f"{module_name}/removed_build_cache_module.py"
    stale_file = project / "build" / "lib" / sentinel
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("SHOULD_NOT_SHIP = True\n")

    output = tmp_path / "dist"
    completed = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(output),
            str(project),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_paths = set(archive.namelist())
    assert sentinel not in wheel_paths
    assert retired_paths.isdisjoint(wheel_paths)
    if module_name == "castform":
        assert "castform/cli/scaffold/STARTER.md" in wheel_paths
        assert "castform/cli/scaffold/skills/launch-run/SKILL.md" in wheel_paths
