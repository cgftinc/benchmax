from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
BENCHMAX_ROOT = ROOT / "packages" / "benchmax"
BENCHMAX_SOURCE = BENCHMAX_ROOT / "src" / "benchmax"
CASTFORM_ROOT = ROOT / "packages" / "castform"


def test_benchmax_has_no_castform_dependency_or_platform_knowledge() -> None:
    violations: list[str] = []
    forbidden_text = ("CASTFORM_", "castform.com", "~/.castform", "harbor-castform")

    for source_file in BENCHMAX_SOURCE.rglob("*.py"):
        source = source_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "castform" or name.startswith("castform.") for name in names):
                violations.append(f"{source_file}: imports Castform")
        for marker in forbidden_text:
            if marker in source:
                violations.append(f"{source_file}: contains {marker!r}")

    manifest = tomllib.loads((BENCHMAX_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = manifest["project"]
    dependencies = project.get("dependencies", [])
    if any(str(dep).lower().startswith("castform") for dep in dependencies):
        violations.append("benchmax distribution depends on Castform")
    if "castform" in project.get("scripts", {}):
        violations.append("benchmax distribution defines the Castform CLI")

    assert not violations, "\n".join(violations)


def test_workspace_dependency_direction_and_example_manifests() -> None:
    castform = tomllib.loads((CASTFORM_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = castform["project"].get("dependencies", [])
    assert any(str(dep).lower().startswith("benchmax") for dep in dependencies)

    example_manifests = sorted((ROOT / "examples").glob("*/pyproject.toml"))
    example_directories = sorted(path for path in (ROOT / "examples").iterdir() if path.is_dir())
    assert [path.parent for path in example_manifests] == example_directories
    for manifest_path in example_manifests:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["project"].get("dependencies"), manifest_path
        sources = manifest.get("tool", {}).get("uv", {}).get("sources", {})
        workspace_sources = [
            name
            for name, source in sources.items()
            if isinstance(source, dict) and source.get("workspace") is True
        ]
        assert not workspace_sources, (
            f"{manifest_path} is tied to workspace sources {workspace_sources}; "
            "development overrides belong in the workspace root"
        )
