"""Import-shape guards for benchmax.sft.

The serialization gate makes `dataset` depend on `schema`, and `schema` needs
the canonical byte rendering that used to live in `dataset` — a reverse
top-level import would cycle. `serialization` exists to break that, and these
tests are what keeps it broken.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import benchmax.sft
import pytest

SFT_PACKAGE = Path(benchmax.sft.__file__).parent
SFT_MODULES = sorted(path.stem for path in SFT_PACKAGE.glob("*.py") if path.stem != "__init__")


def _imported_module_names(source_path: Path) -> set[str]:
    """Every absolute module name imported by ``source_path``, dotted paths intact."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_sft_modules_are_discovered() -> None:
    # the parametrized tests below are vacuous if the glob finds nothing
    assert set(SFT_MODULES) == {"dataset", "normalize", "schema", "serialization", "validate"}


@pytest.mark.parametrize("module", SFT_MODULES)
def test_module_imports_cleanly_as_the_first_import(module: str) -> None:
    """Each submodule imports in a fresh interpreter with nothing else loaded.

    A cycle only shows up for whichever module the process happens to reach
    first, so importing each one first is the check that actually catches it.
    """
    completed = subprocess.run(
        [sys.executable, "-c", f"import benchmax.sft.{module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_schema_does_not_import_dataset() -> None:
    # dataset imports schema for its serialization gate; the reverse edge is
    # the cycle that neutral `serialization` module exists to prevent.
    imported = _imported_module_names(SFT_PACKAGE / "schema.py")
    assert "benchmax.sft.dataset" not in imported
    assert "benchmax.sft.serialization" in imported


def test_dataset_enforces_the_schema_gate_at_import_level() -> None:
    imported = _imported_module_names(SFT_PACKAGE / "dataset.py")
    assert "benchmax.sft.schema" in imported


@pytest.mark.parametrize("module", SFT_MODULES)
def test_sft_depends_only_on_stdlib_and_benchmax(module: str) -> None:
    """No third-party dependency, and nothing from castform.

    Keeping this package importable from the standard library alone is what
    lets a CLI reach the dataset contract without paying for the rest of the
    environment stack, and is what makes it legal in `packages/benchmax` at
    all (see tests/architecture/test_package_boundary.py).
    """
    roots = {name.split(".")[0] for name in _imported_module_names(SFT_PACKAGE / f"{module}.py")}
    unexpected = roots - sys.stdlib_module_names - {"benchmax"}
    assert not unexpected, f"benchmax.sft.{module} imports {sorted(unexpected)}"
