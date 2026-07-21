"""Tests for env source capture in dump_bundle.

Code generators can build env classes via ``exec()`` in an in-memory namespace, so
``inspect.getsource`` can't recover their text. ``dump_bundle`` accepts an
``env_class_source`` override for exactly this case; these tests pin both the
override path and the introspection gap it exists to paper over.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

from benchmax.bundle import (
    BundlingError,
    _get_source,
    _module_is_project_local,
    _project_root_for_module,
    dump_bundle,
)
from benchmax.envs import BaseEnv

# Defined at module scope so cloudpickle can pickle it by-value when the test
# module is registered as a local module (dump_bundle enforces this).
_TEST_MODULE = sys.modules[__name__]


class MinimalEnv(BaseEnv):
    """Minimal valid BaseEnv subclass for bundling tests."""

    reward_keys = ("score",)

    async def create_dataset(self, split, base_dir):
        raise NotImplementedError

    async def compute_reward(self, *args, **kwargs):
        return {"score": 0.0}


def test_source_introspected_by_default() -> None:
    """With no override, source is recovered from the class via inspect."""
    bundle = dump_bundle(MinimalEnv, local_modules=[_TEST_MODULE])
    assert bundle.metadata.env_class_source is not None
    assert "class MinimalEnv" in bundle.metadata.env_class_source


def test_source_override_wins_over_introspection() -> None:
    """An explicit override is recorded verbatim, even when introspection
    would have succeeded."""
    bundle = dump_bundle(
        MinimalEnv,
        local_modules=[_TEST_MODULE],
        env_class_source="# handed in by the caller\nclass Whatever: ...\n",
    )
    assert bundle.metadata.env_class_source == (
        "# handed in by the caller\nclass Whatever: ...\n"
    )


def test_exec_defined_class_has_no_introspectable_source() -> None:
    """Regression: a class produced by exec() has no
    source file, so _get_source returns None — which is why the override
    parameter is needed."""
    source_code = (
        "from benchmax.envs import BaseEnv\n"
        "class GeneratedEnv(BaseEnv):\n"
        "    reward_keys = ('score',)\n"
        "    async def create_dataset(self, *a, **k): raise NotImplementedError\n"
        "    async def compute_reward(self, *a, **k): return {'score': 0.0}\n"
    )
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__modal_generated_env__",
    }
    exec(source_code, namespace)
    env_class = namespace["GeneratedEnv"]

    assert _get_source(env_class) is None


def test_auto_captures_editable_src_project_and_recursive_module_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An editable install is local source, not a rollout-image dependency.

    The distribution metadata deliberately makes the package look installed in
    the bundling interpreter. The loaded bundle must still work in a clean
    subprocess where the project source is absent from ``sys.path``. Calling
    ``marker`` also proves that the local helper module referenced by the env
    module was captured recursively.
    """

    module_name = "editable_src_env"
    source_root = _write_editable_project(tmp_path, module_name, src_layout=True)
    monkeypatch.syspath_prepend(str(source_root))
    env_module = __import__(f"{module_name}.env", fromlist=["EditableEnv"])

    bundle = dump_bundle(env_module.EditableEnv)
    bundle_file = tmp_path / "bundle.pickle"
    bundle_file.write_bytes(pickle.dumps(bundle))
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    child_code = textwrap.dedent(
        """
        import importlib.util
        import pickle
        import sys
        from pathlib import Path

        from benchmax.bundle import load_bundle

        bundle_path, project_source, module_name = sys.argv[1:]
        assert project_source not in sys.path
        assert importlib.util.find_spec(module_name) is None
        bundle = pickle.loads(Path(bundle_path).read_bytes())
        env = load_bundle(bundle)
        assert env.marker() == "recursive-local-reference"
        """
    )
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(bundle_file),
            str(source_root),
            module_name,
        ],
        cwd=clean_dir,
        env=child_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_strict_mode_rejects_editable_project_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode reports local source even when it has dist metadata."""

    module_name = "strict_editable_env"
    source_root = _write_editable_single_file_project(tmp_path, module_name)
    monkeypatch.syspath_prepend(str(source_root))
    env_module = __import__(module_name, fromlist=["EditableEnv"])

    with pytest.raises(BundlingError, match=rf"local_modules=.*{module_name}"):
        dump_bundle(env_module.EditableEnv, auto_local_modules=False)


def test_auto_captures_editable_single_file_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-file projects receive the same clean-load guarantee as packages."""

    module_name = "single_file_env"
    source_root = _write_editable_single_file_project(tmp_path, module_name)
    monkeypatch.syspath_prepend(str(source_root))
    env_module = __import__(module_name, fromlist=["EditableEnv"])

    bundle = dump_bundle(env_module.EditableEnv)
    bundle_file = tmp_path / "single-file-bundle.pickle"
    bundle_file.write_bytes(pickle.dumps(bundle))
    clean_dir = tmp_path / "single-file-clean"
    clean_dir.mkdir()
    child_code = textwrap.dedent(
        """
        import importlib.util
        import pickle
        import sys
        from pathlib import Path

        from benchmax.bundle import load_bundle

        bundle_path, project_source, module_name = sys.argv[1:]
        assert project_source not in sys.path
        assert importlib.util.find_spec(module_name) is None
        env = load_bundle(pickle.loads(Path(bundle_path).read_bytes()))
        assert type(env).__name__ == "EditableEnv"
        """
    )
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(bundle_file),
            str(source_root),
            module_name,
        ],
        cwd=clean_dir,
        env=child_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_rejects_undeclared_sibling_project_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second project must be explicitly captured or installed remotely."""

    _clear_sibling_modules()
    env_source, sibling_source = _write_sibling_projects(tmp_path)
    monkeypatch.syspath_prepend(str(sibling_source))
    monkeypatch.syspath_prepend(str(env_source))
    env_module = __import__("sibling_env.env", fromlist=["SiblingEnv"])

    with pytest.raises(BundlingError, match="sibling_helpers"):
        dump_bundle(env_module.SiblingEnv)


def test_allows_sibling_project_declared_as_remote_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sibling_modules()
    env_source, sibling_source = _write_sibling_projects(tmp_path)
    monkeypatch.syspath_prepend(str(sibling_source))
    monkeypatch.syspath_prepend(str(env_source))
    env_module = __import__("sibling_env.env", fromlist=["SiblingEnv"])

    bundle = dump_bundle(
        env_module.SiblingEnv,
        pip_dependencies=["sibling-helpers==1.0.0"],
    )

    assert bundle.metadata.pip_dependencies == ["sibling-helpers==1.0.0"]


def test_explicitly_captures_sibling_project_for_clean_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sibling_modules()
    env_source, sibling_source = _write_sibling_projects(tmp_path)
    monkeypatch.syspath_prepend(str(sibling_source))
    monkeypatch.syspath_prepend(str(env_source))
    env_module = __import__("sibling_env.env", fromlist=["SiblingEnv"])
    helper_module = __import__("sibling_helpers.helper", fromlist=["VALUE"])

    bundle = dump_bundle(
        env_module.SiblingEnv,
        local_modules=[helper_module],
    )
    bundle_file = tmp_path / "sibling-bundle.pickle"
    bundle_file.write_bytes(pickle.dumps(bundle))
    clean_dir = tmp_path / "sibling-clean"
    clean_dir.mkdir()
    child_code = textwrap.dedent(
        """
        import importlib.util
        import pickle
        import sys
        from pathlib import Path

        from benchmax.bundle import load_bundle

        bundle_path, env_source, sibling_source = sys.argv[1:]
        assert env_source not in sys.path
        assert sibling_source not in sys.path
        assert importlib.util.find_spec("sibling_env") is None
        assert importlib.util.find_spec("sibling_helpers") is None
        env = load_bundle(pickle.loads(Path(bundle_path).read_bytes()))
        assert type(env).__name__ == "SiblingEnv"
        """
    )
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(bundle_file),
            str(env_source),
            str(sibling_source),
        ],
        cwd=clean_dir,
        env=child_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_framework_stdlib_and_site_packages_are_not_project_local() -> None:
    """Only the environment project's own source is captured by value."""

    project_root = _project_root_for_module(_TEST_MODULE)
    assert project_root is not None
    roots = (project_root,)

    assert _module_is_project_local(__name__, roots)
    assert not _module_is_project_local("benchmax.bundle", roots)
    assert not _module_is_project_local("pathlib", roots)
    # The workspace virtualenv is below the project root, so this assertion
    # specifically guards against treating nested site-packages as source.
    assert not _module_is_project_local("openai", roots)


def _write_editable_project(
    tmp_path: Path,
    module_name: str,
    *,
    src_layout: bool,
) -> Path:
    """Write an importable project that distribution metadata sees as installed."""

    project_root = tmp_path / module_name
    source_root = project_root / "src" if src_layout else project_root
    package_dir = source_root / module_name
    package_dir.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "{module_name.replace("_", "-")}"
            version = "1.0.0"
            """
        )
    )
    (package_dir / "__init__.py").write_text("")
    (package_dir / "helper.py").write_text('VALUE = "recursive-local-reference"\n')
    (package_dir / "env.py").write_text(
        textwrap.dedent(
            """
            from benchmax.envs import BaseEnv
            from . import helper

            class EditableEnv(BaseEnv):
                reward_keys = ("score",)

                async def create_dataset(self, split, base_dir):
                    raise NotImplementedError

                async def compute_reward(self, *args, **kwargs):
                    return {"score": 0.0}

                def marker(self):
                    return helper.VALUE
            """
        )
    )

    dist_info = source_root / f"{module_name}-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {module_name.replace('_', '-')}\n"
        "Version: 1.0.0\n"
    )
    (dist_info / "top_level.txt").write_text(f"{module_name}\n")
    return source_root


def _write_editable_single_file_project(tmp_path: Path, module_name: str) -> Path:
    """Write a single-module project that also looks installed locally."""

    project_root = tmp_path / module_name
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "{module_name.replace("_", "-")}"
            version = "1.0.0"
            """
        )
    )
    (project_root / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            from benchmax.envs import BaseEnv

            class EditableEnv(BaseEnv):
                reward_keys = ("score",)

                async def create_dataset(self, split, base_dir):
                    raise NotImplementedError

                async def compute_reward(self, *args, **kwargs):
                    return {"score": 0.0}
            """
        )
    )
    dist_info = project_root / f"{module_name}-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {module_name.replace('_', '-')}\n"
        "Version: 1.0.0\n"
    )
    (dist_info / "top_level.txt").write_text(f"{module_name}\n")
    return project_root


def _write_sibling_projects(tmp_path: Path) -> tuple[Path, Path]:
    env_project = tmp_path / "env-project"
    env_source = env_project / "src"
    env_package = env_source / "sibling_env"
    env_package.mkdir(parents=True)
    (env_project / "pyproject.toml").write_text(
        '[project]\nname = "sibling-env"\nversion = "1.0.0"\n'
    )
    (env_package / "__init__.py").write_text("")
    (env_package / "env.py").write_text(
        textwrap.dedent(
            """
            from benchmax.envs import BaseEnv
            from sibling_helpers import helper

            class SiblingEnv(BaseEnv):
                reward_keys = ("score",)

                async def create_dataset(self, split, base_dir):
                    raise NotImplementedError

                async def compute_reward(self, *args, **kwargs):
                    return {"score": float(helper.VALUE == "sibling")}
            """
        )
    )

    sibling_project = tmp_path / "helper-project"
    sibling_source = sibling_project / "src"
    sibling_package = sibling_source / "sibling_helpers"
    sibling_package.mkdir(parents=True)
    (sibling_project / "pyproject.toml").write_text(
        '[project]\nname = "sibling-helpers"\nversion = "1.0.0"\n'
    )
    (sibling_package / "__init__.py").write_text("")
    (sibling_package / "helper.py").write_text('VALUE = "sibling"\n')
    dist_info = sibling_source / "sibling_helpers-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: sibling-helpers\nVersion: 1.0.0\n"
    )
    (dist_info / "top_level.txt").write_text("sibling_helpers\n")
    return env_source, sibling_source


def _clear_sibling_modules() -> None:
    for name in tuple(sys.modules):
        if name == "sibling_env" or name.startswith("sibling_env."):
            sys.modules.pop(name, None)
        if name == "sibling_helpers" or name.startswith("sibling_helpers."):
            sys.modules.pop(name, None)
