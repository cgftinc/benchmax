import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition
from benchmax.bundle.bundler import bundle_env, read_bundle_files, write_bundle_files
from benchmax.bundle.errors import (
    IncompatibleBenchmaxError,
    IncompatiblePythonError,
    BundlingError,
    ValidationError,
)
from benchmax.bundle.loader import load_env, load_env_from_files
from benchmax.bundle.payload import BundleMetadata, BundledEnv
from benchmax.bundle.validator import validate_bundle, validate_structure


# ---------------------------------------------------------------------------
# Helpers: minimal concrete BaseEnv subclass
# ---------------------------------------------------------------------------


class MinimalEnv(BaseEnv):
    """Minimal valid BaseEnv subclass for testing."""

    system_prompt = "You are a test env."

    def __init__(self, greeting: str = "hello"):
        self.greeting = greeting

    async def shutdown(self):
        pass

    async def list_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="echo",
                description="Echo a message",
                input_schema={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                },
            )
        ]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return f"{self.greeting}: {tool_args.get('msg', '')}"

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        pass

    async def release_rollout(self, rollout_id: str) -> None:
        pass

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        pass

    async def copy_content_to_workspace(
        self, rollout_id: str, src_content: str | bytes, dst_filename: str
    ) -> None:
        pass

    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        pass

    async def compute_reward(
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        return {"score": 1.0 if completion == ground_truth else 0.0}


class BadInitEnv(MinimalEnv):
    """Env whose __init__ always raises."""

    def __init__(self, **kwargs):
        raise RuntimeError("intentional init failure")


# ---------------------------------------------------------------------------
# Tests: payload.py (BundleMetadata)
# ---------------------------------------------------------------------------


class TestBundleMetadata:
    def test_to_json_bytes_roundtrip(self):
        metadata = BundleMetadata(
            pip_dependencies=["aiohttp>=3.9", "numpy"],
            python_version="3.12",
            benchmax_version="0.1.0",
            constructor_args={"greeting": "hi"},
        )
        data = metadata.to_json_bytes()
        restored = BundleMetadata.from_json_bytes(data)

        assert restored.pip_dependencies == ["aiohttp>=3.9", "numpy"]
        assert restored.python_version == "3.12"
        assert restored.benchmax_version == "0.1.0"
        assert restored.constructor_args == {"greeting": "hi"}
        assert restored.format_version == metadata.format_version


# ---------------------------------------------------------------------------
# Tests: bundler.py (bundle_env + file helpers)
# ---------------------------------------------------------------------------


class TestBundleEnv:
    def test_bundle_minimal_env(self):
        bundle = bundle_env(MinimalEnv, pip_dependencies=["aiohttp"])
        assert isinstance(bundle, BundledEnv)
        assert bundle.metadata.pip_dependencies == ["aiohttp"]
        assert (
            bundle.metadata.python_version
            == f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        assert len(bundle.pickled_class) > 0

    def test_bundle_no_deps(self):
        bundle = bundle_env(MinimalEnv)
        assert bundle.metadata.pip_dependencies == []

    def test_bundle_with_constructor_args(self):
        bundle = bundle_env(
            MinimalEnv,
            constructor_args={"greeting": "yo"},
        )
        assert bundle.metadata.constructor_args == {"greeting": "yo"}

    def test_bundle_not_a_class_raises(self):
        with pytest.raises(ValidationError):
            bundle_env("not a class")  # type: ignore

    def test_bundle_base_env_raises(self):
        with pytest.raises(ValidationError):
            bundle_env(BaseEnv)

    def test_bundle_skip_validation(self):
        bundle = bundle_env(BaseEnv, validate=False)
        assert len(bundle.pickled_class) > 0

    def test_bundle_local_modules(self):
        mod = types.ModuleType("fake_helpers")
        mod.CONSTANT = 42  # type: ignore
        sys.modules["fake_helpers"] = mod

        try:
            bundle = bundle_env(MinimalEnv, local_modules=[mod])
            assert len(bundle.pickled_class) > 0
        finally:
            del sys.modules["fake_helpers"]

    def test_bundle_bad_local_module_raises(self):
        with pytest.raises(
            BundlingError, match="local_modules must contain module objects"
        ):
            bundle_env(MinimalEnv, local_modules=["not_a_module"])  # type: ignore

    def test_write_read_bundle_files(self, tmp_path: Path):
        bundle = bundle_env(MinimalEnv, constructor_args={"greeting": "hi"})
        pickle_path = tmp_path / "env_class.pkl"
        metadata_path = tmp_path / "env_meta.json"

        write_bundle_files(bundle, pickle_path, metadata_path)
        restored = read_bundle_files(pickle_path, metadata_path)

        assert restored.pickled_class == bundle.pickled_class
        assert restored.metadata == bundle.metadata


# ---------------------------------------------------------------------------
# Tests: loader.py (load_env)
# ---------------------------------------------------------------------------


class TestLoadEnv:
    def test_load_from_pickle_and_metadata(self):
        bundle = bundle_env(MinimalEnv)
        env_class = load_env(bundle.pickled_class, bundle.metadata, install_pip_deps=False)
        assert env_class is not None
        assert issubclass(env_class, BaseEnv)
        assert env_class.__name__ == "MinimalEnv"

    def test_load_from_files(self, tmp_path: Path):
        bundle = bundle_env(MinimalEnv)
        pickle_path = tmp_path / "env_class.pkl"
        metadata_path = tmp_path / "env_meta.json"
        write_bundle_files(bundle, pickle_path, metadata_path)

        env_class = load_env_from_files(
            pickle_path, metadata_path, install_pip_deps=False
        )
        assert issubclass(env_class, BaseEnv)

    def test_full_roundtrip_instance(self):
        bundle = bundle_env(MinimalEnv, constructor_args={"greeting": "hi"})
        env = load_env(bundle.pickled_class, bundle.metadata)
        assert isinstance(env, BaseEnv)
        assert env.greeting == "hi"  # type: ignore

    @pytest.mark.asyncio
    async def test_roundtrip_async_methods(self):
        bundle = bundle_env(MinimalEnv)
        env_class = load_env(bundle.pickled_class, bundle.metadata, install_pip_deps=False)
        env = env_class(greeting="test")  # type: ignore

        tools = await env.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

        result = await env.run_tool("r1", "echo", msg="world")
        assert result == "test: world"

        reward = await env.compute_reward("r1", "answer", "answer")
        assert reward == {"score": 1.0}

        await env.shutdown()

    def test_python_version_mismatch_raises(self):
        bundle = bundle_env(MinimalEnv)
        metadata = BundleMetadata(
            pip_dependencies=[],
            python_version="3.11",
            benchmax_version="0.1.0",
        )
        with pytest.raises(IncompatiblePythonError, match="Python 3.11"):
            load_env(bundle.pickled_class, metadata, install_pip_deps=False)

    def test_python_version_mismatch_allowed(self):
        bundle = bundle_env(MinimalEnv)
        metadata = BundleMetadata(
            pip_dependencies=[],
            python_version="3.11",
            benchmax_version="0.1.0",
        )
        env_class = load_env(
            bundle.pickled_class,
            metadata,
            install_pip_deps=False,
            check_python_version=False,
        )
        assert issubclass(env_class, BaseEnv)

    def test_benchmax_version_mismatch_raises(self):
        bundle = bundle_env(MinimalEnv)
        metadata = BundleMetadata(
            pip_dependencies=[],
            python_version="3.12",
            benchmax_version="0.1.0",
        )
        with patch("importlib.metadata.version") as get_version:
            get_version.return_value = "9.9.9"
            with pytest.raises(IncompatibleBenchmaxError):
                load_env(
                    bundle.pickled_class,
                    metadata,
                    install_pip_deps=False,
                    check_benchmax_version=True,
                )

    def test_load_no_install(self):
        bundle = bundle_env(MinimalEnv, pip_dependencies=["some-package"])
        with patch("benchmax.bundle.loader.subprocess") as mock_sub:
            env_class = load_env(
                bundle.pickled_class,
                bundle.metadata,
                install_pip_deps=False,
            )
            mock_sub.run.assert_not_called()
        assert issubclass(env_class, BaseEnv)

    def test_load_with_install(self):
        bundle = bundle_env(MinimalEnv, pip_dependencies=["some-package"])
        with patch("benchmax.bundle.loader.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            load_env(bundle.pickled_class, bundle.metadata, install_pip_deps=True)
            mock_sub.run.assert_called_once()
            cmd = mock_sub.run.call_args[0][0]
            assert "pip" in cmd
            assert "some-package" in cmd

    def test_instantiate_false_returns_class(self):
        bundle = bundle_env(MinimalEnv, constructor_args={"greeting": "hi"})
        env_class = load_env(
            bundle.pickled_class, bundle.metadata, instantiate=False
        )
        assert isinstance(env_class, type)

    def test_explicit_constructor_overrides_metadata(self):
        bundle = bundle_env(MinimalEnv, constructor_args={"greeting": "meta"})
        env = load_env(
            bundle.pickled_class,
            bundle.metadata,
            constructor_args={"greeting": "explicit"},
        )
        assert env.greeting == "explicit"  # type: ignore


# ---------------------------------------------------------------------------
# Tests: validator.py (validate_bundle - isolated validation)
# ---------------------------------------------------------------------------


class TestValidateBundle:
    """Tests for validate_bundle which runs isolated venv validation."""

    def test_validate_bundle_mocked_subprocess(self):
        bundle = bundle_env(MinimalEnv, pip_dependencies=["aiohttp"])

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            mock_sub.run.return_value.stdout = "OK: MinimalEnv loaded"
            mock_sub.run.return_value.stderr = ""

            warnings = validate_bundle(bundle)

            assert mock_sub.run.call_count >= 2
            assert isinstance(warnings, list)

    def test_validate_bundle_venv_creation_failure(self):
        bundle = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 1
            mock_sub.run.return_value.stderr = "venv creation failed"

            warnings = validate_bundle(bundle)

            assert any("Failed to create venv" in w for w in warnings)

    def test_validate_bundle_pip_install_failure(self):
        bundle = bundle_env(MinimalEnv, pip_dependencies=["nonexistent-pkg-xyz"])

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.side_effect = [
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type(
                    "Result",
                    (),
                    {"returncode": 1, "stdout": "", "stderr": "pip install failed"},
                )(),
            ]

            with pytest.raises(ValidationError, match="Failed to install dependencies"):
                validate_bundle(bundle)

    def test_validate_bundle_smoke_test_failure(self):
        bundle = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.side_effect = [
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                type(
                    "Result",
                    (),
                    {
                        "returncode": 1,
                        "stdout": "",
                        "stderr": "ImportError: missing dep",
                    },
                )(),
            ]

            with pytest.raises(ValidationError, match="Isolated smoke test failed"):
                validate_bundle(bundle)

    def test_validate_bundle_with_constructor_args(self):
        bundle = bundle_env(MinimalEnv, constructor_args={"greeting": "test"})

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            mock_sub.run.return_value.stdout = "OK: MinimalEnv loaded, 1 tools"
            mock_sub.run.return_value.stderr = ""

            warnings = validate_bundle(bundle)

            assert isinstance(warnings, list)
            calls = mock_sub.run.call_args_list
            assert len(calls) >= 3

    def test_validate_bundle_timeout_warning(self):
        import subprocess as real_subprocess

        bundle = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.side_effect = real_subprocess.TimeoutExpired("cmd", 120)
            mock_sub.TimeoutExpired = real_subprocess.TimeoutExpired

            warnings = validate_bundle(bundle)

            assert any("timed out" in w for w in warnings)

    def test_validate_bundle_cleanup_on_success(self):
        import tempfile as real_tempfile

        bundle = bundle_env(MinimalEnv)
        created_dirs = []

        original_mkdtemp = real_tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = original_mkdtemp(*args, **kwargs)
            created_dirs.append(d)
            return d

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            mock_sub.run.return_value.stdout = "OK"
            mock_sub.run.return_value.stderr = ""

            with patch("benchmax.bundle.validator.tempfile.mkdtemp", tracking_mkdtemp):
                validate_bundle(bundle)

            assert len(created_dirs) == 1
            assert not Path(created_dirs[0]).exists()
