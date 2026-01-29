import struct
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition
from benchmax.bundle.bundler import (
    EnvPayload,
    bundle_env,
)
from benchmax.bundle.validator import validate_structure, validate_payload
from benchmax.bundle.errors import (
    IncompatiblePythonError,
    BundlingError,
    ValidationError,
)
from benchmax.bundle.payload import FORMAT_VERSION, MAGIC
from benchmax.bundle.loader import load_env


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
# Tests: payload.py
# ---------------------------------------------------------------------------


class TestEnvPayload:
    def test_to_bytes_and_from_bytes_roundtrip(self):
        payload = EnvPayload(
            pickled_class=b"fake-pickle-data",
            pip_dependencies=["aiohttp>=3.9", "numpy"],
            python_version="3.12",
            benchmax_version="0.1.0",
            extra_metadata={"custom": "value"},
        )
        data = payload.to_bytes()
        restored = EnvPayload.from_bytes(data)

        assert restored.pickled_class == b"fake-pickle-data"
        assert restored.pip_dependencies == ["aiohttp>=3.9", "numpy"]
        assert restored.python_version == "3.12"
        assert restored.benchmax_version == "0.1.0"
        assert restored.extra_metadata == {"custom": "value"}

    def test_magic_bytes(self):
        payload = EnvPayload(
            pickled_class=b"data",
            pip_dependencies=[],
            python_version="3.12",
            benchmax_version="0.1.0",
        )
        data = payload.to_bytes()
        assert data[:4] == MAGIC

    def test_format_version(self):
        payload = EnvPayload(
            pickled_class=b"data",
            pip_dependencies=[],
            python_version="3.12",
            benchmax_version="0.1.0",
        )
        data = payload.to_bytes()
        version = struct.unpack("!H", data[4:6])[0]
        assert version == FORMAT_VERSION

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="bad magic bytes"):
            EnvPayload.from_bytes(b"BADXsomething_else_here")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            EnvPayload.from_bytes(b"BM")

    def test_unsupported_version_raises(self):
        data = MAGIC + struct.pack("!H", 999) + struct.pack("!I", 0)
        with pytest.raises(ValueError, match="Unsupported payload format version"):
            EnvPayload.from_bytes(data)

    def test_truncated_metadata_raises(self):
        # Claim metadata is 1000 bytes but only provide 5
        data = (
            MAGIC
            + struct.pack("!H", FORMAT_VERSION)
            + struct.pack("!I", 1000)
            + b"short"
        )
        with pytest.raises(ValueError, match="truncated metadata"):
            EnvPayload.from_bytes(data)

    def test_empty_extra_metadata_default(self):
        payload = EnvPayload(
            pickled_class=b"data",
            pip_dependencies=[],
            python_version="3.12",
            benchmax_version="0.1.0",
        )
        data = payload.to_bytes()
        restored = EnvPayload.from_bytes(data)
        assert restored.extra_metadata == {}


# ---------------------------------------------------------------------------
# Tests: bundler.py (bundle_env)
# ---------------------------------------------------------------------------


class TestBundleEnv:
    def test_bundle_minimal_env(self):
        payload = bundle_env(MinimalEnv, pip_dependencies=["aiohttp"])
        assert isinstance(payload, EnvPayload)
        assert payload.pip_dependencies == ["aiohttp"]
        assert (
            payload.python_version
            == f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        assert len(payload.pickled_class) > 0

    def test_bundle_no_deps(self):
        payload = bundle_env(MinimalEnv)
        assert payload.pip_dependencies == []

    def test_bundle_with_extra_metadata(self):
        payload = bundle_env(
            MinimalEnv,
            extra_metadata={"experiment": "test-run-1"},
        )
        assert payload.extra_metadata == {"experiment": "test-run-1"}

    def test_bundle_not_a_class_raises(self):
        with pytest.raises(ValidationError):
            bundle_env("not a class")  # type: ignore

    def test_bundle_base_env_raises(self):
        with pytest.raises(ValidationError):
            bundle_env(BaseEnv)

    def test_bundle_skip_validation(self):
        # Even BaseEnv should serialize if we skip validation
        # (it will fail to instantiate on remote, but that's the user's problem)
        payload = bundle_env(BaseEnv, validate=False)
        assert len(payload.pickled_class) > 0

    def test_bundle_local_modules(self):
        # Create a fake module
        mod = types.ModuleType("fake_helpers")
        mod.CONSTANT = 42  # type: ignore
        sys.modules["fake_helpers"] = mod

        try:
            payload = bundle_env(MinimalEnv, local_modules=[mod])
            assert len(payload.pickled_class) > 0
        finally:
            del sys.modules["fake_helpers"]

    def test_bundle_bad_local_module_raises(self):
        with pytest.raises(
            BundlingError, match="local_modules must contain module objects"
        ):
            bundle_env(MinimalEnv, local_modules=["not_a_module"])  # type: ignore


# ---------------------------------------------------------------------------
# Tests: loader.py (load_env)
# ---------------------------------------------------------------------------


class TestLoadEnv:
    def test_load_from_payload(self):
        payload = bundle_env(MinimalEnv)
        env_class = load_env(payload, install_deps=False)
        assert env_class is not None
        assert issubclass(env_class, BaseEnv)
        assert env_class.__name__ == "MinimalEnv"

    def test_load_from_bytes(self):
        payload = bundle_env(MinimalEnv)
        data = payload.to_bytes()
        env_class = load_env(data, install_deps=False)
        assert issubclass(env_class, BaseEnv)

    def test_full_roundtrip(self):
        """Bundle → to_bytes → from_bytes → load → instantiate → use."""
        payload = bundle_env(MinimalEnv, pip_dependencies=["aiohttp"])
        data = payload.to_bytes()
        env_class = load_env(data, install_deps=False)

        env = env_class(greeting="hi")  # type: ignore
        assert env.greeting == "hi"  # type: ignore

    @pytest.mark.asyncio
    async def test_roundtrip_async_methods(self):
        payload = bundle_env(MinimalEnv)
        env_class = load_env(payload, install_deps=False)
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
        payload = EnvPayload(
            pickled_class=b"data",
            pip_dependencies=[],
            python_version="3.11",  # wrong version
            benchmax_version="0.1.0",
        )
        with pytest.raises(IncompatiblePythonError, match="Python 3.11"):
            load_env(payload, install_deps=False)

    def test_python_version_mismatch_allowed(self):
        """allow_python_mismatch=True skips the version check."""
        payload = bundle_env(MinimalEnv)
        # Manually override version to simulate mismatch
        mismatched = EnvPayload(
            pickled_class=payload.pickled_class,
            pip_dependencies=[],
            python_version="3.11",
            benchmax_version="0.1.0",
        )
        # Should not raise — the pickle is valid for current Python
        env_class = load_env(mismatched, install_deps=False, allow_python_mismatch=True)
        assert issubclass(env_class, BaseEnv)

    def test_load_no_install(self):
        """install_deps=False should not call subprocess."""
        payload = bundle_env(MinimalEnv, pip_dependencies=["some-package"])
        with patch("benchmax.bundle.loader.subprocess") as mock_sub:
            env_class = load_env(payload, install_deps=False)
            mock_sub.run.assert_not_called()
        assert issubclass(env_class, BaseEnv)

    def test_load_with_install(self):
        """install_deps=True should call pip install."""
        payload = bundle_env(MinimalEnv, pip_dependencies=["some-package"])
        with patch("benchmax.bundle.loader.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            load_env(payload, install_deps=True)
            mock_sub.run.assert_called_once()
            cmd = mock_sub.run.call_args[0][0]
            assert "pip" in cmd
            assert "some-package" in cmd


# ---------------------------------------------------------------------------
# Tests: validator.py (_validate_structure - used by bundle_env)
# ---------------------------------------------------------------------------


class TestValidateStructure:
    """Tests for _validate_structure which runs structural checks before bundling."""

    def test_validate_minimal_env(self):
        warnings = validate_structure(MinimalEnv, [])
        # Should pass with no fatal errors
        assert isinstance(warnings, list)

    def test_validate_not_subclass(self):
        with pytest.raises(ValidationError, match="not a subclass"):
            validate_structure(str, [])  # type: ignore

    def test_validate_base_env_directly(self):
        with pytest.raises(ValidationError, match="Cannot bundle BaseEnv directly"):
            validate_structure(BaseEnv, [])

    def test_validate_incomplete_subclass(self):
        class IncompleteEnv(BaseEnv):
            pass  # missing all abstract methods

        with pytest.raises(ValidationError, match="unimplemented abstract methods"):
            validate_structure(IncompleteEnv, [])

    def test_validate_bad_dependency_string(self):
        with pytest.raises(ValidationError, match="Invalid pip dependency"):
            validate_structure(MinimalEnv, [""])

    def test_validate_stdlib_warning(self):
        warnings = validate_structure(MinimalEnv, ["os"])
        assert any("stdlib" in w for w in warnings)

    def test_validate_non_importable_dep_warning(self):
        warnings = validate_structure(MinimalEnv, ["totally-fake-package-xyz123"])
        assert any("not importable" in w for w in warnings)

    def test_validate_obscure_but_real_package_warning(self):
        """Test that a real but obscure package (not installed) triggers 'not importable' warning."""
        # 'bidict' is a real PyPI package (bidirectional dict) but unlikely to be installed
        warnings = validate_structure(MinimalEnv, ["bidict"])
        # Should warn that it's not currently importable (since it's not installed)
        assert any("not importable" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tests: validator.py (validate_payload - isolated validation)
# ---------------------------------------------------------------------------


class TestValidatePayload:
    """Tests for validate_payload which runs isolated venv validation."""

    def test_validate_payload_mocked_subprocess(self):
        """Test that validate_payload calls subprocess to create venv and run tests."""
        payload = bundle_env(MinimalEnv, pip_dependencies=["aiohttp"])

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            # Mock successful venv creation
            mock_sub.run.return_value.returncode = 0
            mock_sub.run.return_value.stdout = "OK: MinimalEnv loaded"
            mock_sub.run.return_value.stderr = ""

            warnings = validate_payload(payload)

            # Should have called subprocess at least twice (venv create + pip install + smoke test)
            assert mock_sub.run.call_count >= 2
            assert isinstance(warnings, list)

    def test_validate_payload_venv_creation_failure(self):
        """Test handling of venv creation failure."""
        payload = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            # Mock venv creation failure
            mock_sub.run.return_value.returncode = 1
            mock_sub.run.return_value.stderr = "venv creation failed"

            warnings = validate_payload(payload)

            # Should return a warning about venv creation failure
            assert any("Failed to create venv" in w for w in warnings)

    def test_validate_payload_pip_install_failure(self):
        """Test handling of pip install failure."""
        payload = bundle_env(MinimalEnv, pip_dependencies=["nonexistent-pkg-xyz"])

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            # Calls: venv creation, get-pip download, pip bootstrap, pip install deps (fails)
            mock_sub.run.side_effect = [
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # venv
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # get-pip download
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # pip bootstrap
                type(
                    "Result",
                    (),
                    {"returncode": 1, "stdout": "", "stderr": "pip install failed"},
                )(),  # deps
            ]

            with pytest.raises(ValidationError, match="Failed to install dependencies"):
                validate_payload(payload)

    def test_validate_payload_smoke_test_failure(self):
        """Test handling of smoke test failure in isolated env."""
        payload = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            # Calls: venv creation, get-pip download, pip bootstrap, pip install deps, smoke test (fails)
            mock_sub.run.side_effect = [
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # venv
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # get-pip download
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # pip bootstrap
                type(
                    "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
                )(),  # deps
                type(
                    "Result",
                    (),
                    {
                        "returncode": 1,
                        "stdout": "",
                        "stderr": "ImportError: missing dep",
                    },
                )(),  # smoke
            ]

            with pytest.raises(ValidationError, match="Isolated smoke test failed"):
                validate_payload(payload)

    def test_validate_payload_with_constructor_args(self):
        """Test that constructor_args are passed to the smoke test."""
        payload = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            mock_sub.run.return_value.stdout = "OK: MinimalEnv loaded, 1 tools"
            mock_sub.run.return_value.stderr = ""

            warnings = validate_payload(payload, constructor_args={"greeting": "test"})

            assert isinstance(warnings, list)
            # Verify the script was written with constructor_args
            calls = mock_sub.run.call_args_list
            # The last call should be running the smoke test script
            assert len(calls) >= 3

    def test_validate_payload_timeout_warning(self):
        """Test that timeout during validation returns a warning."""
        import subprocess as real_subprocess

        payload = bundle_env(MinimalEnv)

        with patch("benchmax.bundle.validator.subprocess") as mock_sub:
            # Simulate timeout on one of the calls
            mock_sub.run.side_effect = real_subprocess.TimeoutExpired("cmd", 120)
            mock_sub.TimeoutExpired = real_subprocess.TimeoutExpired

            warnings = validate_payload(payload)

            assert any("timed out" in w for w in warnings)

    def test_validate_payload_cleanup_on_success(self):
        """Test that temp directory is cleaned up after successful validation."""
        import tempfile as real_tempfile

        payload = bundle_env(MinimalEnv)
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
                validate_payload(payload)

            # Verify temp dir was created and then cleaned up
            assert len(created_dirs) == 1
            assert not Path(created_dirs[0]).exists()
