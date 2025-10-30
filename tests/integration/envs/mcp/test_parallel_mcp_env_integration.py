"""
Fast integration tests using reused local servers.

These tests use session-scoped server fixtures for speed.
Server state may be contaminated between tests, but that's acceptable
for fast iteration during development.

Tests cover:
- Basic rollout lifecycle
- All three reward types (stateless, memory variable, workspace file)
- Concurrent rollouts and isolation
- Workspace file operations
- Error handling
"""

import asyncio
import pytest
from pathlib import Path

from benchmax.envs.mcp.parallel_mcp_env import ParallelMcpEnv
from benchmax.envs.mcp.provisioners import ManualProvisioner


# ===== Basic Rollout Lifecycle Tests =====


class TestBasicRolloutLifecycle:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_basic_rollout_lifecycle(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test complete rollout: init → tool → reward → cleanup."""
        rollout_id = unique_rollout_id

        # Initialize rollout
        await local_env.init_rollout(rollout_id)

        # Run simple tool
        result = await local_env.run_tool(rollout_id, "hello_world", name="TestUser")
        assert "Hello, TestUser!" in result

        # Compute stateless reward
        result = await local_env.compute_reward(
            rollout_id,
            completion="Hello, TestUser!",
            ground_truth={"completion": "Hello, TestUser!"},
        )

        # compute_reward returns dict with results from all reward functions
        # For stateless reward, completion should be 1.0
        assert "completion" in result, f"Expected 'completion' key in result: {result}"
        assert result["completion"] == 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_list_tools(self, local_env: ParallelMcpEnv):
        """Verify all expected tools are available."""
        tools = await local_env.list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            "hello_world",
            "define_variable",
            "get_variable",
            "evaluate",
            "append_log",
            "read_log",
            "allocate_memory",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rollout_already_initialized_error(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test that initializing the same rollout twice raises error."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        with pytest.raises(RuntimeError, match="already initialized"):
            await local_env.init_rollout(rollout_id)

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_tool_on_uninitialized_rollout_error(self, local_env: ParallelMcpEnv):
        """Test that running tool on uninitialized rollout raises error."""
        fake_rollout_id = "nonexistent-rollout"

        with pytest.raises(RuntimeError, match="not initialized"):
            await local_env.run_tool(fake_rollout_id, "hello_world", name="Test")


# ===== Reward Type Tests =====


class TestRewardType:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_stateless_reward_success(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test stateless reward with matching completion."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)
        result = await local_env.run_tool(rollout_id, "hello_world", name="Alice")

        reward = await local_env.compute_reward(
            rollout_id,
            completion="Hello, Alice!",
            ground_truth={"completion": "Hello, Alice!"},
        )
        assert reward["completion"] == 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_stateless_reward_failure(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test stateless reward with mismatched completion."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)
        await local_env.run_tool(rollout_id, "hello_world", name="Bob")

        reward = await local_env.compute_reward(
            rollout_id,
            completion="Wrong response",
            ground_truth={"completion": "Hello, Bob!"},
        )
        assert reward["completion"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_variable_reward_success(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test memory variable reward with correct variable value."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Define variable
        await local_env.run_tool(rollout_id, "define_variable", name="x", value=42.0)

        # Compute reward checking variable
        reward = await local_env.compute_reward(
            rollout_id,
            completion="ignored",
            ground_truth={"variable": {"name": "x", "expected_value": 42.0}},
        )
        assert reward["variable"] == 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_variable_reward_failure(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test memory variable reward with incorrect variable value."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Define variable with wrong value
        await local_env.run_tool(rollout_id, "define_variable", name="y", value=10.0)

        # Compute reward expecting different value
        reward = await local_env.compute_reward(
            rollout_id,
            completion="ignored",
            ground_truth={"variable": {"name": "y", "expected_value": 20.0}},
        )
        assert reward["variable"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_variable_reward_undefined_variable(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test memory variable reward with undefined variable returns 0."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Don't define any variable
        # Compute reward checking non-existent variable
        reward = await local_env.compute_reward(
            rollout_id,
            completion="ignored",
            ground_truth={"variable": {"name": "nonexistent", "expected_value": 100.0}},
        )
        assert reward["variable"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_workspace_file_reward_success(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test workspace file reward with correct file content."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Append to log file
        await local_env.run_tool(
            rollout_id, "append_log", filename="test.log", message="success"
        )

        # Compute reward checking file content
        reward = await local_env.compute_reward(
            rollout_id,
            completion="ignored",
            ground_truth={
                "log": {"filename": "test.log", "expected_content": "success"}
            },
        )
        assert reward["log"] == 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_workspace_file_reward_failure(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test workspace file reward with incorrect file content."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Append different content
        await local_env.run_tool(
            rollout_id, "append_log", filename="test.log", message="wrong"
        )

        # Compute reward expecting different content
        reward = await local_env.compute_reward(
            rollout_id,
            completion="ignored",
            ground_truth={
                "log": {"filename": "test.log", "expected_content": "correct"}
            },
        )
        assert reward["log"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_workspace_file_reward_missing_file(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test workspace file reward with missing file returns 0."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Don't create any file
        # Compute reward checking non-existent file
        reward = await local_env.compute_reward(
            rollout_id,
            completion="ignored",
            ground_truth={
                "log": {"filename": "nonexistent.log", "expected_content": "anything"}
            },
        )
        assert reward["log"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_integration_multiple_tools_and_rewards(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Full integration: use all tools and verify multiple reward signals."""
        rollout_id = unique_rollout_id

        # Initialize environment
        await local_env.init_rollout(rollout_id)

        # Define a variable and verify its effect
        await local_env.run_tool(rollout_id, "define_variable", name="x", value=5.0)

        # Use evaluate to compute something with that variable
        result = await local_env.run_tool(rollout_id, "evaluate", expression="x * 2")
        assert "10" in result or "10.0" in result

        # Append some logs
        await local_env.run_tool(
            rollout_id, "append_log", filename="actions.log", message="computed 10"
        )

        # Read back the log file to verify content consistency
        log_contents = await local_env.run_tool(
            rollout_id, "read_log", filename="actions.log"
        )
        assert "computed 10" in log_contents

        # Run a stateless tool as a separate check
        greeting = await local_env.run_tool(rollout_id, "hello_world", name="Eve")
        assert "Eve" in greeting

        # Compute reward that checks all three reward types simultaneously
        reward = await local_env.compute_reward(
            rollout_id,
            completion=greeting,
            ground_truth={
                "completion": "Hello, Eve!",  # stateless check
                "variable": {"name": "x", "expected_value": 5.0},  # memory var check
                "log": {
                    "filename": "actions.log",
                    "expected_content": "computed 10",
                },  # workspace file check
            },
        )

        # Verify that all reward components exist and have correct scores
        assert reward["completion"] == 1.0
        assert reward["variable"] == 1.0
        assert reward["log"] == 1.0

        # Cleanup rollout
        await local_env.release_rollout(rollout_id)


# ===== Tool Execution Tests =====


class TestToolExecution:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_variable_define_and_get(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test defining and retrieving variables."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Define variable
        result = await local_env.run_tool(
            rollout_id, "define_variable", name="pi", value=3.14
        )
        assert "pi" in result

        # Get variable
        result = await local_env.run_tool(rollout_id, "get_variable", name="pi")
        assert "3.14" in result

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_evaluate_expression(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test evaluating arithmetic expressions with variables."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Define variables
        await local_env.run_tool(rollout_id, "define_variable", name="a", value=10.0)
        await local_env.run_tool(rollout_id, "define_variable", name="b", value=5.0)

        # Evaluate expression
        result = await local_env.run_tool(rollout_id, "evaluate", expression="a + b")
        assert "15" in result or "15.0" in result

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_evaluate_expression_error(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test that invalid expressions are handled gracefully."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Try to evaluate undefined variable
        result = await local_env.run_tool(
            rollout_id, "evaluate", expression="undefined_var + 1"
        )

        assert "error" in result.lower() or "undefined" in result.lower()

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_append_and_read_log(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test appending to and reading workspace files."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Append multiple messages
        await local_env.run_tool(
            rollout_id, "append_log", filename="app.log", message="Line 1"
        )
        await local_env.run_tool(
            rollout_id, "append_log", filename="app.log", message="Line 2"
        )

        # Read log
        result = await local_env.run_tool(rollout_id, "read_log", filename="app.log")
        assert "Line 1" in result
        assert "Line 2" in result

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_read_nonexistent_file_error(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test that reading nonexistent files is handled gracefully."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        result = await local_env.run_tool(
            rollout_id, "read_log", filename="nonexistent.log"
        )
        assert "not found" in result.lower() or "error" in result.lower()

        # Cleanup
        await local_env.release_rollout(rollout_id)


# ===== Workspace File Operations Tests =====


class TestWorkspaceFileOperations:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_upload_download_text_file(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str, tmp_path: Path
    ):
        """Test uploading and downloading text files."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Create local file
        src_file = tmp_path / "test.txt"
        src_file.write_text("Test content", encoding="utf-8")

        # Upload
        await local_env.copy_to_workspace(rollout_id, src_file)

        # Download
        dst_file = tmp_path / "downloaded.txt"
        await local_env.copy_from_workspace(rollout_id, "test.txt", dst_file)

        # Verify
        assert dst_file.read_text(encoding="utf-8") == "Test content"

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_upload_content_various_types(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str, tmp_path: Path
    ):
        """Test uploading various content types (UTF-8, binary, unicode)."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        test_cases = {
            "utf8.txt": ("UTF-8 text content", "utf-8"),
            "unicode.txt": ("你好, мир, hello 🌍", "utf-8"),
            "json.json": ('{"key": "value"}', "utf-8"),
        }

        # Upload all files
        for filename, (content, encoding) in test_cases.items():
            await local_env.copy_content_to_workspace(
                rollout_id, content, filename, encoding=encoding
            )

        # Download and verify
        for filename, (expected_content, encoding) in test_cases.items():
            dst_file = tmp_path / filename
            await local_env.copy_from_workspace(rollout_id, filename, dst_file)
            actual = dst_file.read_text(encoding=encoding)
            assert actual == expected_content, f"Mismatch in {filename}"

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_upload_binary_content(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str, tmp_path: Path
    ):
        """Test uploading and downloading binary content."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Upload binary data
        binary_data = b"\x00\x01\x02\x03\xff\xfe"
        await local_env.copy_content_to_workspace(rollout_id, binary_data, "binary.bin")

        # Download and verify
        dst_file = tmp_path / "binary.bin"
        await local_env.copy_from_workspace(rollout_id, "binary.bin", dst_file)
        assert dst_file.read_bytes() == binary_data

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_upload_with_custom_filename(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str, tmp_path: Path
    ):
        """Test uploading file with custom destination filename."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Create local file
        src_file = tmp_path / "original.txt"
        src_file.write_text("Content", encoding="utf-8")

        # Upload with different name
        await local_env.copy_to_workspace(
            rollout_id, src_file, dst_filename="renamed.txt"
        )

        # Download using new name
        dst_file = tmp_path / "downloaded.txt"
        await local_env.copy_from_workspace(rollout_id, "renamed.txt", dst_file)
        assert dst_file.read_text(encoding="utf-8") == "Content"

        # Cleanup
        await local_env.release_rollout(rollout_id)


# ===== Concurrency Tests =====


class TestConcurrency:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_rollouts(self, local_env: ParallelMcpEnv):
        """Test multiple concurrent rollouts with different operations."""
        num_rollouts = local_env.num_servers * 2
        rollout_ids = [
            f"concurrent-{i}-{asyncio.get_event_loop().time()}"
            for i in range(num_rollouts)
        ]

        async def run_rollout(rollout_id: str, index: int):
            await local_env.init_rollout(rollout_id)

            # Each rollout does different operations
            if index % 3 == 0:
                # Stateless operation
                result = await local_env.run_tool(
                    rollout_id, "hello_world", name=f"User{index}"
                )
                reward = await local_env.compute_reward(
                    rollout_id,
                    completion=result,
                    ground_truth={"completion": f"Hello, User{index}!"},
                )
                expected_key = "completion"
            elif index % 3 == 1:
                # Memory variable operation
                await local_env.run_tool(
                    rollout_id,
                    "define_variable",
                    name=f"var{index}",
                    value=float(index),
                )
                reward = await local_env.compute_reward(
                    rollout_id,
                    completion="ignored",
                    ground_truth={
                        "variable": {
                            "name": f"var{index}",
                            "expected_value": float(index),
                        }
                    },
                )
                expected_key = "variable"
            else:
                # Workspace file operation
                await local_env.run_tool(
                    rollout_id,
                    "append_log",
                    filename=f"log{index}.txt",
                    message=f"msg{index}",
                )
                reward = await local_env.compute_reward(
                    rollout_id,
                    completion="ignored",
                    ground_truth={
                        "log": {
                            "filename": f"log{index}.txt",
                            "expected_content": f"msg{index}",
                        }
                    },
                )
                expected_key = "log"

            return rollout_id, reward, expected_key

        # Run all rollouts concurrently
        results = await asyncio.gather(
            *[run_rollout(rid, i) for i, rid in enumerate(rollout_ids)]
        )

        # Verify all succeeded
        assert len(results) == num_rollouts
        for rollout_id, reward, expected_key in results:
            assert (
                reward[expected_key] == 1.0
            ), f"Rollout {rollout_id} failed with reward {reward}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rollout_isolation(self, local_env: ParallelMcpEnv):
        """Test that rollouts don't interfere with each other's state."""
        rollout_ids = [
            f"isolated-{i}-{asyncio.get_event_loop().time()}" for i in range(local_env.num_servers)
        ]

        async def setup_rollout(rollout_id: str, var_value: float):
            await local_env.init_rollout(rollout_id)
            await local_env.run_tool(
                rollout_id, "define_variable", name="shared_var", value=var_value
            )

        # Initialize all rollouts with different values
        await asyncio.gather(
            *[setup_rollout(rid, float(i * 10)) for i, rid in enumerate(rollout_ids)]
        )

        # Verify each rollout has its own value
        async def verify_rollout(rollout_id: str, expected_value: float):
            reward = await local_env.compute_reward(
                rollout_id,
                completion="ignored",
                ground_truth={
                    "variable": {
                        "name": "shared_var",
                        "expected_value": expected_value,
                    }
                },
            )
            return reward["variable"]

        results = await asyncio.gather(
            *[verify_rollout(rid, float(i * 10)) for i, rid in enumerate(rollout_ids)]
        )

        # All should succeed with their own isolated values
        assert all(r == 1.0 for r in results), "Rollout isolation violated"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sequential_rollouts_cleanup(self, local_env: ParallelMcpEnv):
        """Test that sequential rollouts properly cleanup state."""
        base_rollout = f"sequential-{asyncio.get_event_loop().time()}"

        for i in range(3):
            rollout_id = f"{base_rollout}-{i}"
            await local_env.init_rollout(rollout_id)

            # Define variable
            await local_env.run_tool(
                rollout_id, "define_variable", name="counter", value=float(i)
            )

            # Verify value
            reward = await local_env.compute_reward(
                rollout_id,
                completion="ignored",
                ground_truth={
                    "variable": {"name": "counter", "expected_value": float(i)}
                },
            )
            assert reward["variable"] == 1.0

            # Rollout is automatically released after compute_reward

        # All rollouts completed successfully without state leakage

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_workspace_operations(
        self, local_env: ParallelMcpEnv, tmp_path: Path
    ):
        """Test concurrent file upload/download operations."""
        num_rollouts = int(local_env.num_servers * 1.5)
        rollout_ids = [
            f"workspace-{i}-{asyncio.get_event_loop().time()}"
            for i in range(num_rollouts)
        ]

        async def run_file_operations(rollout_id: str, index: int):
            await local_env.init_rollout(rollout_id)

            # Upload unique content
            content = f"Content for rollout {index}"
            await local_env.copy_content_to_workspace(
                rollout_id, content, f"file{index}.txt"
            )

            # Download and verify
            dst_file = tmp_path / f"download_{index}.txt"
            await local_env.copy_from_workspace(
                rollout_id, f"file{index}.txt", dst_file
            )
            downloaded = dst_file.read_text(encoding="utf-8")

            # Cleanup
            await local_env.release_rollout(rollout_id)

            return downloaded == content

        results = await asyncio.gather(
            *[run_file_operations(rid, i) for i, rid in enumerate(rollout_ids)]
        )

        assert all(results), "Some file operations failed"


# ===== Error Handling Tests =====


class TestErrorHandling:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_tool_error_handling(
        self, local_env: ParallelMcpEnv, unique_rollout_id: str
    ):
        """Test that tool errors are handled gracefully without crashing."""
        rollout_id = unique_rollout_id

        await local_env.init_rollout(rollout_id)

        # Try to get undefined variable (should return error string, not crash)
        result = await local_env.run_tool(rollout_id, "get_variable", name="undefined")
        assert isinstance(result, str)
        assert "error" in result.lower() or "not defined" in result.lower()

        # Rollout should still be usable
        result = await local_env.run_tool(rollout_id, "hello_world", name="Test")
        assert "Hello, Test!" in result

        # Cleanup
        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_compute_reward_on_uninitialized_rollout(
        self, local_env: ParallelMcpEnv
    ):
        """Test that computing reward on uninitialized rollout raises error."""
        fake_rollout_id = "nonexistent-reward-test"

        with pytest.raises(RuntimeError, match="not initialized"):
            await local_env.compute_reward(
                fake_rollout_id, completion="test", ground_truth={"completion": "test"}
            )


# ===== API validation Tests =====


class TestApiSecretValidation:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_init_rollout_with_wrong_secret(
        self, local_servers_with_secret, tmp_path
    ):
        """Test that init_rollout succeeds with correct secret and fails with wrong secret."""
        addresses, correct_secret = local_servers_with_secret

        # --- Correct secret env ---
        correct_env = ParallelMcpEnv(
            workdir_path=tmp_path / "correct_env",
            provisioner=ManualProvisioner(addresses),
            api_secret=correct_secret,
            provision_at_init=True,
        )

        rollout_id = "correct-secret-rollout"
        await asyncio.wait_for(correct_env.init_rollout(rollout_id), timeout=5)

        tools = await correct_env.list_tools()
        assert len(tools) > 0

        await correct_env.shutdown()

        # --- Wrong secret env ---
        wrong_env = ParallelMcpEnv(
            workdir_path=tmp_path / "wrong_env",
            provisioner=ManualProvisioner(addresses),
            api_secret="wrongsecret",
            provision_at_init=True,
        )

        # init_rollout should fail
        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await asyncio.wait_for(wrong_env.init_rollout("wrong-secret-rollout"), timeout=5)

        await wrong_env.shutdown()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_file_transfer_with_tampered_secret(self, local_env: ParallelMcpEnv, tmp_path: Path):
        """Test copying files into and from workspace fails after tampering with API secret."""
        rollout_id = "file-transfer-rollout"
        await asyncio.wait_for(local_env.init_rollout(rollout_id), timeout=5)

        # Prepare a local file
        local_file = tmp_path / "hello.txt"
        local_file.write_text("hello world")

        # Copy into workspace with correct secret
        await local_env.copy_to_workspace(rollout_id, local_file, "remote_hello.txt")

        # Copy back from workspace
        downloaded_file = tmp_path / "downloaded.txt"
        await local_env.copy_from_workspace(rollout_id, "remote_hello.txt", downloaded_file)
        assert downloaded_file.read_text() == "hello world"

        # Tamper _api_secret
        local_env._api_secret = "wrongsecret"

        # Copy operations should fail
        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await local_env.copy_to_workspace(rollout_id, local_file, "fail.txt")

        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await local_env.copy_from_workspace(rollout_id, "remote_hello.txt", tmp_path / "fail.txt")

        await local_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_compute_reward_with_tampered_secret(self, local_env: ParallelMcpEnv):
        """Test that compute_reward fails after tampering with API secret."""
        rollout_id = "reward-rollout"
        await asyncio.wait_for(local_env.init_rollout(rollout_id), timeout=5)

        # Run a simple tool to generate some reward
        await local_env.run_tool(rollout_id, "hello_world", name="Tester")
        rewards = await local_env.compute_reward(
            rollout_id,
            completion="Hello, TestUser!",
            ground_truth={"completion": "Hello, TestUser!"},
        )
        assert "completion" in rewards

        # Tamper _api_secret
        local_env._api_secret = "wrongsecret"

        # Authenticated operation should now fail
        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await local_env.compute_reward(
                rollout_id,
                completion="Hello, TestUser!",
                ground_truth={"completion": "Hello, TestUser!"},
            )

        await local_env.release_rollout(rollout_id)
