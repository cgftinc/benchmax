"""
End-to-end integration tests with fresh provisioners.

These tests provision dedicated servers for complete isolation and
test the full workflow from provisioning to teardown.

Tests are marked as 'slow' and 'remote' (for Skypilot tests).
Use timeouts to prevent test hangs.
"""

import asyncio
import pytest
import random
from pathlib import Path
from typing import Dict

from benchmax.envs.mcp import ParallelMcpEnv


# ===== Helper Functions =====

async def execute_stress_rollout(
    env: ParallelMcpEnv,
    index: int,
    tmp_path: Path
) -> tuple[str, Dict[str, float], bool]:
    """
    Execute a single stress test rollout with random operations.
    
    Returns:
        (rollout_id, reward, success)
    """
    rollout_id = f"stress-{index}-{asyncio.get_event_loop().time()}"
    
    try:
        await env.init_rollout(rollout_id)
        
        # Random number of tool calls (1-5)
        num_calls = random.randint(1, 5)
        for _ in range(num_calls):
            tool_choice = random.choice(["hello_world", "define_variable", "evaluate", "append_log"])
            
            if tool_choice == "hello_world":
                await env.run_tool(rollout_id, "hello_world", name=f"User{random.randint(1, 100)}")
            elif tool_choice == "define_variable":
                var_name = random.choice(["x", "y", "z", "a", "b"])
                await env.run_tool(rollout_id, "define_variable", name=var_name, value=float(random.randint(1, 100)))
            elif tool_choice == "evaluate":
                # May fail if variables not defined, that's ok
                try:
                    await env.run_tool(rollout_id, "evaluate", expression="x + y")
                except:
                    pass
            elif tool_choice == "append_log":
                await env.run_tool(
                    rollout_id,
                    "append_log",
                    filename=f"log{random.randint(1, 3)}.txt",
                    message=f"Message {random.randint(1, 100)}"
                )
        
        # Random file transfers (both upload and download)
        if random.random() < 0.5:
            # Test text or binary upload
            filename = f"random_{index}.txt" if random.random() < 0.5 else f"random_{index}.bin"
            
            if filename.endswith('.txt'):
                # Text content
                content = f"Random text content {random.randint(1, 1000)}"
                await env.copy_content_to_workspace(rollout_id, content, filename)
            else:
                # Binary content
                content = bytes([random.randint(0, 255) for _ in range(20)])
                await env.copy_content_to_workspace(rollout_id, content, filename)
            
            # Also test download
            if random.random() < 0.5:
                try:
                    dst_file = tmp_path / f"stress_download_{index}_{filename}"
                    await env.copy_from_workspace(rollout_id, filename, dst_file)
                    # Verify file was downloaded
                    assert dst_file.exists()
                except:
                    # File might not exist, that's ok
                    pass
        
        # Now set up a known final state for reward computation
        reward_type = random.choice(["completion", "variable", "log"])
        
        if reward_type == "completion":
            result = await env.run_tool(rollout_id, "hello_world", name="FinalUser")
            reward = await env.compute_reward(
                rollout_id,
                completion=result,
                ground_truth={"completion": "Hello, FinalUser!"}
            )
            expected_key = "completion"
        elif reward_type == "variable":
            await env.run_tool(rollout_id, "define_variable", name="final_var", value=42.0)
            reward = await env.compute_reward(
                rollout_id,
                completion="ignored",
                ground_truth={
                    "variable": {
                        "name": "final_var",
                        "expected_value": 42.0
                    }
                }
            )
            expected_key = "variable"
        else:  # log
            await env.run_tool(
                rollout_id,
                "append_log",
                filename="final.log",
                message="Final message"
            )
            reward = await env.compute_reward(
                rollout_id,
                completion="ignored",
                ground_truth={
                    "log": {
                        "filename": "final.log",
                        "expected_content": "Final message"
                    }
                }
            )
            expected_key = "log"
        
        # Verify reward
        success = reward.get(expected_key) == 1.0
        return rollout_id, reward, success
        
    except Exception as e:
        return rollout_id, {}, False


async def run_comprehensive_workflow(
    env: ParallelMcpEnv,
    tmp_path: Path,
    prefix: str = ""
) -> None:
    """
    Shared comprehensive workflow that tests all features.
    
    Args:
        env: The environment to test
        tmp_path: Temporary directory for file operations
        prefix: Prefix for rollout_id and filenames (e.g., "sky-" for cloud)
    """
    # Verify tools are available
    tools = await env.list_tools()
    assert len(tools) > 0, "No tools available"
    tool_names = [t.name for t in tools]
    assert "hello_world" in tool_names
    assert "define_variable" in tool_names
    assert "evaluate" in tool_names
    assert "append_log" in tool_names
    
    rollout_id = f"{prefix}comprehensive-{asyncio.get_event_loop().time()}"
    await env.init_rollout(rollout_id)
    
    # Test 1: hello_world tool
    user_name = f"{prefix}User" if prefix else "Alice"
    result = await env.run_tool(rollout_id, "hello_world", name=user_name)
    assert f"Hello, {user_name}!" in result
    
    # Test 2: define_variable tool
    await env.run_tool(rollout_id, "define_variable", name="x", value=10.0)
    await env.run_tool(rollout_id, "define_variable", name="y", value=20.0)
    await env.run_tool(rollout_id, "define_variable", name="z", value=5.0)
    
    # Test 3: evaluate tool
    result = await env.run_tool(rollout_id, "evaluate", expression="x + y")
    assert "30" in result or "30.0" in result
    
    result = await env.run_tool(rollout_id, "evaluate", expression="x * z")
    assert "50" in result or "50.0" in result
    
    # Test 4: append_log tool
    log_filename = f"{prefix}test.log"
    await env.run_tool(rollout_id, "append_log", filename=log_filename, message="Line 1")
    await env.run_tool(rollout_id, "append_log", filename=log_filename, message="Line 2")
    await env.run_tool(rollout_id, "append_log", filename=log_filename, message="Line 3")
    
    # Test 5: read_log tool (if available)
    if "read_log" in tool_names:
        content = await env.run_tool(rollout_id, "read_log", filename=log_filename)
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content
    
    # Test 6: File operations - copy_content_to_workspace (text)
    text_content = f"{prefix}This is test content for workspace"
    await env.copy_content_to_workspace(rollout_id, text_content, f"{prefix}upload_text.txt")
    
    # Test 7: File operations - copy_content_to_workspace (binary)
    binary_content = b"\x00\x01\x02\xFF\xFE"
    await env.copy_content_to_workspace(rollout_id, binary_content, f"{prefix}upload_binary.bin")
    
    # Test 8: File operations - copy_to_workspace
    src_file = tmp_path / f"{prefix}source.txt"
    src_file.write_text(f"{prefix}Source file content", encoding="utf-8")
    await env.copy_to_workspace(rollout_id, src_file)
    
    # Test 9: File operations - copy_from_workspace
    dst_file = tmp_path / f"{prefix}downloaded.log"
    await env.copy_from_workspace(rollout_id, log_filename, dst_file)
    downloaded_content = dst_file.read_text(encoding="utf-8")
    assert "Line 1" in downloaded_content
    assert "Line 2" in downloaded_content
    assert "Line 3" in downloaded_content
    
    # Test 10: Compute reward
    reward_user = f"{prefix}RewardTest"
    hello_result = await env.run_tool(rollout_id, "hello_world", name=reward_user)
    reward = await env.compute_reward(
        rollout_id,
        completion=hello_result,
        ground_truth={
            "completion": f"Hello, {reward_user}!",
            "variable": {
                "name": "x",
                "expected_value": 10.0
            },
            "log": {
                "filename": log_filename,
                "expected_content": "Line 1\nLine 2\nLine 3"
            }
        }
    )
    assert reward["completion"] == 1.0, f"Completion reward failed: {reward}"
    assert reward["variable"] == 1.0, f"Variable reward failed: {reward}"
    assert reward["log"] == 1.0, f"Log reward failed: {reward}"
    
    print(f"✓ Stage 1: Comprehensive workflow completed successfully ({prefix or 'local'})")


async def run_stress_test(
    env: ParallelMcpEnv,
    tmp_path: Path,
    prefix: str = "",
    min_success_rate: float = 1.0
) -> None:
    """
    Shared stress test with concurrent rollouts.
    
    Args:
        env: The environment to test
        tmp_path: Temporary directory for file operations
        prefix: Prefix for logging (e.g., "Skypilot")
        min_success_rate: Minimum acceptable success rate (0.0 to 1.0)
    """
    num_stress_rollouts = env.num_servers * 20
    print(f"Starting {prefix} stress test with {num_stress_rollouts} concurrent rollouts...")
    
    # Execute all stress rollouts concurrently
    results = await asyncio.gather(
        *[execute_stress_rollout(env, i, tmp_path) for i in range(num_stress_rollouts)]
    )
    
    # Verify results
    assert len(results) == num_stress_rollouts
    successes = sum(1 for _, _, success in results if success)
    success_rate = successes / num_stress_rollouts
    
    print(f"✓ Stage 2: Stress test completed with {success_rate:.1%} success rate ({successes}/{num_stress_rollouts})")
    
    assert success_rate >= min_success_rate, \
        f"{prefix} stress test success rate too low: {success_rate:.1%} < {min_success_rate:.1%}"


async def run_oom_rollout(
    env: ParallelMcpEnv,
    index: int,
    max_oom_calls: int
) -> tuple[int, bool]:
    """
    Execute a single OOM rollout that progressively allocates memory until failure.
    
    Returns:
        (index, oom_occurred)
    """
    rollout_id = f"oom-rollout-{index}"
    
    try:
        await env.init_rollout(rollout_id)
        
        # Verify server is working
        result = await env.run_tool(rollout_id, "hello_world", name=f"OOM{index}")
        assert f"Hello, OOM{index}!" in result
        
        # Progressively allocate memory
        oom_occurred = False
        for i in range(max_oom_calls):
            try:
                result = await env.run_tool(
                    rollout_id,
                    "allocate_memory",
                    megabytes=250
                )
                
                if "error" in result.lower() or "failed" in result.lower():
                    oom_occurred = True
                    break
                
                # Longer delay between allocations to stress the system
                await asyncio.sleep(1.0)
                
            except Exception as e:
                # Server crashed or became unresponsive
                oom_occurred = True
                break
        
        # Try to cleanup (may fail if server is dead)
        try:
            await env.release_rollout(rollout_id)
        except Exception:
            pass
        
        return index, oom_occurred
        
    except Exception as e:
        print(f"Rollout {index} failed with exception: {type(e).__name__}")
        return index, False


# ===== Test 1: Local E2E Comprehensive + Stress =====

@pytest.mark.asyncio
@pytest.mark.slow
async def test_e2e_local_comprehensive_with_stress(fresh_local_env: ParallelMcpEnv, tmp_path: Path) -> None:
    """
    Complete E2E test for local environment with two stages:
    
    Stage 1: Comprehensive workflow testing all features
    Stage 2: Concurrency stress test with random operations
    """
    async def run_test():
        # Stage 1: Comprehensive Workflow
        await run_comprehensive_workflow(fresh_local_env, tmp_path, prefix="")
        
        # Stage 2: Concurrency Stress Test
        await run_stress_test(fresh_local_env, tmp_path, prefix="Local", min_success_rate=1.0)
    
    # Run with timeout
    await asyncio.wait_for(run_test(), timeout=300.0)


# ===== Test 2: Skypilot E2E Comprehensive + Stress =====

@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.remote
async def test_e2e_skypilot_comprehensive_with_stress(fresh_skypilot_env: ParallelMcpEnv, tmp_path: Path) -> None:
    """
    Complete E2E test for Skypilot environment with two stages:
    
    Stage 1: Comprehensive workflow testing all features
    Stage 2: Concurrency stress test with random operations
    """
    async def run_test():
        # Stage 1: Comprehensive Workflow
        await run_comprehensive_workflow(fresh_skypilot_env, tmp_path, prefix="sky-")
        
        # Stage 2: Concurrency Stress Test (allow 95% success rate for cloud)
        await run_stress_test(fresh_skypilot_env, tmp_path, prefix="Skypilot", min_success_rate=0.95)
    
    # Run with extended timeout for cloud operations
    await asyncio.wait_for(run_test(), timeout=600.0)


# ===== Test 3: Skypilot OOM Handling =====

@pytest.mark.asyncio
@pytest.mark.slow  
@pytest.mark.remote
async def test_skypilot_oom_handling(fresh_skypilot_env: ParallelMcpEnv) -> None:
    """
    Test OOM (Out of Memory) handling on Skypilot servers.
    
    Triggers multiple concurrent OOM scenarios across all servers,
    then verifies the server pool can recover and provision new servers.
    """
    async def run_test():
        env = fresh_skypilot_env
        
        # Configuration
        num_rollouts = env.num_servers * 2  # 2 rollouts per server
        max_oom_calls = 20  # Max allocations per rollout

        # Run OOM rollouts
        rollout_tasks = [
            run_oom_rollout(env, num_rollouts + i + 1, max_oom_calls)
            for i in range(num_rollouts)
        ]

        results = await asyncio.gather(*rollout_tasks, return_exceptions=True)
        
        # Count OOMs in this round
        total_ooms = sum(1 for r in results if isinstance(r, tuple) and r[1])
    
        assert total_ooms > 0, "Expected at least one OOM to occur"
        
        # Wait for all recovery tasks to complete
        assert env._server_pool
        await asyncio.gather(
            *list(env._server_pool._recovery_tasks),
            return_exceptions=True
        )

        # Verify all servers have recovered
        available_servers = len(env._server_pool._unassigned_servers)
        assert available_servers == env.num_servers, \
            f"Not all servers recovered: {available_servers}/{env.num_servers}"
        
        # Final verification: provision new rollouts to ensure system is healthy
        verification_rollouts = env.num_servers
        verification_tasks = []
        
        for i in range(verification_rollouts):
            rollout_id = f"post-oom-verify-{i}"
            verification_tasks.append(verify_healthy_rollout(env, rollout_id))
        
        await asyncio.gather(*verification_tasks)
    
    # Extended timeout for OOM test
    await asyncio.wait_for(run_test(), timeout=600.0)


async def verify_healthy_rollout(env: ParallelMcpEnv, rollout_id: str) -> None:
    """Helper to verify a server can handle a simple rollout."""
    await env.init_rollout(rollout_id)
    result = await env.run_tool(rollout_id, "hello_world", name="Recovery")
    assert "Hello, Recovery!" in result
    await env.release_rollout(rollout_id)