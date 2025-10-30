"""
Integration and end-to-end tests for MathEnv.

All tests are marked with @pytest.mark.slow and can make real external calls.
"""

import asyncio
import importlib.util
import random
import pytest
import uuid
from pathlib import Path
from datasets import DatasetDict
from typing import AsyncGenerator, Dict, List, Tuple

from benchmax.envs.math.math_env import MathEnv
from benchmax.envs.mcp.provisioners.local_provisioner import LocalProvisioner
from benchmax.envs.mcp.provisioners.manual_provisioner import ManualProvisioner


# Fixtures
@pytest.fixture(scope="session")
def math_workdir() -> Path:
    """Path to math MCP workdir."""
    spec = importlib.util.find_spec("benchmax.envs.math")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate benchmax.envs.math package")

    math_pkg_dir = Path(spec.submodule_search_locations[0])
    workdir = math_pkg_dir / "workdir"

    if not workdir.exists():
        raise FileNotFoundError(f"Expected workdir not found at: {workdir}")

    return workdir


@pytest.fixture(scope="session")
async def local_math_servers(
    math_workdir: Path,
) -> AsyncGenerator[Tuple[List[str], str], None]:
    """
    Provision local calculator servers once for entire test session.
    Returns (addresses, api_secret) tuple.
    """
    api_secret = uuid.uuid4().hex
    provisioner = LocalProvisioner(
        workdir_path=math_workdir,
        num_servers=4,
        base_port=8080,
    )

    addresses = await provisioner.provision_servers(api_secret)

    yield addresses, api_secret

    await provisioner.teardown()


@pytest.fixture
async def math_env(
    local_math_servers: Tuple[List[str], str], math_workdir: Path
) -> AsyncGenerator[MathEnv, None]:
    """
    Create fresh MathEnv using reused local servers.
    Each test gets clean env instance.
    """
    addresses, api_secret = local_math_servers

    manual_provisioner = ManualProvisioner(addresses)
    env = MathEnv(
        workdir_path=math_workdir,
        provisioner=manual_provisioner,
        api_secret=api_secret,
        provision_at_init=True,
    )

    yield env

    await env.shutdown()


@pytest.fixture(scope="session")
def math_dataset() -> DatasetDict:
    """Load arithmetic50 dataset once for entire test session."""
    dataset, _ = MathEnv.load_dataset("dawidmt/arithmetic50")
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected DatasetDict from load_dataset")
    return dataset


@pytest.fixture
def sample_dataset_example(math_dataset: DatasetDict) -> Dict[str, str]:
    """Return a random example from the dataset."""
    test_split = math_dataset["test"]
    idx = random.randint(0, len(test_split) - 1)
    return test_split[idx]


class TestMathDataset:
    @pytest.mark.slow
    def test_load_dataset(self, math_dataset: DatasetDict) -> None:
        # Verify structure
        assert "test" in math_dataset

        # Check test split
        test_split = math_dataset["test"]
        for field in ["task", "answer"]:
            assert (
                field in test_split.column_names
            ), f"Dataset should have '{field}' field"

        # Verify minimum row count
        assert len(test_split) > 5

        # Sanity check: verify first example can be preprocessed
        first_example = test_split[0]
        standardized = MathEnv.dataset_preprocess(first_example)
        assert standardized["prompt"] == first_example["task"]
        assert standardized["ground_truth"] == first_example["answer"]


class TestMathTools:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_list_tools(self, math_env: MathEnv) -> None:
        """Verify calculator tool is in tool list."""
        tools = await math_env.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "calculate" in tool_names, f"Calculator not found in tools: {tool_names}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_calculator_simple(
        self, math_env: MathEnv, unique_rollout_id: str
    ) -> None:
        """Test calculator with simple addition."""
        await math_env.init_rollout(unique_rollout_id)

        result = await math_env.run_tool(
            unique_rollout_id, "calculate", expression="2+2"
        )

        # Result might be string "4" or number 4
        result_str = str(result).strip()
        assert result_str == "4" or result == 4, f"Expected 4, got {result}"

        await math_env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_calculator_complex_expression(
        self, math_env: MathEnv, unique_rollout_id: str
    ) -> None:
        """Test calculator with complex expression."""
        await math_env.init_rollout(unique_rollout_id)

        result = await math_env.run_tool(
            unique_rollout_id, "calculate", expression="(10+5)*2"
        )

        result_str = str(result).strip()
        assert result_str == "30" or result == 30, f"Expected 30, got {result}"

        await math_env.release_rollout(unique_rollout_id)


class TestMathReward:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_correct_answer(
        self, math_env: MathEnv, unique_rollout_id: str
    ):
        await math_env.init_rollout(unique_rollout_id)

        completion = "<answer>42</answer>"
        reward = await math_env.compute_reward(
            unique_rollout_id, completion=completion, ground_truth="42"
        )
        assert reward["match"] == 1.0

        await math_env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_incorrect_answer(
        self, math_env: MathEnv, unique_rollout_id: str
    ):
        await math_env.init_rollout(unique_rollout_id)

        completion = "<answer>41</answer>"
        reward = await math_env.compute_reward(
            unique_rollout_id, completion=completion, ground_truth="42"
        )
        assert reward["match"] == 0.0

        await math_env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_missing_tag(self, math_env: MathEnv, unique_rollout_id: str):
        await math_env.init_rollout(unique_rollout_id)

        completion = "42"
        reward = await math_env.compute_reward(
            unique_rollout_id, completion=completion, ground_truth="42"
        )
        assert reward["match"] == 0.0

        await math_env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_whitespace_and_case_insensitivity(self, math_env: MathEnv, unique_rollout_id: str):
        await math_env.init_rollout(unique_rollout_id)

        completion = "<answer> 42 </answer>    "
        reward = await math_env.compute_reward(
            unique_rollout_id, completion=completion, ground_truth="42"
        )
        assert reward["match"] == 1.0

        await math_env.release_rollout(unique_rollout_id)


class TestMathEndToEnd:
    """End-to-end tests for full MathEnv workflows."""

    async def _run_single_rollout(
        self,
        rollout_id: str,
        env: MathEnv,
        task: str,
        ground_truth: str,
        completion: str,
    ) -> Dict[str, float]:
        """
        Execute full rollout workflow:
        1. Preprocess example
        2. Init rollout
        3. List tools (verify calculator present)
        4. Compute reward with completion

        Returns reward dict.
        """
        # Preprocess
        example = MathEnv.dataset_preprocess({"task": task, "answer": ground_truth})

        # Init rollout - only takes rollout_id
        await env.init_rollout(rollout_id)

        # List tools (no arguments)
        tools = await env.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "calculate" in tool_names, f"Calculator tool not found in {tool_names}"

        # Compute reward - returns dict with reward function results
        rewards = await env.compute_reward(
            rollout_id,
            completion=completion,
            ground_truth=example["ground_truth"],
        )

        return rewards

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rollout_correct_answer(self, math_env: MathEnv) -> None:
        """Full rollout with correct answer should return reward 1.0."""
        rewards = await self._run_single_rollout(
            rollout_id="test-rollout-correct",
            env=math_env,
            task="Calculate 5*7",
            ground_truth="35",
            completion="Let me calculate. <answer>35</answer>",
        )

        # Check the 'match' reward function result
        assert "match" in rewards
        assert rewards["match"] == 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rollout_incorrect_answer(self, math_env: MathEnv) -> None:
        """Full rollout with incorrect answer should return reward 0.0."""
        rewards = await self._run_single_rollout(
            rollout_id="test-rollout-incorrect",
            env=math_env,
            task="Calculate 3+4",
            ground_truth="7",
            completion="<answer>8</answer>",
        )

        assert "match" in rewards
        assert rewards["match"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rollout_missing_tags(self, math_env: MathEnv) -> None:
        """Rollout without answer tags should return reward 0.0."""
        rewards = await self._run_single_rollout(
            rollout_id="test-rollout-missing-tags",
            env=math_env,
            task="Calculate 10/2",
            ground_truth="5",
            completion="The answer is 5",
        )

        assert "match" in rewards
        assert rewards["match"] == 0.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_parallel_rollouts(self, math_env: MathEnv) -> None:
        """Multiple rollouts should execute correctly in parallel."""
        results = await asyncio.gather(
            # Correct answer
            self._run_single_rollout(
                rollout_id="test-rollout-parallel-1",
                env=math_env,
                task="Calculate 2+2",
                ground_truth="4",
                completion="<answer>4</answer>",
            ),
            # Incorrect answer
            self._run_single_rollout(
                rollout_id="test-rollout-parallel-2",
                env=math_env,
                task="Calculate 3*3",
                ground_truth="9",
                completion="<answer>10</answer>",
            ),
            # Correct answer
            self._run_single_rollout(
                rollout_id="test-rollout-parallel-3",
                env=math_env,
                task="Calculate 10/2",
                ground_truth="5",
                completion="<answer>5</answer>",
            ),
        )

        # Extract 'match' reward from each result
        match_rewards = [r["match"] for r in results]
        assert match_rewards == [
            1.0,
            0.0,
            1.0,
        ], f"Expected [1.0, 0.0, 1.0], got {match_rewards}"
