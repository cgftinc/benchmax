"""
Integration and end-to-end tests for CRMEnv.

"""

import asyncio
import importlib.util
import json
import random
import pytest
import uuid
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any, cast

from benchmax.envs.crm.crm_env import CRMEnv
from benchmax.envs.mcp.provisioners.local_provisioner import LocalProvisioner
from benchmax.envs.mcp.provisioners.manual_provisioner import ManualProvisioner


# Fixtures
@pytest.fixture(scope="session")
def crm_workdir() -> Path:
    """Path to CRM MCP workdir."""
    spec = importlib.util.find_spec("benchmax.envs.crm")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate benchmax.envs.crm package")

    crm_pkg_dir = Path(spec.submodule_search_locations[0])
    workdir = crm_pkg_dir / "workdir"

    if not workdir.exists():
        raise FileNotFoundError(f"Expected workdir not found at: {workdir}")

    return workdir


@pytest.fixture(scope="session")
async def local_crm_servers(
    crm_workdir: Path,
) -> AsyncGenerator[Tuple[List[str], str], None]:
    """
    Provision local CRM servers once for entire test session.
    Returns (addresses, api_secret) tuple.
    """
    api_secret = uuid.uuid4().hex
    provisioner = LocalProvisioner(
        workdir_path=crm_workdir,
        num_servers=4,
        base_port=8100,
    )

    addresses = await provisioner.provision_servers(api_secret)

    yield addresses, api_secret

    await provisioner.teardown()


@pytest.fixture
async def env(
    local_crm_servers: Tuple[List[str], str],
    crm_workdir: Path,
) -> AsyncGenerator[CRMEnv, None]:
    """
    Create fresh CRMEnv using reused local servers.
    Each test gets clean env instance.
    """
    addresses, api_secret = local_crm_servers

    manual_provisioner = ManualProvisioner(addresses)
    env = CRMEnv(
        workdir_path=crm_workdir,
        provisioner=manual_provisioner,
        api_secret=api_secret,
        provision_at_init=True,
    )

    yield env

    await env.shutdown()


@pytest.fixture(scope="session")
def crm_dataset() -> Dataset:
    """Load CRMArenaPro dataset once for entire test session."""
    # Load the b2b split for testing (smaller and sufficient for tests)
    dataset_dict = load_dataset("Salesforce/CRMArenaPro", "CRMArenaPro")

    # Verify it's a DatasetDict
    assert isinstance(
        dataset_dict, DatasetDict
    ), "Expected DatasetDict from load_dataset"

    # Get the b2b split
    dataset = dataset_dict["b2b"]

    # Verify it's a Dataset
    assert isinstance(dataset, Dataset), "Expected Dataset instance"

    return dataset


@pytest.fixture
def sample_dataset_example(crm_dataset: Dataset) -> Dict[str, Any]:
    """Return a random example from the dataset."""
    idx = random.randint(0, len(crm_dataset) - 1)
    return crm_dataset[idx]


@pytest.fixture(scope="session")
def exact_match_examples(crm_dataset: Dataset) -> List[Dict[str, Any]]:
    """Return an example with exact_match reward metric."""
    exact_match_examples: List[Dict[str, Any]] = []
    for i in range(len(crm_dataset)):
        ex = crm_dataset[i]
        ex_dict = cast(Dict[str, Any], ex)
        if ex_dict.get("reward_metric") == "exact_match":
            exact_match_examples.append(ex_dict)

    if not exact_match_examples:
        pytest.skip("No exact_match examples in dataset")
    return exact_match_examples

@pytest.fixture
def exact_match_example(exact_match_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return random.choice(exact_match_examples)

@pytest.fixture(scope="session")
def exact_match_example_multiple_ground_truth_items(
    exact_match_examples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return an example with exact_match reward metric with multiple ground truth item"""
    for item in exact_match_examples:
        ground_truth: Optional[List] = item.get("answer")
        if ground_truth and len(ground_truth) > 1:
            return item

    pytest.skip("No exact_match examples with multiple ground truth found in dataset")


@pytest.fixture(scope="session")
def fuzzy_match_examples(crm_dataset: Dataset) -> List[Dict[str, Any]]:
    """Return examples with fuzzy_match reward metric."""
    fuzzy_match_examples: List[Dict[str, Any]] = []
    for i in range(len(crm_dataset)):
        ex = crm_dataset[i]
        ex_dict = cast(Dict[str, Any], ex)
        if ex_dict.get("reward_metric") == "fuzzy_match":
            fuzzy_match_examples.append(ex_dict)

    if not fuzzy_match_examples:
        pytest.skip("No fuzzy_match examples in dataset")
    return fuzzy_match_examples

@pytest.fixture
def fuzzy_match_example(fuzzy_match_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return random.choice(fuzzy_match_examples)


class TestCRMDataset:
    @pytest.mark.slow
    def test_load_dataset(self, crm_dataset: Dataset) -> None:
        """Verify dataset loads correctly with all required fields."""
        # Verify minimum row count
        assert len(crm_dataset) > 0, "Dataset should not be empty"

        # Check required fields in dataset
        first_example = crm_dataset[0]
        required_fields = ["idx", "answer", "task", "persona", "query", "reward_metric"]
        for field in required_fields:
            assert field in first_example, f"Dataset should have '{field}' field"

        # Verify answer is a list
        assert isinstance(first_example["answer"], list), "Answer should be a list"

        # Verify reward_metric is valid
        assert first_example["reward_metric"] in [
            "exact_match",
            "fuzzy_match",
        ], "reward_metric should be 'exact_match' or 'fuzzy_match'"

    @pytest.mark.slow
    def test_dataset_preprocess(self, crm_dataset: Dataset, env: CRMEnv) -> None:
        """Verify dataset preprocessing produces valid CRMExample."""
        first_example = crm_dataset[0]

        standardized = CRMEnv.dataset_preprocess(first_example)

        # Check StandardizedExample fields
        assert standardized["prompt"] is not None
        assert standardized["ground_truth"] is not None
        assert isinstance(standardized["ground_truth"], list)

        # Check that prompt contains expected components
        assert first_example["persona"] in standardized["prompt"]
        assert first_example["task"] in standardized["prompt"]
        assert first_example["query"] in standardized["prompt"]

        # Check CRMExample specific fields
        assert standardized["reward_metric"] == first_example["reward_metric"]
        assert standardized["init_rollout_args"] == None


class TestCRMTools:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_list_tools(self, env: CRMEnv) -> None:
        """Verify Salesforce tools are in tool list."""
        tools = await env.list_tools()
        tool_names = [tool.name for tool in tools]

        # Check for key Salesforce tools
        expected_tools = [
            "get_cases",
            "search_knowledge_articles",
            "get_start_date",
            "get_period",
            "calculate_average_handle_time",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Should have '{tool_name}' tool available"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_start_date_tool(
        self, env: CRMEnv, unique_rollout_id: str
    ) -> None:
        """Test get_start_date tool with various inputs."""
        await env.init_rollout(unique_rollout_id)

        # Test valid inputs
        result = await env.run_tool(
            unique_rollout_id,
            "get_start_date",
            end_date="2024-01-31T00:00:00Z",
            period="month",
            interval_count=1,
        )

        assert isinstance(result, str)
        assert "2023-12-31T00:00:00Z" in result

        # Clean up
        await env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_period_tool(
        self, env: CRMEnv, unique_rollout_id: str
    ) -> None:
        """Test get_period tool with various period types."""
        await env.init_rollout(unique_rollout_id)

        # Test with month
        result_str = await env.run_tool(
            unique_rollout_id, "get_period", period_name="January", year=2024
        )

        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert "start_date" in result
        assert "end_date" in result
        assert "2024-01-01" in result["start_date"]

        # Test with quarter
        result_str = await env.run_tool(
            unique_rollout_id, "get_period", period_name="Q1", year=2024
        )

        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert "start_date" in result
        assert "end_date" in result
        assert "2024-01-01" in result["start_date"]

        # Clean up
        await env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_find_max_min_value_tools(
        self, env: CRMEnv, unique_rollout_id: str
    ) -> None:
        """Test find_id_with_max_value and find_id_with_min_value tools."""
        await env.init_rollout(unique_rollout_id)

        test_data = {"agent1": 10, "agent2": 25, "agent3": 15}

        # Test max value
        result = await env.run_tool(
            unique_rollout_id, "find_id_with_max_value", values_by_id=test_data
        )

        assert isinstance(result, str)
        assert "agent2" in result

        # Test min value
        result = await env.run_tool(
            unique_rollout_id, "find_id_with_min_value", values_by_id=test_data
        )

        assert isinstance(result, str)
        assert "agent1" in result

        # Clean up
        await env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_calculate_average_handle_time_tool(
        self, env: CRMEnv, unique_rollout_id: str
    ) -> None:
        """Test calculate_average_handle_time tool."""
        await env.init_rollout(unique_rollout_id)

        test_cases = [
            {
                "CreatedDate": "2024-01-01T10:00:00.000+0000",
                "ClosedDate": "2024-01-01T11:30:00.000+0000",
                "OwnerId": "agent1",
            },
            {
                "CreatedDate": "2024-01-02T09:00:00.000+0000",
                "ClosedDate": "2024-01-02T10:00:00.000+0000",
                "OwnerId": "agent1",
            },
            {
                "CreatedDate": "2024-01-01T14:00:00.000+0000",
                "ClosedDate": "2024-01-01T16:00:00.000+0000",
                "OwnerId": "agent2",
            },
        ]

        result_str = await env.run_tool(
            unique_rollout_id, "calculate_average_handle_time", cases=test_cases
        )

        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert "agent1" in result
        assert "agent2" in result
        # agent1 average: (90 + 60) / 2 = 75 minutes
        assert 70 <= result["agent1"] <= 80
        # agent2 average: 120 minutes
        assert result["agent2"] == 120.0

        # Clean up
        await env.release_rollout(unique_rollout_id)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_month_to_case_count_tool(
        self, env: CRMEnv, unique_rollout_id: str
    ) -> None:
        """Test get_month_to_case_count tool."""
        await env.init_rollout(unique_rollout_id)

        test_cases = [
            {"CreatedDate": "2024-01-15T10:00:00.000+0000"},
            {"CreatedDate": "2024-01-20T10:00:00.000+0000"},
            {"CreatedDate": "2024-02-10T10:00:00.000+0000"},
            {"CreatedDate": "2024-03-05T10:00:00.000+0000"},
            {"CreatedDate": "2024-03-25T10:00:00.000+0000"},
        ]

        result_str = await env.run_tool(
            unique_rollout_id, "get_month_to_case_count", cases=test_cases
        )

        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert result.get("January") == 2
        assert result.get("February") == 1
        assert result.get("March") == 2

        # Clean up
        await env.release_rollout(unique_rollout_id)


class TestCRMReward:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_exact_match_perfect(
        self,
        env: CRMEnv,
        exact_match_example: Dict[str, Any],
        unique_rollout_id: str,
    ) -> None:
        """Test exact match reward with perfect answer."""
        standardized = CRMEnv.dataset_preprocess(exact_match_example)

        await env.init_rollout(unique_rollout_id)

        # Create completion with exact ground truth
        ground_truth_items = standardized["ground_truth"]
        completion = (
            f"<answer>{', '.join(str(item) for item in ground_truth_items)}</answer>"
        )

        rewards = await env.compute_reward(
            unique_rollout_id,
            completion=completion,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        assert "match" in rewards
        assert rewards["match"] == 1.0, "Perfect exact match should yield reward of 1.0"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_exact_match_partial(
        self,
        env: CRMEnv,
        exact_match_example_multiple_ground_truth_items: Dict[str, Any],
        unique_rollout_id: str,
    ) -> None:
        """Test exact match reward with partial overlap."""
        standardized = CRMEnv.dataset_preprocess(exact_match_example_multiple_ground_truth_items)

        await env.init_rollout(unique_rollout_id)

        # Create completion with only some ground truth items
        ground_truth_items: List[str] = standardized["ground_truth"]
        completion = f"<answer>{ground_truth_items[0]}</answer>"

        rewards = await env.compute_reward(
            unique_rollout_id,
            completion=completion,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        assert "match" in rewards
        assert (
            0.0 < rewards["match"] < 1.0
        ), "Partial match should yield reward between 0 and 1"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_exact_match_wrong(
        self,
        env: CRMEnv,
        exact_match_example: Dict[str, Any],
        unique_rollout_id: str,
    ) -> None:
        """Test exact match reward with wrong answer."""
        standardized = CRMEnv.dataset_preprocess(exact_match_example)

        await env.init_rollout(unique_rollout_id)

        # Create completion with wrong answer
        completion = "<answer>completely_wrong_answer_12345</answer>"

        rewards = await env.compute_reward(
            unique_rollout_id,
            completion=completion,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        assert "match" in rewards
        assert rewards["match"] < 0.5, "Wrong answer should yield low reward"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_fuzzy_match_exact(
        self,
        env: CRMEnv,
        fuzzy_match_example: Dict[str, Any],
        unique_rollout_id: str,
    ) -> None:
        """Test fuzzy match reward with exact answer."""
        standardized = CRMEnv.dataset_preprocess(fuzzy_match_example)

        await env.init_rollout(unique_rollout_id)

        # Get ground truth (should be single item for fuzzy match)
        ground_truth = str(standardized["ground_truth"][0])

        # Test exact match
        completion_exact = f"<answer>{ground_truth}</answer>"
        rewards = await env.compute_reward(
            unique_rollout_id,
            completion=completion_exact,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        assert "match" in rewards
        assert rewards["match"] > 0.8, "Exact fuzzy match should yield high reward"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_fuzzy_match_partial(
        self,
        env: CRMEnv,
        fuzzy_match_example: Dict[str, Any],
        unique_rollout_id: str,
    ) -> None:
        """Test fuzzy match reward with partial answer."""
        standardized = CRMEnv.dataset_preprocess(fuzzy_match_example)

        await env.init_rollout(unique_rollout_id)

        # Get ground truth (should be single item for fuzzy match)
        ground_truth_items: List[str] = standardized["ground_truth"]

        # Test partial match (if ground truth has multiple words)
        words = ground_truth_items[0].split()
        partial_answer = " ".join(words[: len(words) // 2])
        completion_partial = f"<answer>{partial_answer}</answer>"
        rewards = await env.compute_reward(
            unique_rollout_id,
            completion=completion_partial,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        assert "match" in rewards
        assert (
            0.0 < rewards["match"] < 1.0
        ), "Partial fuzzy match should yield intermediate reward"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reward_no_answer_tags(
        self,
        env: CRMEnv,
        sample_dataset_example: Dict[str, Any],
        unique_rollout_id: str,
    ) -> None:
        """Test reward when completion has no answer tags."""
        standardized = CRMEnv.dataset_preprocess(sample_dataset_example)

        
        await env.init_rollout(unique_rollout_id)

        # Completion without answer tags
        completion = "I think the answer is something but I'm not sure"

        rewards = await env.compute_reward(
            unique_rollout_id,
            completion=completion,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        assert "match" in rewards
        assert rewards["match"] == 0.0, "Missing answer tags should yield 0 reward"

        await env.release_rollout(unique_rollout_id)


class TestCRMEndToEnd:
    """End-to-end tests for full CRMEnv workflows."""

    async def _run_single_rollout(
        self,
        rollout_id: str,
        env: CRMEnv,
        example: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Execute full rollout workflow:
        1. Preprocess example
        2. Init rollout
        3. Run some tools
        4. Compute reward with dummy completion

        Returns reward dict.
        """
        # Preprocess
        standardized = CRMEnv.dataset_preprocess(example)

        # Init rollout
        await env.init_rollout(rollout_id)

        # Run a few tools to verify environment is working
        result1 = await env.run_tool(
            rollout_id, "get_period", period_name="January", year=2024
        )

        # Verify tool returned something
        assert result1 is not None

        result2 = await env.run_tool(
            rollout_id,
            "get_start_date",
            end_date="2024-06-30T00:00:00Z",
            period="month",
            interval_count=3,
        )
        assert result2 is not None

        result3 = await env.run_tool(
            rollout_id,
            "find_id_with_max_value",
            values_by_id={"a": 1, "b": 5, "c": 3},
        )
        assert result3 is not None

        # Create a dummy completion (wrong answer)
        completion = "<answer>dummy_answer</answer>"

        # Compute reward
        rewards = await env.compute_reward(
            rollout_id,
            completion=completion,
            ground_truth=standardized["ground_truth"],
            reward_metric=standardized["reward_metric"],
        )

        return rewards

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rollout_basic_workflow(
        self, env: CRMEnv, crm_dataset: Dataset
    ) -> None:
        """Full rollout with basic workflow."""
        first_example = crm_dataset[0]

        reward = await self._run_single_rollout(
            rollout_id="test-rollout-basic",
            env=env,
            example=first_example,
        )

        assert "match" in reward
        # Dummy answer should yield low reward
        assert 0.0 <= reward["match"] <= 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_parallel_rollouts(
        self, env: CRMEnv, crm_dataset: Dataset
    ) -> None:
        """Multiple rollouts should execute correctly in parallel."""
        # Get three examples
        num_examples = min(3, len(crm_dataset))
        examples = [crm_dataset[i] for i in range(num_examples)]

        results = await asyncio.gather(
            *[
                self._run_single_rollout(
                    rollout_id=f"test-rollout-parallel-{i}",
                    env=env,
                    example=examples[i],
                )
                for i in range(num_examples)
            ]
        )

        # All should return reward dicts
        assert len(results) == num_examples
        for reward in results:
            assert "match" in reward
            assert 0.0 <= reward["match"] <= 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_system_prompt_included(self, env: CRMEnv) -> None:
        """Verify system prompt is correctly set."""
        assert env.system_prompt is not None
        assert len(env.system_prompt) > 0
        assert "Salesforce" in env.system_prompt
        assert "<answer>" in env.system_prompt

