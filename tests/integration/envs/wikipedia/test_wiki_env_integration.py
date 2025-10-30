import os
import random
import asyncio
import pytest
from typing import Dict
from dotenv import load_dotenv
from datasets import DatasetDict

from benchmax.envs.wikipedia.wiki_env import WikipediaEnv

load_dotenv()


@pytest.fixture(scope="session")
def wiki_dataset() -> DatasetDict:
    """Load bamboogle dataset once for entire test session."""
    dataset, _ = WikipediaEnv.load_dataset("chiayewken/bamboogle")
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected DatasetDict from load_dataset")
    return dataset


@pytest.fixture
def wiki_env() -> WikipediaEnv:
    """Create WikipediaEnv instance with API keys from environment."""
    api_keys = os.getenv("WIKIPEDIA_API_KEYS")
    keys_list = api_keys.split(",") if api_keys else None
    return WikipediaEnv(wikipedia_api_keys=keys_list)


@pytest.fixture
def sample_dataset_example(wiki_dataset: DatasetDict) -> Dict[str, str]:
    """Return a random example from the dataset."""
    test_split = wiki_dataset["test"]
    idx = random.randint(0, len(test_split) - 1)
    return test_split[idx]


class TestWikipediaDataset:
    @pytest.mark.slow
    def test_load_dataset(
        self, wiki_dataset: DatasetDict, wiki_env: WikipediaEnv
    ) -> None:
        # Verify structure
        assert "test" in wiki_dataset, "Dataset should contain 'test' split"

        # Check test split
        test_split = wiki_dataset["test"]
        for field in ["Question", "Answer"]:
            assert field in test_split.column_names, f"Dataset should have '{field}' field"

        # Verify minimum row count
        assert len(test_split) >= 5

        # Sanity check: verify first example can be preprocessed
        first_example = test_split[0]
        standardized = WikipediaEnv.dataset_preprocess(first_example)
        assert standardized["prompt"] == first_example["Question"]
        assert standardized["ground_truth"] == first_example["Answer"]


class TestWikipediaTools:
    @pytest.mark.asyncio
    async def test_list_tools(self, wiki_env: WikipediaEnv):
        tools = await wiki_env.list_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 2
        # Expect search and article tools to be present (names may vary)
        tool_names = [t.name for t in tools]

        assert any("search" in n.lower() for n in tool_names), f"Tools: {tool_names}"
        assert any("article" in n.lower() for n in tool_names), f"Tools: {tool_names}"

    @pytest.mark.asyncio
    async def test_run_invalid_tool_raises(self, wiki_env: WikipediaEnv):
        with pytest.raises(Exception):
            await wiki_env.run_tool(
                rollout_id="tool_test", tool_name="nonexistent_tool"
            )

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_wikipedia_real_query(self, wiki_env: WikipediaEnv) -> None:
        """Test real Wikipedia search API call."""
        result = await wiki_env.run_tool(
            rollout_id="integration_test",
            tool_name="search_wikipedia",
            q="Python programming",
            limit=5,
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain search results about Python programming
        assert "Python" in result or "programming" in result.lower()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_article_real_title(self, wiki_env: WikipediaEnv) -> None:
        """Test real Wikipedia article fetch API call."""
        result = await wiki_env.run_tool(
            rollout_id="integration_test",
            tool_name="get_wikipedia_article",
            title="Python (programming language)",
            max_chars=1000,
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Article should contain relevant information about Python
        assert "programming" in result.lower() or "language" in result.lower()


class TestWikipediaReward:
    @pytest.mark.asyncio
    async def test_reward_correct_answer(
        self, wiki_env: WikipediaEnv, unique_rollout_id: str
    ):
        completion = "<answer>Berlin</answer>"
        reward = await wiki_env.compute_reward(
            unique_rollout_id, completion, ground_truth="Berlin"
        )
        assert reward["text_match"] == 1.0

    @pytest.mark.asyncio
    async def test_reward_incorrect_answer(
        self, wiki_env: WikipediaEnv, unique_rollout_id: str
    ):
        completion = "<answer>Paris</answer>"
        reward = await wiki_env.compute_reward(
            unique_rollout_id, completion, ground_truth="Berlin"
        )
        assert reward["text_match"] == 0.0

    @pytest.mark.asyncio
    async def test_reward_missing_answer_tag(
        self, wiki_env: WikipediaEnv, unique_rollout_id: str
    ):
        completion = "The capital is Berlin."
        reward = await wiki_env.compute_reward(
            unique_rollout_id, completion, ground_truth="Berlin"
        )
        assert reward["text_match"] == 0.0

    @pytest.mark.asyncio
    async def test_reward_whitespace_and_case_insensitivity(
        self, wiki_env: WikipediaEnv, unique_rollout_id: str
    ):
        completion = "  <answer> berlin </answer> "
        reward = await wiki_env.compute_reward(
            unique_rollout_id, completion, ground_truth="Berlin"
        )
        assert reward["text_match"] == 1.0


class TestWikipediaEnvE2E:
    """End-to-end tests simulating full workflows."""

    async def _run_single_rollout(
        self,
        wiki_env: WikipediaEnv,
        example: Dict[str, str],
        rollout_id: str,
        provide_correct_answer: bool = True,
    ) -> float:
        """
        Helper method to run a single rollout from start to finish.

        Args:
            wiki_env: WikipediaEnv instance
            example: Raw dataset example with Question and Answer
            rollout_id: Unique identifier for this rollout
            provide_correct_answer: If True, use ground_truth; if False, use wrong answer

        Returns:
            Reward score (0.0 or 1.0)
        """
        # Preprocess
        standardized = WikipediaEnv.dataset_preprocess(example)

        # Init rollout
        await wiki_env.init_rollout(rollout_id)

        # List tools
        tools = await wiki_env.list_tools()
        assert len(tools) == 2

        # Run search tool
        search_result = await wiki_env.run_tool(
            rollout_id=rollout_id,
            tool_name="search_wikipedia",
            q=standardized["prompt"].split()[:3],  # Use first few words as search query
            limit=3,
        )
        assert isinstance(search_result, str)

        # Create completion with answer
        if provide_correct_answer:
            completion = (
                f"Based on my research, <answer>{standardized['ground_truth']}</answer>"
            )
        else:
            completion = f"Based on my research, <answer>Incorrect Answer</answer>"

        # Compute reward
        rewards = await wiki_env.compute_reward(
            rollout_id=rollout_id,
            completion=completion,
            ground_truth=standardized["ground_truth"],
        )

        return rewards["text_match"]

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_e2e_single_rollout_correct_answer(
        self, wiki_env: WikipediaEnv, sample_dataset_example: Dict[str, str]
    ) -> None:
        """Test full workflow with correct answer returns reward of 1.0."""
        reward = await self._run_single_rollout(
            wiki_env=wiki_env,
            example=sample_dataset_example,
            rollout_id="e2e_correct",
            provide_correct_answer=True,
        )

        assert reward == 1.0, f"Expected reward 1.0 for correct answer, got {reward}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_e2e_single_rollout_incorrect_answer(
        self, wiki_env: WikipediaEnv, sample_dataset_example: Dict[str, str]
    ) -> None:
        """Test full workflow with incorrect answer returns reward of 0.0."""
        reward = await self._run_single_rollout(
            wiki_env=wiki_env,
            example=sample_dataset_example,
            rollout_id="e2e_incorrect",
            provide_correct_answer=False,
        )

        assert reward == 0.0, f"Expected reward 0.0 for incorrect answer, got {reward}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_e2e_multiple_rollouts_parallel(
        self, wiki_env: WikipediaEnv, wiki_dataset: DatasetDict
    ) -> None:
        """Test multiple rollouts running in parallel with mixed correct/incorrect answers."""
        # Get 5 random examples from dataset
        test_split = wiki_dataset["test"]
        num_examples = min(5, len(test_split))
        indices = random.sample(range(len(test_split)), num_examples)
        examples = [test_split[i] for i in indices]

        # Create tasks for parallel execution
        # Mix of correct and incorrect answers: [True, False, True, False, True]
        tasks = []
        for i, example in enumerate(examples):
            provide_correct = i % 2 == 0  # Alternate correct/incorrect
            task = self._run_single_rollout(
                wiki_env=wiki_env,
                example=example,
                rollout_id=f"parallel_rollout_{i}",
                provide_correct_answer=provide_correct,
            )
            tasks.append(task)

        # Execute all rollouts in parallel
        rewards = await asyncio.gather(*tasks)

        # Verify all rollouts completed
        assert len(rewards) == num_examples

        # Verify reward distribution matches expectations
        for i, reward in enumerate(rewards):
            expected_reward = 1.0 if (i % 2 == 0) else 0.0
            assert (
                reward == expected_reward
            ), f"Rollout {i}: expected {expected_reward}, got {reward}"

        # Verify we have both correct and incorrect answers
        assert 1.0 in rewards, "Should have at least one correct answer"
        assert 0.0 in rewards, "Should have at least one incorrect answer"
