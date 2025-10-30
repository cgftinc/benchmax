"""
Unit tests for MathEnv.

All tests are fast with no external service calls.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock
from fastmcp import Client

from benchmax.envs.math.math_env import MathEnv
from benchmax.envs.math.workdir.reward_fn import text_match_reward
from benchmax.envs.mcp.provisioners.base_provisioner import BaseProvisioner


# Fixtures
@pytest.fixture
def math_env(tmp_path: Path) -> MathEnv:
    """Fixture to create a MathEnv instance."""
    env = MathEnv(
        workdir_path=tmp_path,
        provisioner=Mock(spec=BaseProvisioner),
        provision_at_init=False,
    )
    return env


@pytest.fixture
def mock_mcp_client() -> Mock:
    """Mock MCP client for reward function testing."""
    return Mock(spec=Client)


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    """Mock workspace path for reward function testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


# Test Classes
class TestDatasetPreprocess:
    """Test dataset preprocessing logic."""

    def test_valid_example(self, math_env: MathEnv) -> None:
        """Test preprocessing with valid task and answer."""
        # Create a mock MathEnv that doesn't initialize the parent
        example: Dict[str, str] = {"task": "Calculate 2+2", "answer": "4"}

        result = math_env.dataset_preprocess(example)

        assert result["prompt"] == "Calculate 2+2"
        assert result["ground_truth"] == "4"
        assert result["init_rollout_args"] == None

    def test_missing_answer(self, math_env: MathEnv) -> None:
        """Test preprocessing when answer is missing."""
        example: Dict[str, str] = {"task": "1+1"}

        result = math_env.dataset_preprocess(example)

        assert result["prompt"] == "1+1"
        assert result["ground_truth"] == ""
        assert result["init_rollout_args"] == None

    def test_missing_task(self, math_env: MathEnv) -> None:
        """Test preprocessing when task is missing."""
        example: Dict[str, str] = {"answer": "5"}

        result = math_env.dataset_preprocess(example)

        assert result["prompt"] == ""
        assert result["ground_truth"] == "5"
        assert result["init_rollout_args"] == None

    def test_extra_fields_ignored(self, math_env: MathEnv) -> None:
        """Test that extra fields in example are ignored."""
        example: Dict[str, Any] = {
            "task": "3*3",
            "answer": "9",
            "metadata": "extra",
            "difficulty": "easy",
        }

        result = math_env.dataset_preprocess(example)

        assert result["prompt"] == "3*3"
        assert result["ground_truth"] == "9"
        assert result["init_rollout_args"] == None


class TestRewardComputation:
    """Test reward computation with text matching."""

    @pytest.mark.asyncio
    async def test_exact_match(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test exact numeric match in answer tags."""
        completion = "<answer>4</answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive_tags(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test that answer tags are case insensitive."""
        completion = "<ANSWER>4</ANSWER>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_float_comparison(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test float string comparison."""
        completion = "<answer>4.0</answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_float_precision(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test float with multiple decimal places."""
        completion = "<answer>3.14159</answer>"
        ground_truth = "3.14159"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_incorrect_answer(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test incorrect answer returns 0."""
        completion = "<answer>5</answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_missing_tags(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test missing answer tags returns 0."""
        completion = "The answer is 4"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_empty_tags(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test empty answer tags returns 0."""
        completion = "<answer></answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        # Empty string cannot be converted to float, should return 0.0
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_normalization(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test whitespace is normalized in answer tags."""
        completion = "<answer>  4  </answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_surrounding_text(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test answer with surrounding text."""
        completion = "Let me think. <answer>35</answer> That's correct."
        ground_truth = "35"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_multiple_tags_uses_first(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test that only first answer tag is used."""
        completion = "<answer>4</answer> No wait <answer>5</answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_negative_numbers(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test negative numbers are handled correctly."""
        completion = "<answer>-10</answer>"
        ground_truth = "-10"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_decimal_numbers(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test decimal numbers are handled correctly."""
        completion = "<answer>0.5</answer>"
        ground_truth = "0.5"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_non_numeric_answer(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Test non-numeric answer returns 0 (cannot convert to float)."""
        completion = "<answer>not a number</answer>"
        ground_truth = "4"

        reward = await text_match_reward(
            completion=completion,
            ground_truth=ground_truth,
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
        )

        # Non-numeric strings cannot be converted to float, should return 0.0
        assert reward == 0.0
