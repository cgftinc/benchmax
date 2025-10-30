"""
Unit tests for CRMEnv.

All tests are fast with no external service calls.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

from benchmax.envs.crm.crm_env import CRMEnv
from benchmax.envs.crm.workdir.reward_fn import (
    crm_matching_reward_function,
    parse_answers,
    parse_text_to_tokens,
)
from benchmax.envs.mcp.provisioners.base_provisioner import BaseProvisioner


# Fixtures
@pytest.fixture
def crm_env(tmp_path: Path) -> CRMEnv:
    """Fixture to create a CRMEnv instance without initializing the parent."""
    env = CRMEnv(
        workdir_path=tmp_path,
        provisioner=Mock(spec=BaseProvisioner),
        provision_at_init=False,
    )
    env._servers_provisioned = True
    return env


@pytest.fixture
def mock_mcp_client() -> Mock:
    return Mock()


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestDatasetPreprocess:
    """Tests for CRMEnv.dataset_preprocess."""

    def test_valid_example(self, crm_env: CRMEnv) -> None:
        """Valid example returns a CRMExample with expected fields."""
        example: Dict[str, Any] = {
            "task": "Find all open cases",
            "persona": "You are a sales manager",
            "query": "What are the case IDs?",
            "answer": ["Case001", "Case002"],
            "reward_metric": "exact_match",
        }

        result = crm_env.dataset_preprocess(example)

        assert result is not None
        assert "You are a sales manager" in result["prompt"]
        assert "Find all open cases" in result["prompt"]
        assert "What are the case IDs?" in result["prompt"]
        assert result["ground_truth"] == ["Case001", "Case002"]
        assert result["reward_metric"] == "exact_match"
        assert result["init_rollout_args"] is None

    def test_valid_example_with_metadata(self, crm_env: CRMEnv) -> None:
        """Valid example with metadata.required includes it in prompt."""
        example: Dict[str, Any] = {
            "task": "Find accounts",
            "persona": "You are an analyst",
            "query": "List account names",
            "answer": ["Acme Corp"],
            "reward_metric": "fuzzy_match",
            "metadata": {
                "required": "Only include active accounts",
                "optional": "This should be ignored",
            },
        }

        result = crm_env.dataset_preprocess(example)

        assert "You are an analyst" in result["prompt"]
        assert "Find accounts" in result["prompt"]
        assert "Only include active accounts" in result["prompt"]
        assert "List account names" in result["prompt"]
        assert result["ground_truth"] == ["Acme Corp"]

    def test_missing_task_raises(self, crm_env: CRMEnv) -> None:
        """Missing task field should raise ValueError."""
        example: Dict[str, Any] = {
            "persona": "You are a manager",
            "query": "What?",
            "answer": ["Something"],
            "reward_metric": "exact_match",
        }

        with pytest.raises(ValueError, match="task"):
            crm_env.dataset_preprocess(example)

    def test_missing_persona_raises(self, crm_env: CRMEnv) -> None:
        """Missing persona field should raise ValueError."""
        example: Dict[str, Any] = {
            "task": "Do something",
            "query": "What?",
            "answer": ["Something"],
            "reward_metric": "exact_match",
        }

        with pytest.raises(ValueError, match="persona"):
            crm_env.dataset_preprocess(example)

    def test_missing_query_raises(self, crm_env: CRMEnv) -> None:
        """Missing query field should raise ValueError."""
        example: Dict[str, Any] = {
            "task": "Do something",
            "persona": "You are a manager",
            "answer": ["Something"],
            "reward_metric": "exact_match",
        }

        with pytest.raises(ValueError, match="query"):
            crm_env.dataset_preprocess(example)

    def test_none_answer_raises(self, crm_env: CRMEnv) -> None:
        """Answer field being None should raise ValueError."""
        example: Dict[str, Any] = {
            "task": "Do something",
            "persona": "You are a manager",
            "query": "What?",
            "answer": None,
            "reward_metric": "exact_match",
        }

        with pytest.raises(ValueError, match="answer"):
            crm_env.dataset_preprocess(example)

    def test_missing_reward_metric_raises(self, crm_env: CRMEnv) -> None:
        """Missing reward_metric field should raise ValueError."""
        example: Dict[str, Any] = {
            "task": "Do something",
            "persona": "You are a manager",
            "query": "What?",
            "answer": ["Something"],
        }

        with pytest.raises(ValueError, match="reward_metric"):
            crm_env.dataset_preprocess(example)

    def test_empty_answer_list(self, crm_env: CRMEnv) -> None:
        """Empty answer list should be valid."""
        example: Dict[str, Any] = {
            "task": "Find cases",
            "persona": "You are a manager",
            "query": "Any open cases?",
            "answer": [],
            "reward_metric": "exact_match",
        }

        result = crm_env.dataset_preprocess(example)

        assert result["ground_truth"] == []


class TestRewardComputation:
    """Test reward computation for CRM matching."""

    def test_exact_match_perfect(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Perfect match returns 1.0."""
        score = crm_matching_reward_function(
            completion="<answer>apple</answer>",
            ground_truth=["apple"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        assert score == 1.0

    def test_exact_match_partial(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Partial match returns IoU score."""
        score = crm_matching_reward_function(
            completion="<answer>apple banana</answer>",
            ground_truth=["apple", "banana", "cherry"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        # IoU: intersection={apple, banana}, union={apple, banana, cherry}
        # IoU = 2/3 ≈ 0.667
        assert abs(score - 2 / 3) < 0.01

    def test_exact_match_no_match(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """No overlap returns 0.0."""
        score = crm_matching_reward_function(
            completion="<answer>zebra</answer>",
            ground_truth=["apple", "banana"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        assert score == 0.0

    def test_exact_match_both_empty(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Empty completion and ground truth returns 1.0."""
        score = crm_matching_reward_function(
            completion="<answer></answer>",
            ground_truth=[],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        assert score == 1.0

    def test_exact_match_completion_empty(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Empty completion with non-empty ground truth returns 0.0."""
        score = crm_matching_reward_function(
            completion="<answer></answer>",
            ground_truth=["apple"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        assert score == 0.0

    def test_exact_match_ground_truth_empty(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Non-empty completion with empty ground truth returns 0.0."""
        score = crm_matching_reward_function(
            completion="<answer>apple</answer>",
            ground_truth=[],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        assert score == 0.0

    def test_fuzzy_match_perfect(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Perfect fuzzy match returns 1.0."""
        score = crm_matching_reward_function(
            completion="<answer>hello world</answer>",
            ground_truth=["hello world"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="fuzzy_match",
        )

        assert score == 1.0

    def test_fuzzy_match_partial(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Partial fuzzy match returns F1 score > 0 and < 1."""
        score = crm_matching_reward_function(
            completion="<answer>hello world test</answer>",
            ground_truth=["hello world"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="fuzzy_match",
        )

        assert 0 < score < 1

    def test_unknown_metric(
        self, mock_mcp_client: Mock, mock_workspace: Path, capsys
    ) -> None:
        """Unknown reward metric returns 0.0 and prints warning."""
        score = crm_matching_reward_function(
            completion="<answer>test</answer>",
            ground_truth=["test"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="unknown_metric",
        )

        assert score == 0.0
        captured = capsys.readouterr()
        assert "Unknown reward metric" in captured.out

    def test_missing_reward_metric(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        """Missing reward_metric in kwargs raises ValueError."""
        with pytest.raises(ValueError, match="reward metric"):
            crm_matching_reward_function(
                completion="<answer>test</answer>",
                ground_truth=["test"],
                mcp_client=mock_mcp_client,
                workspace=mock_workspace,
                # missing reward_metric
            )

    def test_no_answer_tags(self, mock_mcp_client: Mock, mock_workspace: Path) -> None:
        """Completion without answer tags results in empty parsed answer."""
        score = crm_matching_reward_function(
            completion="just some text without tags",
            ground_truth=["apple"],
            mcp_client=mock_mcp_client,
            workspace=mock_workspace,
            reward_metric="exact_match",
        )

        # Empty parsed answer vs non-empty ground truth = 0.0
        assert score == 0.0


class TestRewardComputationHelper:
    """Test helper functions for reward computation."""

    def test_parse_answers_valid(self) -> None:
        """Extract text from answer tags."""
        result = parse_answers("<answer>apple banana</answer>")
        assert result == "apple banana"

    def test_parse_answers_no_tags(self) -> None:
        """Return empty string when no tags present."""
        result = parse_answers("just some text")
        assert result == ""

    def test_parse_answers_case_insensitive(self) -> None:
        """Handle case variations in tags."""
        result1 = parse_answers("<ANSWER>test</ANSWER>")
        result2 = parse_answers("<AnSwEr>test</AnSwEr>")
        assert result1 == "test"
        assert result2 == "test"

    def test_parse_answers_multiline(self) -> None:
        """Handle multiline content."""
        result = parse_answers("<answer>\n  apple\n  banana\n</answer>")
        assert "apple" in result
        assert "banana" in result

    def test_parse_answers_xml_entities(self) -> None:
        """Unescape XML entities."""
        result = parse_answers("<answer>rock &amp; roll</answer>")
        assert result == "rock & roll"

    def test_parse_text_to_tokens_simple_space(self) -> None:
        """Simple space-separated tokens."""
        assert parse_text_to_tokens("hello world test") == {"hello", "world", "test"}

    def test_parse_text_to_tokens_comma_separation(self) -> None:
        """Comma-separated tokens."""
        assert parse_text_to_tokens("apple,banana,cherry") == {
            "apple",
            "banana",
            "cherry",
        }

    def test_parse_text_to_tokens_mixed_separators(self) -> None:
        """Mixed separators (spaces, commas, pipes, tabs)."""
        assert parse_text_to_tokens("apple, banana cherry | orange\tgrape") == {
            "apple",
            "banana",
            "cherry",
            "orange",
            "grape",
        }

    def test_parse_text_to_tokens_case_normalization(self) -> None:
        """Normalize to lowercase."""
        assert parse_text_to_tokens("HELLO World TeSt") == {"hello", "world", "test"}

    def test_parse_text_to_tokens_quote_removal(self) -> None:
        """Remove quotes from text."""
        assert parse_text_to_tokens('"hello world"') == {"hello", "world"}

    def test_parse_text_to_tokens_empty_string(self) -> None:
        """Empty string returns empty set."""
        assert parse_text_to_tokens("") == set()

    def test_parse_text_to_tokens_whitespace_only(self) -> None:
        """Whitespace-only string returns empty set."""
        assert parse_text_to_tokens("   \t\n  ") == set()
