import pytest
from pathlib import Path
import tempfile

from benchmax.envs.crm.crm_env import parse_text_to_tokens, reward_func


@pytest.fixture(scope="session")
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ----------------------
# Exact Match Tests
# ----------------------

def test_exact_match_single_perfect(temp_workspace):
    result = reward_func(
        prompt="What fruit?",
        completion="apple",
        ground_truth=["apple"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 1.0


def test_exact_match_single_no_match(temp_workspace):
    result = reward_func(
        prompt="What fruit?",
        completion="zebra",
        ground_truth=["apple"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 0.0


def test_exact_match_single_partial(temp_workspace):
    result = reward_func(
        prompt="List fruits",
        completion="apple banana",
        ground_truth=["apple banana cherry"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert abs(result - 2/3) < 0.01


def test_exact_match_multiple_perfect(temp_workspace):
    result = reward_func(
        prompt="List colors",
        completion="red blue green yellow",
        ground_truth=["red blue", "green yellow"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 1.0


def test_exact_match_multiple_partial(temp_workspace):
    result = reward_func(
        prompt="List animals",
        completion="cat dog",
        ground_truth=["cat dog", "bird fish"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 0.5


def test_exact_match_extra_tokens(temp_workspace):
    result = reward_func(
        prompt="List primary colors",
        completion="red blue green yellow",
        ground_truth=["red blue", "green"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 0.75


def test_exact_match_none_ground_truth(temp_workspace):
    result = reward_func(
        prompt="What is nothing?",
        completion="None",
        ground_truth=[],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 1.0


def test_exact_match_empty_completion(temp_workspace):
    result = reward_func(
        prompt="Say something",
        completion="",
        ground_truth=["apple"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 0.0


def test_exact_match_both_empty(temp_workspace):
    result = reward_func(
        prompt="Say nothing",
        completion="",
        ground_truth=[""],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 1.0


def test_exact_match_with_punctuation_and_case(temp_workspace):
    result = reward_func(
        prompt="List items",
        completion="Apple, Banana Cherry",
        ground_truth=["apple", "cherry"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result > 0.6


# ----------------------
# Fuzzy Match Tests
# ----------------------

def test_fuzzy_match_single_perfect(temp_workspace):
    result = reward_func(
        prompt="What fruit?",
        completion="apple",
        ground_truth=["apple"],
        workspace=temp_workspace,
        reward_metric="fuzzy_match"
    )
    assert result == 1.0


def test_fuzzy_match_single_partial(temp_workspace):
    result = reward_func(
        prompt="What is hello?",
        completion="hello world",
        ground_truth=["hello"],
        workspace=temp_workspace,
        reward_metric="fuzzy_match"
    )
    assert result > 0.5


# ----------------------
# Edge Case Tests
# ----------------------

def test_unknown_reward_metric(temp_workspace, capsys):
    result = reward_func(
        prompt="Test",
        completion="test",
        ground_truth=["test"],
        workspace=temp_workspace,
        reward_metric="unknown_metric"
    )
    assert result == 0.0
    captured = capsys.readouterr()
    assert "Unknown reward metric" in captured.out


def test_none_completion(temp_workspace):
    result = reward_func(
        prompt="Say something",
        completion="None",
        ground_truth=["hello"],
        workspace=temp_workspace,
        reward_metric="exact_match"
    )
    assert result == 0.0


# ----------------------
# Helper Function Tests
# ----------------------

class TestParseTextToTokens:

    def test_simple_space_separation(self):
        assert parse_text_to_tokens("hello world test") == {"hello", "world", "test"}

    def test_comma_separation(self):
        assert parse_text_to_tokens("apple,banana,cherry") == {"apple", "banana", "cherry"}

    def test_mixed_separators(self):
        assert parse_text_to_tokens("apple, banana cherry | orange\tgrape") == {
            "apple", "banana", "cherry", "orange", "grape"
        }

    def test_case_normalization(self):
        assert parse_text_to_tokens("HELLO World TeSt") == {"hello", "world", "test"}

    def test_quote_removal(self):
        assert parse_text_to_tokens('"hello world"') == {"hello", "world"}

    def test_empty_string(self):
        assert parse_text_to_tokens("") == set()

    def test_whitespace_only(self):
        assert parse_text_to_tokens("   \t\n  ") == set()
