import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientResponse

from benchmax.envs.wikipedia.wiki_env import WikipediaEnv


@pytest.fixture
def wiki_env() -> WikipediaEnv:
    """Create WikipediaEnv instance without API keys for unit tests."""
    return WikipediaEnv(wikipedia_api_keys=None)


class TestDatasetPreprocess:
    """Tests for dataset preprocessing."""

    @pytest.mark.asyncio
    async def test_dataset_preprocess_valid_example(
        self, wiki_env: WikipediaEnv
    ) -> None:
        """Test preprocessing a valid dataset example."""
        example = {"Question": "Who created Python?", "Answer": "Guido van Rossum"}

        result = wiki_env.dataset_preprocess(example)

        assert result["prompt"] == "Who created Python?"
        assert result["ground_truth"] == "Guido van Rossum"


class TestComputeReward:
    """Tests for reward computation."""

    @pytest.mark.asyncio
    async def test_compute_reward_exact_match(self, wiki_env: WikipediaEnv) -> None:
        """Test that exact match returns 1.0."""
        completion = "The answer is <answer>Paris</answer>"
        ground_truth = "Paris"

        rewards = await wiki_env.compute_reward(
            rollout_id="test", completion=completion, ground_truth=ground_truth
        )

        assert rewards["text_match"] == 1.0

    @pytest.mark.asyncio
    async def test_compute_reward_case_insensitive(
        self, wiki_env: WikipediaEnv
    ) -> None:
        """Test that matching is case-insensitive."""
        completion = "The answer is <answer>PARIS</answer>"
        ground_truth = "paris"

        rewards = await wiki_env.compute_reward(
            rollout_id="test", completion=completion, ground_truth=ground_truth
        )

        assert rewards["text_match"] == 1.0

    @pytest.mark.asyncio
    async def test_compute_reward_no_match(self, wiki_env: WikipediaEnv) -> None:
        """Test that wrong answer returns 0.0."""
        completion = "The answer is <answer>London</answer>"
        ground_truth = "Paris"

        rewards = await wiki_env.compute_reward(
            rollout_id="test", completion=completion, ground_truth=ground_truth
        )

        assert rewards["text_match"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_reward_missing_answer_tags(
        self, wiki_env: WikipediaEnv
    ) -> None:
        """Test that missing answer tags returns 0.0."""
        completion = "The answer is Paris"
        ground_truth = "Paris"

        rewards = await wiki_env.compute_reward(
            rollout_id="test", completion=completion, ground_truth=ground_truth
        )

        assert rewards["text_match"] == 0.0


class TestListTools:
    """Tests for listing available tools."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_two_tools(self, wiki_env: WikipediaEnv) -> None:
        """Test that exactly 2 tools are returned with correct names."""
        tools = await wiki_env.list_tools()

        assert len(tools) == 2

        tool_names = {tool.name for tool in tools}
        assert "search_wikipedia" in tool_names
        assert "get_wikipedia_article" in tool_names

        # Verify each tool has required schema properties
        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.input_schema is not None
            assert "type" in tool.input_schema
            assert "properties" in tool.input_schema


class TestRunTool:
    """Tests for tool execution with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_run_tool_search_wikipedia_success(
        self, mock_safe_request: AsyncMock, wiki_env: WikipediaEnv
    ) -> None:
        """Test successful Wikipedia search with mocked response."""
        mock_response = MagicMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "query": {
                    "search": [
                        {
                            "title": "Python (programming language)",
                            "snippet": "Python is a high-level programming language",
                        },
                        {
                            "title": "Python (genus)",
                            "snippet": "Python is a genus of constricting snakes",
                        },
                    ]
                }
            }
        )
        mock_safe_request.return_value = mock_response

        with patch("benchmax.envs.wikipedia.wiki_env.safe_request"):
            result = await wiki_env.run_tool(
                rollout_id="test", tool_name="search_wikipedia", q="Python", limit=5
            )

        assert isinstance(result, str)
        assert "Python (programming language)" in result
        assert "high-level programming language" in result

    @pytest.mark.asyncio
    @patch("benchmax.envs.wikipedia.wiki_env.safe_request")
    async def test_run_tool_get_article_success(
        self, mock_safe_request: AsyncMock, wiki_env: WikipediaEnv
    ) -> None:
        """Test successful article fetch with mocked response."""
        mock_response = MagicMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "query": {
                    "pages": {
                        "12345": {
                            "extract": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."
                        }
                    }
                }
            }
        )
        mock_safe_request.return_value = mock_response

        result = await wiki_env.run_tool(
            rollout_id="test",
            tool_name="get_wikipedia_article",
            title="Python (programming language)",
            max_chars=1000,
        )

        assert isinstance(result, str)
        assert "Python is a high-level" in result
        assert "code readability" in result

    @pytest.mark.asyncio
    async def test_run_tool_missing_required_param(
        self, wiki_env: WikipediaEnv
    ) -> None:
        """Test error handling when required parameter is missing."""
        # Search without query
        result = await wiki_env.run_tool(
            rollout_id="test",
            tool_name="search_wikipedia",
            q="",  # Empty query
            limit=5,
        )

        assert isinstance(result, str)
        assert "Error" in result


class TestInitRollout:
    """Tests for rollout initialization."""

    @pytest.mark.asyncio
    async def test_init_rollout_completes(self, wiki_env: WikipediaEnv) -> None:
        """Test that init_rollout completes without errors (no-op)."""
        # Should complete without raising exceptions
        await wiki_env.init_rollout(rollout_id="test_rollout_1")
        await wiki_env.init_rollout(rollout_id="test_rollout_2", extra_arg="value")

        # No assertions needed - just verify no exceptions raised
        assert True
