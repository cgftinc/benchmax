import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from benchmax.envs.search.search_env import SearchEnv


@pytest.fixture
def search_env() -> SearchEnv:
    """Create SearchEnv instance with test credentials for unit tests."""
    return SearchEnv(
        api_key="test-api-key",
        corpus_id="test-corpus-id",
        base_url="http://localhost:3000"
    )


class TestInitialization:
    """Tests for SearchEnv initialization."""

    def test_init_with_custom_base_url(self) -> None:
        """Test initialization with custom base URL."""
        env = SearchEnv(
            api_key="my-key",
            corpus_id="my-corpus",
            base_url="https://api.example.com"
        )
        assert env._base_url == "https://api.example.com"

    def test_init_without_api_key_raises(self) -> None:
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            SearchEnv(api_key="", corpus_id="test-corpus", base_url="http://localhost:3000")

    def test_init_without_corpus_id_raises(self) -> None:
        """Test that missing corpus ID raises ValueError."""
        with pytest.raises(ValueError, match="corpus_id is required"):
            SearchEnv(api_key="test-key", corpus_id="", base_url="http://localhost:3000")


class TestDatasetPreprocess:
    """Tests for dataset preprocessing."""

    @pytest.mark.asyncio
    async def test_dataset_preprocess_valid_example(
        self, search_env: SearchEnv
    ) -> None:
        """Test preprocessing a valid dataset example."""
        example = {"Question": "What is PostgreSQL?", "Answer": "A database"}

        result = search_env.dataset_preprocess(example)

        assert result["prompt"] == "What is PostgreSQL?"
        assert result["ground_truth"] == "A database"
        assert result["init_rollout_args"] == {}


class TestComputeReward:
    """Tests for reward computation."""

    @pytest.mark.asyncio
    async def test_compute_reward_with_reference_chunks(self, search_env: SearchEnv) -> None:
        """Test that overlap with reference chunks returns score >= 0.25."""
        completion = "PostgreSQL is a powerful database system used for storing data."
        ground_truth = "PostgreSQL"
        reference_chunks = ["PostgreSQL is a powerful database", "system used for storing"]

        rewards = await search_env.compute_reward(
            rollout_id="test",
            completion=completion,
            ground_truth=ground_truth,
            reference_chunks=reference_chunks
        )

        assert "chunk_overlap" in rewards
        assert rewards["chunk_overlap"] >= 0.25

    @pytest.mark.asyncio
    async def test_compute_reward_high_overlap(
        self, search_env: SearchEnv
    ) -> None:
        """Test that high overlap returns higher score."""
        completion = "PostgreSQL is a powerful open-source database system"
        ground_truth = "PostgreSQL"
        reference_chunks = ["PostgreSQL is a powerful open-source database system"]

        rewards = await search_env.compute_reward(
            rollout_id="test",
            completion=completion,
            ground_truth=ground_truth,
            reference_chunks=reference_chunks
        )

        assert rewards["chunk_overlap"] >= 0.9

    @pytest.mark.asyncio
    async def test_compute_reward_low_overlap_below_threshold(self, search_env: SearchEnv) -> None:
        """Test that low overlap below 0.25 threshold returns 0.0."""
        completion = "ABCDEFGHIJKLMNOP"
        ground_truth = "PostgreSQL"
        reference_chunks = ["QRSTUVWXYZ"]

        rewards = await search_env.compute_reward(
            rollout_id="test",
            completion=completion,
            ground_truth=ground_truth,
            reference_chunks=reference_chunks
        )

        assert rewards["chunk_overlap"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_reward_no_reference_chunks(
        self, search_env: SearchEnv
    ) -> None:
        """Test that missing reference chunks returns 0.0."""
        completion = "The answer is PostgreSQL"
        ground_truth = "PostgreSQL"

        rewards = await search_env.compute_reward(
            rollout_id="test", completion=completion, ground_truth=ground_truth
        )

        assert rewards["chunk_overlap"] == 0.0


class TestListTools:
    """Tests for listing available tools."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_one_tool(self, search_env: SearchEnv) -> None:
        """Test that exactly 1 tool is returned with correct name."""
        tools = await search_env.list_tools()

        assert len(tools) == 1

        tool = tools[0]
        assert tool.name == "search_corpus"

        # Verify tool has required schema properties
        assert tool.description is not None
        assert tool.input_schema is not None
        assert "type" in tool.input_schema
        assert "properties" in tool.input_schema
        assert "query" in tool.input_schema["properties"]
        assert "metadata" in tool.input_schema["properties"]
        assert "filename" in tool.input_schema["properties"]
        assert "limit" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["query"]


class TestRunTool:
    """Tests for tool execution with mocked HTTP calls."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.search.search_env.aiohttp.ClientSession")
    async def test_run_tool_search_corpus_success(
        self, mock_session_cls: MagicMock, search_env: SearchEnv
    ) -> None:
        """Test successful corpus search with mocked response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {
                        "id": "chunk-1",
                        "filename": "config.json",
                        "content": "Database configuration: host=localhost, port=5432",
                        "metadata": {"type": "config", "page": 1},
                        "score": 0.85
                    },
                    {
                        "id": "chunk-2",
                        "filename": "README.md",
                        "content": "Configuration instructions: Set up your database connection.",
                        "metadata": {"type": "documentation"},
                        "score": 0.72
                    }
                ],
                "total": 2
            }
        )

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="database configuration",
            limit=5
        )

        assert isinstance(result, str)
        assert "config.json" in result
        assert "Database configuration" in result
        assert "0.85" in result
        assert "Total: 2" in result

    @pytest.mark.asyncio
    @patch("benchmax.envs.search.search_env.aiohttp.ClientSession")
    async def test_run_tool_search_with_metadata_filter(
        self, mock_session_cls: MagicMock, search_env: SearchEnv
    ) -> None:
        """Test search with metadata filter."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {
                        "id": "chunk-1",
                        "filename": "config.json",
                        "content": "Database configuration",
                        "metadata": {"type": "config"},
                        "score": None  # null when using metadata filter
                    }
                ],
                "total": 1
            }
        )

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="configuration",
            metadata={"type": "config"}
        )

        assert isinstance(result, str)
        assert "config.json" in result
        assert "(filtered)" in result  # null score shows as filtered
        assert "Total: 1" in result

    @pytest.mark.asyncio
    @patch("benchmax.envs.search.search_env.aiohttp.ClientSession")
    async def test_run_tool_search_with_filename_filter(
        self, mock_session_cls: MagicMock, search_env: SearchEnv
    ) -> None:
        """Test search with filename filter."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {
                        "id": "chunk-1",
                        "filename": "app.config.ts",
                        "content": "TypeScript configuration",
                        "metadata": {},
                        "score": 0.90
                    }
                ],
                "total": 1
            }
        )

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="configuration",
            filename=".*\\.config\\.ts$"
        )

        assert isinstance(result, str)
        assert "app.config.ts" in result
        assert "TypeScript configuration" in result

    @pytest.mark.asyncio
    @patch("benchmax.envs.search.search_env.aiohttp.ClientSession")
    async def test_run_tool_no_results(
        self, mock_session_cls: MagicMock, search_env: SearchEnv
    ) -> None:
        """Test search with no results."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [],
                "total": 0
            }
        )

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="nonexistent"
        )

        assert isinstance(result, str)
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_run_tool_missing_required_param(
        self, search_env: SearchEnv
    ) -> None:
        """Test error handling when required parameter is missing."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query=""  # Empty query
        )

        assert isinstance(result, str)
        assert "Error" in result
        assert "query" in result.lower()

    @pytest.mark.asyncio
    @patch("benchmax.envs.search.search_env.aiohttp.ClientSession")
    async def test_run_tool_api_error(
        self, mock_session_cls: MagicMock, search_env: SearchEnv
    ) -> None:
        """Test error handling for API errors."""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Corpus not found")

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="test"
        )

        assert isinstance(result, str)
        assert "Error" in result
        assert "404" in result


class TestInitRollout:
    """Tests for rollout initialization."""

    @pytest.mark.asyncio
    async def test_init_rollout_completes(self, search_env: SearchEnv) -> None:
        """Test that init_rollout completes without errors (no-op)."""
        await search_env.init_rollout(rollout_id="test_rollout_1")
        await search_env.init_rollout(rollout_id="test_rollout_2", extra_arg="value")

        # No assertions needed - just verify no exceptions raised
        assert True

    @pytest.mark.asyncio
    async def test_release_rollout_completes(self, search_env: SearchEnv) -> None:
        """Test that release_rollout completes without errors (no-op)."""
        await search_env.release_rollout(rollout_id="test_rollout_1")

        # No assertions needed - just verify no exceptions raised
        assert True


class TestShutdown:
    """Tests for environment shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_completes(self, search_env: SearchEnv) -> None:
        """Test that shutdown completes without errors (no-op)."""
        await search_env.shutdown()

        # No assertions needed - just verify no exceptions raised
        assert True
