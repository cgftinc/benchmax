"""Integration tests for SearchEnv using real API."""

import os
import pytest
from dotenv import load_dotenv

from benchmax.envs.search.search_env import SearchEnv
from .test_utils import (
    create_account,
    sign_in,
    create_api_key,
    delete_api_key,
    create_corpus,
    upload_chunks,
    delete_corpus,
    get_test_files,
)

load_dotenv()


@pytest.fixture(scope="module")
async def test_credentials():
    """Yield test credentials and corpus, with cleanup."""
    # Get credentials from environment
    base_url = os.getenv("SEARCH_API_BASE_URL", "http://localhost:3000")
    email = os.getenv("SEARCH_TEST_EMAIL")
    password = os.getenv("SEARCH_TEST_PASSWORD")

    if not email or not password:
        pytest.skip("SEARCH_TEST_EMAIL and SEARCH_TEST_PASSWORD must be set in .env")

    # Setup: create account and API key
    await create_account(base_url, email, password)
    session_cookie = await sign_in(base_url, email, password)
    api_key, key_id = await create_api_key(base_url, session_cookie, "Search Test Key")

    # Create corpus and upload test data
    corpus_id = await create_corpus(base_url, api_key, "Integration Test Corpus")

    # Upload all test files
    test_files = get_test_files()
    for file_data in test_files:
        await upload_chunks(
            base_url,
            api_key,
            corpus_id,
            file_data["filename"],
            file_data["chunks"]
        )

    # Yield credentials for tests
    yield {
        "base_url": base_url,
        "api_key": api_key,
        "corpus_id": corpus_id,
        "session_cookie": session_cookie,
        "key_id": key_id,
    }

    # Cleanup
    try:
        await delete_corpus(base_url, api_key, corpus_id)
    except Exception as e:
        print(f"Failed to cleanup corpus: {e}")

    try:
        await delete_api_key(base_url, session_cookie, key_id)
    except Exception as e:
        print(f"Failed to cleanup API key: {e}")


@pytest.fixture
def search_env(test_credentials):
    """Create SearchEnv instance with test credentials."""
    return SearchEnv(
        api_key=test_credentials["api_key"],
        corpus_id=test_credentials["corpus_id"],
        base_url=test_credentials["base_url"]
    )


class TestSearchEnvironmentSetup:
    """Tests for search environment initialization."""

    @pytest.mark.asyncio
    async def test_env_creation(self, search_env: SearchEnv):
        """Test that environment can be created successfully."""
        assert search_env is not None
        assert search_env._api_key is not None
        assert search_env._corpus_id is not None


class TestSearchTools:
    """Tests for search tool functionality."""

    @pytest.mark.asyncio
    async def test_list_tools(self, search_env: SearchEnv):
        """Test listing available tools."""
        tools = await search_env.list_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0].name == "search_corpus"

    @pytest.mark.asyncio
    async def test_run_invalid_tool_raises(self, search_env: SearchEnv):
        """Test that running invalid tool raises exception."""
        with pytest.raises(Exception):
            await search_env.run_tool(
                rollout_id="test", tool_name="nonexistent_tool"
            )

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_basic_search(self, search_env: SearchEnv):
        """Test basic search without filters."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="database",
            limit=10
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "database" in result.lower()
        assert "Total:" in result

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, search_env: SearchEnv):
        """Test search with metadata filtering."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="configuration",
            metadata={"type": "config"}
        )

        assert isinstance(result, str)
        assert "config" in result.lower()
        # When using metadata filter, scores are null (filtered)
        assert "(filtered)" in result or "score:" in result

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_search_with_filename_substring(self, search_env: SearchEnv):
        """Test search with filename substring match."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="configuration",
            filename="json"
        )

        assert isinstance(result, str)
        # Results should be from JSON files
        assert ".json" in result

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_search_with_filename_regex(self, search_env: SearchEnv):
        """Test search with filename regex pattern."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="documentation",
            filename=".*\\.md$"
        )

        assert isinstance(result, str)
        # Results should be from Markdown files
        assert ".md" in result or "README" in result

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_search_with_limit(self, search_env: SearchEnv):
        """Test search with result limit."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="configuration",
            limit=2
        )

        assert isinstance(result, str)
        # Should have at most 2 results
        result_count = result.count("Content:")
        assert result_count <= 2

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_search_no_results(self, search_env: SearchEnv):
        """Test search that returns no results."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="xyzabc123nonexistent",
            limit=10
        )

        assert isinstance(result, str)
        assert "No results found" in result

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_search_combined_filters(self, search_env: SearchEnv):
        """Test search with both metadata and filename filters."""
        result = await search_env.run_tool(
            rollout_id="test",
            tool_name="search_corpus",
            query="configuration",
            metadata={"type": "config"},
            filename=".*\\.json$"
        )

        assert isinstance(result, str)
        # Should only contain JSON files with type=config metadata
        if "No results found" not in result:
            assert ".json" in result


class TestSearchReward:
    """Tests for reward computation."""

    @pytest.mark.asyncio
    async def test_reward_correct_answer(self, search_env: SearchEnv, unique_rollout_id: str):
        """Test reward for correct answer."""
        completion = "<answer>PostgreSQL</answer>"
        reward = await search_env.compute_reward(
            unique_rollout_id, completion, ground_truth="PostgreSQL"
        )
        assert reward["text_match"] == 1.0

    @pytest.mark.asyncio
    async def test_reward_incorrect_answer(self, search_env: SearchEnv, unique_rollout_id: str):
        """Test reward for incorrect answer."""
        completion = "<answer>MySQL</answer>"
        reward = await search_env.compute_reward(
            unique_rollout_id, completion, ground_truth="PostgreSQL"
        )
        assert reward["text_match"] == 0.0

    @pytest.mark.asyncio
    async def test_reward_missing_answer_tag(self, search_env: SearchEnv, unique_rollout_id: str):
        """Test reward when answer tag is missing."""
        completion = "The database is PostgreSQL."
        reward = await search_env.compute_reward(
            unique_rollout_id, completion, ground_truth="PostgreSQL"
        )
        assert reward["text_match"] == 0.0

    @pytest.mark.asyncio
    async def test_reward_case_insensitive(self, search_env: SearchEnv, unique_rollout_id: str):
        """Test reward with case-insensitive matching."""
        completion = "<answer>postgresql</answer>"
        reward = await search_env.compute_reward(
            unique_rollout_id, completion, ground_truth="PostgreSQL"
        )
        assert reward["text_match"] == 1.0


class TestSearchEnvE2E:
    """End-to-end tests simulating full workflows."""

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_e2e_search_and_answer(self, search_env: SearchEnv):
        """Test end-to-end workflow: search, extract info, compute reward."""
        # Initialize rollout
        rollout_id = "e2e_test"
        await search_env.init_rollout(rollout_id)

        # Search for database information
        search_result = await search_env.run_tool(
            rollout_id=rollout_id,
            tool_name="search_corpus",
            query="PostgreSQL database",
            limit=3
        )

        assert isinstance(search_result, str)
        assert len(search_result) > 0

        # Simulate model extracting answer from search results
        # In this test, we know the answer should be PostgreSQL
        completion = "Based on the search results, <answer>PostgreSQL</answer>"
        ground_truth = "PostgreSQL"

        # Compute reward
        rewards = await search_env.compute_reward(
            rollout_id=rollout_id,
            completion=completion,
            ground_truth=ground_truth
        )

        assert rewards["text_match"] == 1.0

        # Release rollout
        await search_env.release_rollout(rollout_id)

    @pytest.mark.slow
    @pytest.mark.remote
    @pytest.mark.asyncio
    async def test_e2e_filtered_search(self, search_env: SearchEnv):
        """Test end-to-end with filtered search."""
        rollout_id = "e2e_filtered"
        await search_env.init_rollout(rollout_id)

        # Search with filters
        search_result = await search_env.run_tool(
            rollout_id=rollout_id,
            tool_name="search_corpus",
            query="configuration",
            metadata={"type": "config"},
            filename="json",
            limit=5
        )

        assert isinstance(search_result, str)

        # Verify result contains expected content
        if "No results found" not in search_result:
            assert "config" in search_result.lower() or "configuration" in search_result.lower()

        await search_env.release_rollout(rollout_id)


class TestSearchDatasetPreprocess:
    """Tests for dataset preprocessing."""

    def test_dataset_preprocess(self, search_env: SearchEnv):
        """Test preprocessing a dataset example."""
        example = {
            "Question": "What is the default PostgreSQL port?",
            "Answer": "5432"
        }

        result = search_env.dataset_preprocess(example)

        assert result["prompt"] == "What is the default PostgreSQL port?"
        assert result["ground_truth"] == "5432"
        assert result["init_rollout_args"] == {}
