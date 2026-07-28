"""Tests for SearchEnv — multi-component reward search environment."""

from __future__ import annotations

import asyncio
import pickle
from unittest.mock import AsyncMock, patch

import cloudpickle
import pytest
from benchmax.envs import BaseRollout, StaticBearerAuth
from benchmax.rewards import Judge, RubricEvaluation
from castform.rag.corpus.search_client import SearchClient
from main import SearchEnv, _extract_answer_block

JUDGE_ARGS = {
    "judge_base_url": "http://judge.test/v1",
    "judge_model": "gpt-4o",
    "judge_auth": StaticBearerAuth("test-key"),
}


class StubSearch:
    """Minimal SearchClient for testing."""

    def __init__(self, modes=None, results=None):
        self._modes = modes or ["vector"]
        self._results = results if results is not None else ["result one", "result two"]

    async def search(self, query, mode="auto", top_k=10):
        return [
            {
                "content": r,
                "source": f"doc_{i}",
                "metadata": {"file": f"doc_{i}", "section": f"section_{i}"},
                "score": 10.0 - i,
            }
            for i, r in enumerate(self._results[:top_k])
        ]

    async def embed(self, text):
        return [0.1, 0.2, 0.3]

    @property
    def available_modes(self):
        return self._modes

    def get_params(self):
        return {"backend": "stub"}


def _make_env(**overrides):
    defaults = {"search": StubSearch(), **JUDGE_ARGS}
    defaults.update(overrides)
    return SearchEnv(**defaults)


class TestInit:
    def test_isinstance_search_client(self):
        assert isinstance(StubSearch(), SearchClient)

    def test_requires_judge_credentials(self):
        # judge_base_url and judge_model are required; auth is declared separately.
        with pytest.raises(ValueError, match="requires judge_base_url"):
            SearchEnv(search=StubSearch(), judge_base_url="", judge_model="m")

        with pytest.raises(ValueError, match="requires judge_base_url"):
            SearchEnv(search=StubSearch(), judge_base_url="u", judge_model="")

    def test_tool_schema_has_query(self):
        env = _make_env()
        parameters = env._tools["search"][0]["function"]["parameters"]
        assert "query" in parameters["properties"]
        assert "query" in parameters["required"]

    def test_no_mode_property_with_single_mode(self):
        env = _make_env(search=StubSearch(modes=["vector"]))
        parameters = env._tools["search"][0]["function"]["parameters"]
        assert "mode" not in parameters["properties"]

    def test_mode_property_with_multiple_modes(self):
        env = _make_env(search=StubSearch(modes=["lexical", "vector", "hybrid"]))
        parameters = env._tools["search"][0]["function"]["parameters"]
        assert "mode" in parameters["properties"]
        assert "hybrid" in parameters["properties"]["mode"]["enum"]

    def test_default_mode_hybrid_preferred(self):
        env = _make_env(search=StubSearch(modes=["lexical", "vector", "hybrid"]))
        assert env._default_mode == "hybrid"

    def test_default_mode_lexical_when_no_hybrid(self):
        env = _make_env(search=StubSearch(modes=["lexical", "vector"]))
        assert env._default_mode == "lexical"

    def test_default_system_prompt_is_empty(self):
        assert SearchEnv.system_prompt is None

    def test_render_system_prompt_includes_corpus_description(self):
        prompt = SearchEnv.render_system_prompt(
            corpus_description="Korean legal statutes", max_search_calls=4
        )
        assert "Korean legal statutes" in prompt

    def test_render_system_prompt_includes_max_search_calls(self):
        prompt = SearchEnv.render_system_prompt(
            corpus_description="docs", max_search_calls=4
        )
        assert "4 times" in prompt

    def test_default_max_search_calls_is_ten(self):
        env = _make_env()
        assert env._max_search_calls == 10

    def test_default_weights_are_the_audited_shape(self):
        env = _make_env()
        assert env._w_correctness == pytest.approx(1.0)
        assert env._w_retrieval_hit == pytest.approx(0.3)
        assert env._w_citation_precision == pytest.approx(0.3)
        assert env._w_length == pytest.approx(0.2)

    def test_removed_weight_kwargs_fail_loudly(self):
        # The pre-audit weights are GONE, not aliased. **kwargs would swallow
        # them silently (caller's weights become no-ops), so __init__ rejects
        # them explicitly.
        for kwarg in ("w_conciseness", "w_citation_recall", "w_search_efficiency"):
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                _make_env(**{kwarg: 0.5})

    def test_primary_reward_key_is_answer_correctness(self):
        assert SearchEnv.PRIMARY_REWARD_KEY == "answer_correctness"

    def test_subclass_can_set_plain_system_prompt(self):
        class CustomEnv(SearchEnv):
            system_prompt = "Override prompt"

        assert CustomEnv.system_prompt == "Override prompt"

    def test_subclass_sets_system_prompt_via_render_helper(self):
        class CustomEnv(SearchEnv):
            system_prompt = SearchEnv.render_system_prompt(
                corpus_description="Korean law", max_search_calls=7
            )

        assert "Korean law" in CustomEnv.system_prompt
        assert "7 times" in CustomEnv.system_prompt

    def test_render_uses_overridden_template(self):
        class CustomEnv(SearchEnv):
            SYSTEM_PROMPT_TEMPLATE = (
                "Search over {corpus_description} with {max_search_calls} budget."
            )

        assert (
            CustomEnv.render_system_prompt(
                corpus_description="Korean law", max_search_calls=7
            )
            == "Search over Korean law with 7 budget."
        )

    def test_render_preserves_json_like_literals(self):
        # RAG prompts frequently include JSON few-shot examples. The regex
        # substitution should leave them untouched instead of crashing.
        class CustomEnv(SearchEnv):
            SYSTEM_PROMPT_TEMPLATE = (
                'Example: {"answer": "X"} for {corpus_description}.'
            )

        assert (
            CustomEnv.render_system_prompt(
                corpus_description="legal docs", max_search_calls=5
            )
            == 'Example: {"answer": "X"} for legal docs.'
        )

    def test_render_preserves_unknown_placeholders(self):
        # An unknown {name} placeholder passes through verbatim rather than
        # raising KeyError, so users can author templates forward-compatibly.
        class CustomEnv(SearchEnv):
            SYSTEM_PROMPT_TEMPLATE = (
                "Use {corpus_description}. Future hook: {custom_var}."
            )

        assert (
            CustomEnv.render_system_prompt(
                corpus_description="legal docs", max_search_calls=5
            )
            == "Use legal docs. Future hook: {custom_var}."
        )


class TestSearchTool:
    def test_empty_query_returns_error(self):
        env = _make_env()
        result = asyncio.run(env._search_tool(query=""))
        assert result.startswith("Error")

    def test_returns_formatted_results(self):
        env = _make_env(search=StubSearch(results=["foo", "bar"]))
        result = asyncio.run(env._search_tool(query="test"))
        assert "foo" in result
        assert "bar" in result

    def test_no_results(self):
        env = _make_env(search=StubSearch(results=[]))
        result = asyncio.run(env._search_tool(query="test"))
        assert result == "No results found."

    def test_metadata_search_includes_source_labels(self):
        env = _make_env(search=StubSearch(results=["foo", "bar"]))
        result = asyncio.run(env._search_tool(query="test"))
        assert "[source: doc_0]" in result
        assert "Metadata:" in result

    def test_delegates_to_search_client(self):
        calls = []

        class TrackingSearch(StubSearch):
            async def search(self, query, mode="auto", top_k=10):
                calls.append({"query": query, "mode": mode, "top_k": top_k})
                return [
                    {"content": "result", "source": "", "metadata": {}, "score": 1.0}
                ]

        env = _make_env(search=TrackingSearch())
        asyncio.run(env._search_tool(query="test query", limit=5))
        assert len(calls) == 1
        assert calls[0]["query"] == "test query"
        assert calls[0]["top_k"] == 5


def _msgs(content):
    """Wrap a completion string in a messages list (assistant-only)."""
    if isinstance(content, list):
        return content
    return [{"role": "assistant", "content": content}]


async def _compute_reward(env, rollout_id, messages, example_args):
    return await type(env).compute_reward(
        env,
        BaseRollout(
            rollout_id=rollout_id,
            termination_reason="finished",
            messages=messages,
            example_args=example_args,
        ),
    )


class TestComputeReward:
    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_all_components_returned(self, mock_eval):
        mock_eval.return_value = RubricEvaluation(0.8, "", "")
        env = _make_env()
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("The answer is <answer>42 [Source: doc_a]</answer>"),
                {
                    "question": "What?",
                    "ground_truth": "42",
                    "reference_chunks": [
                        {"content": "...", "metadata": {"file": "doc_a"}}
                    ],
                },
            )
        )
        # Exactly the audited 4-component shape — no conciseness/recall/
        # search_efficiency keys in the default reward.
        assert set(result) == {
            "answer_correctness",
            "retrieval_hit",
            "citation_precision",
            "answer_length",
        }

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_correctness_score(self, mock_eval):
        mock_eval.return_value = RubricEvaluation(0.5, "", "")
        env = _make_env(w_correctness=2.0)
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("<answer>partial</answer>"),
                {"question": "Q?", "ground_truth": "full answer"},
            )
        )
        assert result["answer_correctness"] == pytest.approx(1.0)  # 0.5 * 2.0

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_answer_length_gated_on_correctness(self, mock_eval):
        # Correctness=0 → the brevity term is 0 (short-but-wrong earns nothing).
        mock_eval.return_value = RubricEvaluation(0.0, "", "")
        env = _make_env()
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("<answer>wrong</answer>"),
                {"question": "Q?", "ground_truth": "right"},
            )
        )
        assert result["answer_length"] == 0.0

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_answer_length_is_deterministic_brevity(self, mock_eval):
        # Correct + short → w_length * (1 - len/ANSWER_LENGTH_CAP); correct +
        # over the cap → clamps to 0. No second judge call is ever made.
        mock_eval.return_value = RubricEvaluation(1.0, "", "")
        env = _make_env(w_length=1.0)
        short = "42"
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs(f"<answer>{short}</answer>"),
                {"question": "Q?", "ground_truth": "42"},
            )
        )
        assert result["answer_length"] == pytest.approx(1.0 - len(short) / 600)
        assert mock_eval.await_count == 1  # ONE judge call (correctness only)

        long = "x" * 700  # >= ANSWER_LENGTH_CAP
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs(f"<answer>{long}</answer>"),
                {"question": "Q?", "ground_truth": "42"},
            )
        )
        assert result["answer_length"] == 0.0

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_no_answer_tag_short_circuits_before_judge(self, mock_eval):
        # Strict extraction: no committed <answer> → zeros, judge never called.
        env = _make_env()
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("I think it's 42 but I never commit an answer tag"),
                {"question": "Q?", "ground_truth": "42"},
            )
        )
        assert all(v == 0.0 for v in result.values())
        mock_eval.assert_not_awaited()

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_citation_exact_match(self, mock_eval):
        mock_eval.return_value = RubricEvaluation(1.0, "", "")
        env = _make_env(w_retrieval_hit=1.0, w_citation_precision=1.0)
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs(
                    "<answer>Found it [Source: statute_a] [Source: statute_b]</answer>"
                ),
                {
                    "question": "Q?",
                    "ground_truth": "answer",
                    "reference_chunks": [
                        {"content": "...", "metadata": {"file": "statute_a"}},
                        {"content": "...", "metadata": {"file": "statute_b"}},
                    ],
                },
            )
        )
        assert result["retrieval_hit"] == pytest.approx(1.0)
        assert result["citation_precision"] == pytest.approx(1.0)

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_citation_matches_title_path_variant_by_default(self, mock_eval):
        # The default canonicalizer is LOOSE (id-hash OR title-path): a cited
        # 'docs/Statute_A.md' matches a bare 'statute_a' gold file id.
        mock_eval.return_value = RubricEvaluation(1.0, "", "")
        env = _make_env(w_retrieval_hit=1.0, w_citation_precision=1.0)
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("<answer>Found it [Source: docs/Statute_A.md]</answer>"),
                {
                    "question": "Q?",
                    "ground_truth": "answer",
                    "reference_chunks": [
                        {"content": "...", "metadata": {"file": "statute_a"}},
                    ],
                },
            )
        )
        assert result["retrieval_hit"] == pytest.approx(1.0)
        assert result["citation_precision"] == pytest.approx(1.0)

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_citation_partial_recall(self, mock_eval):
        mock_eval.return_value = RubricEvaluation(1.0, "", "")
        env = _make_env(w_retrieval_hit=1.0, w_citation_precision=1.0)
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("<answer>Found it [Source: statute_a]</answer>"),
                {
                    "question": "Q?",
                    "ground_truth": "answer",
                    "reference_chunks": [
                        {"content": "...", "metadata": {"file": "statute_a"}},
                        {"content": "...", "metadata": {"file": "statute_b"}},
                    ],
                },
            )
        )
        assert result["retrieval_hit"] == pytest.approx(0.5)
        assert result["citation_precision"] == pytest.approx(1.0)

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_gated_rewards_scaled_by_partial_correctness(self, mock_eval):
        # correctness=0.5. The GATED components (precision, length) scale by
        # correctness; retrieval_hit does NOT (it's ungated by design).
        mock_eval.return_value = RubricEvaluation(0.5, "", "")
        env = _make_env(
            w_correctness=1.0,
            w_retrieval_hit=1.0,
            w_citation_precision=1.0,
            w_length=1.0,
        )
        answer = "partial [Source: doc_a]"
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs(f"<answer>{answer}</answer>"),
                {
                    "question": "Q?",
                    "ground_truth": "full",
                    "reference_chunks": [
                        {"content": "...", "metadata": {"file": "doc_a"}}
                    ],
                },
            )
        )
        assert result["answer_correctness"] == pytest.approx(0.5)  # not gated
        assert result["retrieval_hit"] == pytest.approx(1.0)  # UNGATED: full recall
        assert result["citation_precision"] == pytest.approx(0.5)  # 1.0 * 1.0 * 0.5
        assert result["answer_length"] == pytest.approx((1.0 - len(answer) / 600) * 0.5)

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_retrieval_hit_survives_wrong_answer(self, mock_eval):
        # The core audit fix: correctness=0 → the GATED precision is zeroed, but
        # the UNGATED retrieval_hit still credits citing the gold source.
        mock_eval.return_value = RubricEvaluation(0.0, "", "")
        env = _make_env(w_retrieval_hit=1.0, w_citation_precision=1.0)
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("<answer>wrong [Source: doc_a]</answer>"),
                {
                    "question": "Q?",
                    "ground_truth": "right",
                    "reference_chunks": [
                        {"content": "...", "metadata": {"file": "doc_a"}}
                    ],
                },
            )
        )
        assert result["answer_correctness"] == 0.0
        assert result["retrieval_hit"] == pytest.approx(1.0)  # UNGATED
        assert result["citation_precision"] == 0.0  # gated → 0

    @patch(
        "main.evaluate_single_rubric",
        new_callable=AsyncMock,
    )
    def test_judge_failure_is_infrastructure_failure(self, mock_eval):
        mock_eval.side_effect = RuntimeError("judge down")
        env = _make_env(w_retrieval_hit=1.0)
        with pytest.raises(RuntimeError, match="judge down"):
            asyncio.run(
                _compute_reward(
                    env,
                    "r1",
                    _msgs("<answer>42 [Source: doc_a]</answer>"),
                    {
                        "question": "Q?",
                        "ground_truth": "42",
                        "reference_chunks": [
                            {"content": "...", "metadata": {"file": "doc_a"}}
                        ],
                    },
                )
            )

    def test_empty_completion_returns_zeros(self):
        env = _make_env()
        result = asyncio.run(
            _compute_reward(
                env,
                "r1",
                _msgs("   "),
                {"ground_truth": "42"},
            )
        )
        assert all(v == 0.0 for v in result.values())


class TestCitationScoring:
    def test_no_reference_chunks_returns_zero(self):
        env = _make_env()
        recall, precision = env._score_citations("answer [Source: x]", [])
        assert recall == 0.0
        assert precision == 0.0

    def test_no_citations_returns_zero_precision(self):
        env = _make_env()
        recall, precision = env._score_citations(
            "answer without citations",
            [{"content": "...", "metadata": {"file": "doc_a"}}],
        )
        assert recall == 0.0
        assert precision == 0.0

    def test_canonicalize_id_strips_whitespace(self):
        env = _make_env()
        assert env._canonicalize_id("  doc_a  ") == "doc_a"

    def test_default_canonicalizer_is_loose(self):
        # SearchEnv now defaults to id-hash OR title-path matching.
        env = _make_env()
        assert env._canonicalize_id("docs/Geography.md") == "geography"
        assert env._canonicalize_id("Geography.md") == "geography"
        assert env._canonicalize_id("geography") == "geography"

    def test_score_citations_matches_title_path_to_bare_id(self):
        env = _make_env()
        recall, precision = env._score_citations(
            "answer [Source: docs/Geography.md]",
            [{"content": "...", "metadata": {"file": "geography"}}],
        )
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)

    def test_email_style_thread_ids_work_via_metadata_file(self):
        env = _make_env()
        recall, precision = env._score_citations(
            "answer [Source: thread_123]",
            [
                {
                    "content": "...",
                    "metadata": {"file": "thread_123", "thread_id": "thread_123"},
                }
            ],
        )
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)


class TestExtractAnswerBlock:
    def test_normal_closed_block(self):
        assert _extract_answer_block("reasoning <answer>42</answer>") == "42"

    def test_no_tag_returns_empty(self):
        # A completion with no <answer> opener scores as no answer — never its
        # full reasoning text.
        assert _extract_answer_block("just reasoning, never committed") == ""

    def test_last_block_wins_on_self_correction(self):
        # Multiple openers → the LAST committed block (a self-correction wins).
        text = "<answer>first guess</answer> on reflection <answer>final</answer>"
        assert _extract_answer_block(text) == "final"

    def test_forgives_missing_close_tag(self):
        # An unclosed final block → everything after the last opener.
        assert (
            _extract_answer_block("thinking... <answer>committed but unclosed")
            == "committed but unclosed"
        )

    def test_empty_text_returns_empty(self):
        assert _extract_answer_block("") == ""

    def test_stray_literal_answer_tag_does_not_hijack(self):
        # A real, closed answer followed by prose mentioning "<answer>" must
        # NOT be hijacked by the literal tag — otherwise the trailing fragment
        # is scored and a correct answer reads as wrong (correctness 0).
        text = (
            "<answer>The capital is Paris. [Source: doc_a]</answer> "
            "let me know if you want it outside <answer> tags"
        )
        assert _extract_answer_block(text) == "The capital is Paris. [Source: doc_a]"

    def test_closed_block_preferred_over_trailing_unclosed_opener(self):
        # A committed (closed) answer wins over a later unclosed opener.
        assert (
            _extract_answer_block("<answer>committed</answer> aside: <answer> draft")
            == "committed"
        )


class TestDatasetParsing:
    def test_extracts_question_answer(self):
        env = _make_env()
        result = env._example_from_row({"question": "What is X?", "answer": "Y"})
        user_msgs = [
            m for m in result.payload["prompt_messages"] if m["role"] == "user"
        ]
        assert user_msgs and user_msgs[0]["content"] == "What is X?"
        assert result.payload["question"] == "What is X?"
        assert result.payload["ground_truth"] == "Y"

    def test_passes_reference_chunks(self):
        env = _make_env()
        result = env._example_from_row(
            {"question": "Q", "answer": "A", "reference_chunks": [{"id": "c1"}]}
        )
        assert result.payload["reference_chunks"] == [{"id": "c1"}]

    def test_environment_prompt_is_part_of_prompt_messages(self):
        class CustomEnv(SearchEnv):
            system_prompt = SearchEnv.render_system_prompt(
                corpus_description="Korean legal statutes", max_search_calls=4
            )

        result = CustomEnv(search=StubSearch(), **JUDGE_ARGS)._example_from_row(
            {"question": "Q", "answer": "A"}
        )
        system_msgs = [
            m for m in result.payload["prompt_messages"] if m["role"] == "system"
        ]
        assert len(system_msgs) == 1
        assert "Korean legal statutes" in system_msgs[0]["content"]


class TestListTools:
    def test_returns_openai_tools(self):
        env = _make_env()
        tools = asyncio.run(env.list_tools())
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search"


class TestPickle:
    def test_class_pickle(self):
        data = cloudpickle.dumps(SearchEnv)
        restored = pickle.loads(data)
        assert restored.__name__ == "SearchEnv"

    def test_instance_pickle_roundtrip(self):
        search = StubSearch(modes=["lexical", "vector"])
        env = _make_env(search=search)
        data = cloudpickle.dumps(env)
        restored = pickle.loads(data)
        assert isinstance(restored, SearchEnv)
        assert restored._default_mode == "lexical"
        result = asyncio.run(restored._search_tool(query="test"))
        assert "result one" in result


# --- free reward helpers (imported by a scaffold main.py) --------------------

from main import (  # noqa: E402
    canonicalize_source_id,
    canonicalize_source_id_loose,
    extract_answer_block,
    extract_reference_ids,
    judge_answer_quality,
    parse_citations,
    score_citations,
    score_search_efficiency,
)


class TestFreeRewardHelpers:
    def test_extract_answer_block_public_name(self):
        # public name is the same strict extractor as the underscore alias
        assert extract_answer_block("<answer>x</answer>") == "x"
        assert extract_answer_block("no tag") == ""
        assert extract_answer_block is _extract_answer_block

    def test_score_citations_recall_precision(self):
        chunks = [
            {"metadata": {"file": "a"}},
            {"metadata": {"file": "b"}},
        ]
        recall, precision = score_citations("cite [Source: a]", chunks)
        assert recall == pytest.approx(0.5)  # 1 of 2 gold cited
        assert precision == pytest.approx(1.0)  # the 1 cite is valid

    def test_score_citations_no_gold_is_zero(self):
        assert score_citations("[Source: a]", []) == (0.0, 0.0)

    def test_score_citations_custom_canonicalize(self):
        # a corpus-robust matcher (case-insensitive) can be injected
        chunks = [{"metadata": {"file": "DocA"}}]
        recall, _ = score_citations(
            "[Source: doca]", chunks, canonicalize=lambda s: s.strip().lower()
        )
        assert recall == pytest.approx(1.0)

    def test_parse_and_reference_id_helpers(self):
        assert parse_citations("x [Source: p ] y") == {"p"}
        assert extract_reference_ids([{"metadata": {"file_path": " q "}}]) == {"q"}
        assert canonicalize_source_id("  z  ") == "z"

    def test_canonicalize_source_id_loose(self):
        # id-hash OR title-path: lowercase, strip dir prefix + extension.
        assert canonicalize_source_id_loose("docs/Geography.md") == "geography"
        assert canonicalize_source_id_loose("Geography.md") == "geography"
        assert canonicalize_source_id_loose("geography") == "geography"
        assert canonicalize_source_id_loose("  AbC123  ") == "abc123"
        assert canonicalize_source_id_loose("a/b/Notes.2024.txt") == "notes.2024"
        assert canonicalize_source_id_loose("") == ""

    def test_canonicalize_source_id_stays_exact(self):
        # The strict matcher is still exported and still exact-path.
        assert canonicalize_source_id("docs/Geography.md") == "docs/Geography.md"
        # ...and remains the default for the free citation helpers.
        assert score_citations(
            "[Source: docs/Geography.md]", [{"metadata": {"file": "geography"}}]
        ) == (0.0, 0.0)

    def test_score_search_efficiency_gates_and_decays(self):
        # incorrect → 0; within baseline → full weight; over budget → 0
        assert (
            score_search_efficiency(
                calls=1,
                correctness=0.0,
                reference_chunk_count=1,
                max_search_calls=5,
                weight=0.1,
            )
            == 0.0
        )
        assert score_search_efficiency(
            calls=2,
            correctness=1.0,
            reference_chunk_count=1,
            max_search_calls=5,
            weight=0.1,
        ) == pytest.approx(0.1)  # baseline = 1 + 2 = 3, 2 calls → no decay
        assert (
            score_search_efficiency(
                calls=99,
                correctness=1.0,
                reference_chunk_count=1,
                max_search_calls=5,
                weight=0.1,
            )
            == 0.0
        )  # over the hard budget

    def test_judge_answer_quality_free_helper(self):
        with patch(
            "main.evaluate_single_rubric",
            new_callable=AsyncMock,
        ) as mock_eval:
            mock_eval.return_value = RubricEvaluation(0.5, "", "")
            c, con = asyncio.run(
                judge_answer_quality(
                    question="Q",
                    ground_truth="G",
                    response="A",
                    judge=Judge(
                        model="m",
                        base_url="u",
                        auth=StaticBearerAuth("k"),
                    ),
                )
            )
            assert c == pytest.approx(0.5) and con == pytest.approx(0.5)

    def test_judge_answer_quality_empty_response_is_zero(self):
        c, con = asyncio.run(
            judge_answer_quality(
                question="Q",
                ground_truth="G",
                response="   ",
                judge=Judge(
                    model="m",
                    base_url="u",
                    auth=StaticBearerAuth("k"),
                ),
            )
        )
        assert (c, con) == (0.0, 0.0)
