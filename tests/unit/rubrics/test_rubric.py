import pytest

from benchmax.rubrics.rubric import (
    Rubric,
    _cache_dict_to_rubric,
    evaluate_rubric_ranking,
    evaluate_single_rubric,
)


def _r(title="T", description="D", type_="positive", score_map=None):
    return Rubric(title=title, description=description, type=type_, score_map=score_map)


def test_cache_dict_to_rubric_constructs():
    r = _cache_dict_to_rubric({"title": "T", "description": "D"}, "negative")
    assert r.title == "T" and r.description == "D" and r.type == "negative"


# ---------------------------------------------------------------------------
# evaluate_single_rubric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_single_rubric_binary_prompt(stub_openai):
    factory = stub_openai(['{"score": 1, "reasoning": "good"}'])
    result = await evaluate_single_rubric(
        rubric=_r(), question="q", response="resp", model_name="m", base_url="u"
    )
    assert result == {
        "score": 1,
        "reasoning": "good",
        "llm_output": '{"score": 1, "reasoning": "good"}',
    }
    prompt = factory.calls[0]["messages"][0]["content"]
    assert "Score 1" in prompt and "Score 0" in prompt  # binary template


@pytest.mark.asyncio
async def test_evaluate_single_rubric_ranged_prompt_uses_score_map(stub_openai):
    factory = stub_openai(['{"score": 3, "reasoning": "ok"}'])
    rubric = _r(score_map={1.0: "bad", 3.0: "mid", 5.0: "best"})
    result = await evaluate_single_rubric(
        rubric=rubric, question="q", response="r", model_name="m", base_url="u"
    )
    assert result["score"] == 3
    prompt = factory.calls[0]["messages"][0]["content"]
    assert "1.0, 3.0, 5.0" in prompt
    assert "- 3.0: mid" in prompt


@pytest.mark.asyncio
async def test_evaluate_single_rubric_includes_ground_truth(stub_openai):
    factory = stub_openai(['{"score": 0}'])
    await evaluate_single_rubric(
        rubric=_r(),
        question="q",
        response="r",
        model_name="m",
        base_url="u",
        ground_truth="THE_GT",
    )
    assert "THE_GT" in factory.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_evaluate_single_rubric_empty_response_returns_zero(stub_openai):
    stub_openai([""])  # empty judge output
    result = await evaluate_single_rubric(
        rubric=_r(), question="q", response="r", model_name="m", base_url="u"
    )
    assert result == {"score": 0, "reasoning": "Empty response", "llm_output": ""}


@pytest.mark.asyncio
async def test_evaluate_single_rubric_exception_returns_zero(stub_openai):
    def raiser(_):
        raise RuntimeError("network down")

    stub_openai(raiser)
    result = await evaluate_single_rubric(
        rubric=_r(), question="q", response="r", model_name="m", base_url="u"
    )
    assert result["score"] == 0
    assert "network down" in result["reasoning"]


# ---------------------------------------------------------------------------
# evaluate_rubric_ranking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranking_all_empty_responses(stub_openai):
    stub_openai([])  # judge should not be called
    out = await evaluate_rubric_ranking(
        rubric=_r(), question="q", responses=["", "", ""], model_name="m", base_url="u"
    )
    assert out["scores"] == [0.0, 0.0, 0.0]
    assert out["ranking"] == []


@pytest.mark.asyncio
async def test_ranking_single_nonempty_short_circuits(stub_openai):
    stub_openai([])  # no judge call needed
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["", "only", ""],
        model_name="m",
        base_url="u",
    )
    assert out["scores"] == [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_ranking_no_gt_uses_position_formula(stub_openai):
    # Tier [[0,2],[1]] over 3 responses -> positions: 0,2 -> mid 0.5; 1 -> 2
    stub_openai(['{"ranking": [[0, 2], [1]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a", "b", "c"],
        model_name="m",
        base_url="u",
    )
    assert out["scores"][0] == pytest.approx(1.0 - 0.5 / 2)
    assert out["scores"][2] == pytest.approx(1.0 - 0.5 / 2)
    assert out["scores"][1] == pytest.approx(1.0 - 2 / 2)  # = 0.0


@pytest.mark.asyncio
async def test_ranking_with_gt_anchors_scores(stub_openai):
    # 2 responses + GT appended as index 2 (`gt_local`); judge places GT in middle.
    # ranking [[0], [2], [1]] -> positions: 0=>0, 2=>1, 1=>2; max_pos=2; gt_pos=1
    # Response 0 (above GT): 0.5 + 0.5 * (1 - 0) / 1 = 1.0
    # Response 1 (below GT): 0.3 * (1 - (2 - 1) / (2 - 1)) = 0.0
    stub_openai(['{"ranking": [[0], [2], [1]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a", "b"],
        model_name="m",
        base_url="u",
        ground_truth="GT",
    )
    assert out["scores"][0] == pytest.approx(1.0)
    assert out["scores"][1] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ranking_below_gt_ceiling_widens_band(stub_openai):
    # 3 responses + GT appended as index 3; judge ranks [[0],[3],[1],[2]] ->
    # positions 0=>0, 3=>1, 1=>2, 2=>3; max_pos=3; gt_pos=1.
    # Response 1 (below GT): C * (1 - (2-1)/(3-1)) = 0.5 * C
    # Response 2 (below GT, worst): C * (1 - (3-1)/(3-1)) = 0.0
    # Default ceiling 0.3 -> 0.15; raising to 0.5 -> 0.25 (more room below the bar).
    for ceiling, expected in ((0.3, 0.15), (0.5, 0.25)):
        stub_openai(['{"ranking": [[0], [3], [1], [2]]}'])
        out = await evaluate_rubric_ranking(
            rubric=_r(),
            question="q",
            responses=["a", "b", "c"],
            model_name="m",
            base_url="u",
            ground_truth="GT",
            below_gt_ceiling=ceiling,
        )
        assert out["scores"][0] == pytest.approx(1.0)  # above GT, unaffected
        assert out["scores"][1] == pytest.approx(expected)
        assert out["scores"][2] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ranking_multi_anchor_bands(stub_openai):
    # 3 responses + 2 anchors appended at indices 3 (acceptable, edge 0.1) and
    # 4 (great, edge 0.5). Judge ranks [[2],[4],[0],[3],[1]] ->
    # positions: 2=>0, 4=>1(great), 0=>2, 3=>3(acceptable), 1=>4; max_pos=4.
    # seams sorted by pos: [(1, 0.5), (3, 0.1)].
    #   resp 2 (p=0, above great):  0.5 + 0.5*(1-0)/1            = 1.0
    #   resp 0 (p=2, between):      0.1 + (0.5-0.1)*(3-2)/(3-1)  = 0.3
    #   resp 1 (p=4, below floor):  0.1*(4-4)/(4-3)              = 0.0
    stub_openai(['{"ranking": [[2], [4], [0], [3], [1]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a", "b", "c"],
        model_name="m",
        base_url="u",
        anchors=["ACCEPTABLE", "GREAT"],
        band_edges=[0.1, 0.5],
    )
    assert out["scores"][2] == pytest.approx(1.0)
    assert out["scores"][0] == pytest.approx(0.3)
    assert out["scores"][1] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ranking_multi_anchor_inverted_collapses(stub_openai):
    # Judge ranks the GREAT anchor (idx 4, edge 0.5) BELOW the ACCEPTABLE anchor
    # (idx 3, edge 0.1) — an inversion. The seams must collapse to one bar at the
    # better position so scoring stays monotonic (a better-ranked response scores
    # higher). ranking [[3],[0],[1],[4],[2]] -> positions 3=>0,0=>1,1=>2,4=>3,2=>4.
    # Collapsed seam: (pos 0, edge 0.5); max_pos=4. resp0 p=1 -> 0.5*(4-1)/4=0.375,
    # resp1 p=2 -> 0.5*(4-2)/4=0.25. Without the fix resp1 would outscore resp0.
    stub_openai(['{"ranking": [[3], [0], [1], [4], [2]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a", "b", "c"],
        model_name="m",
        base_url="u",
        anchors=["ACCEPTABLE", "GREAT"],
        band_edges=[0.1, 0.5],
    )
    assert out["scores"][0] > out["scores"][1]  # better rank -> higher score
    assert out["scores"][0] == pytest.approx(0.375)
    assert out["scores"][1] == pytest.approx(0.25)
    assert out["scores"][2] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ranking_anchors_supersede_ground_truth(stub_openai):
    # When both are passed, anchors win and ground_truth is ignored: only the
    # two anchors are appended (indices 1, 2), not a third gt slot.
    factory = stub_openai(['{"ranking": [[0], [2], [1]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a"],
        model_name="m",
        base_url="u",
        ground_truth="IGNORED",
        anchors=["ACC", "GREAT"],
        band_edges=[0.1, 0.5],
    )
    prompt = factory.calls[0]["messages"][0]["content"]
    assert "IGNORED" not in prompt and "ACC" in prompt and "GREAT" in prompt
    # resp 0 ranked best (above great) -> > 0.5
    assert out["scores"][0] > 0.5


@pytest.mark.asyncio
async def test_ranking_anchors_require_band_edges():
    with pytest.raises(ValueError):
        await evaluate_rubric_ranking(
            rubric=_r(),
            question="q",
            responses=["a"],
            model_name="m",
            base_url="u",
            anchors=["X"],
        )


@pytest.mark.asyncio
async def test_ranking_tied_with_gt_scores_half(stub_openai):
    # ranking [[0, 1, 2]] -> all tied at mid=1; gt_pos==response_pos -> 0.5
    stub_openai(['{"ranking": [[0, 1, 2]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a", "b"],
        model_name="m",
        base_url="u",
        ground_truth="GT",
    )
    assert out["scores"] == [pytest.approx(0.5), pytest.approx(0.5)]


@pytest.mark.asyncio
async def test_ranking_skips_empty_responses_in_judge_input(stub_openai):
    factory = stub_openai(['{"ranking": [[0], [1]]}'])
    await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["A", "", "B"],
        model_name="m",
        base_url="u",
    )
    prompt = factory.calls[0]["messages"][0]["content"]
    # judge sees 2 responses, indices 0 and 1
    assert "--- Response 0 ---\nA" in prompt
    assert "--- Response 1 ---\nB" in prompt
    assert "Response 2" not in prompt


@pytest.mark.asyncio
async def test_ranking_empty_judge_content_returns_empty_ranking(stub_openai):
    stub_openai([""])
    out = await evaluate_rubric_ranking(
        rubric=_r(), question="q", responses=["a", "b"], model_name="m", base_url="u"
    )
    assert out["scores"] == [0.0, 0.0]
    assert out["ranking"] == []
    assert out["reasoning"] == "Empty judge response"


@pytest.mark.asyncio
async def test_ranking_with_gt_missing_from_ranking_falls_back(stub_openai):
    # Judge omits the GT index (2) from its ranking; fallback ignores GT and
    # uses the no-GT positional formula for the remaining responses.
    # ranking [[0],[1]] over items=[a,b,GT] -> positions: 0=>0, 1=>1; max_pos=2
    stub_openai(['{"ranking": [[0], [1]]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(),
        question="q",
        responses=["a", "b"],
        model_name="m",
        base_url="u",
        ground_truth="GT",
    )
    assert out["scores"][0] == pytest.approx(1.0 - 0 / 2)
    assert out["scores"][1] == pytest.approx(1.0 - 1 / 2)


@pytest.mark.asyncio
async def test_ranking_accepts_non_list_tier_entries(stub_openai):
    # Judge returns bare ints instead of single-element lists; code should
    # promote them to tiers of size 1.
    stub_openai(['{"ranking": [0, 1]}'])
    out = await evaluate_rubric_ranking(
        rubric=_r(), question="q", responses=["a", "b"], model_name="m", base_url="u"
    )
    assert out["scores"][0] == pytest.approx(1.0)
    assert out["scores"][1] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ranking_exception_returns_zero_scores(stub_openai):
    def raiser(_):
        raise RuntimeError("x")

    stub_openai(raiser)
    out = await evaluate_rubric_ranking(
        rubric=_r(), question="q", responses=["a", "b"], model_name="m", base_url="u"
    )
    assert out["scores"] == [0.0, 0.0]
    assert "Error:" in out["reasoning"]
