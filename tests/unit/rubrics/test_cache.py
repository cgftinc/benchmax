import json
import os

import pytest

from benchmax.rubrics.cache import (
    _empty_cache_entry,
    _format_cached_rubrics_for_prompt,
    atomic_cache_update,
    filter_and_cache_rubrics,
    get_cache_for_question,
    load_rubric_cache,
    save_rubric_cache,
)


def test_load_returns_empty_when_missing(tmp_cache_file):
    assert not os.path.exists(tmp_cache_file)
    assert load_rubric_cache() == {}


def test_save_and_load_roundtrip(tmp_cache_file):
    cache = {"q1": _empty_cache_entry()}
    cache["q1"]["positive_rubrics"].append({"title": "T", "description": "D"})
    save_rubric_cache(cache)
    assert load_rubric_cache() == cache
    # readable as plain JSON too
    with open(tmp_cache_file) as f:
        assert json.load(f) == cache


def test_get_cache_for_question_creates_entry(tmp_cache_file):
    cache = get_cache_for_question("qX")
    assert cache["qX"] == _empty_cache_entry()


def test_atomic_cache_update_applies_fn(tmp_cache_file):
    def add(c):
        c["q1"] = {"marker": 1}

    atomic_cache_update(add)
    assert load_rubric_cache() == {"q1": {"marker": 1}}


def test_atomic_cache_update_retries_then_succeeds(tmp_cache_file, monkeypatch):
    monkeypatch.setattr("benchmax.rubrics.cache.time.sleep", lambda _: None)
    calls = {"n": 0}

    def flaky(c):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        c["q1"] = {"ok": True}

    atomic_cache_update(flaky, max_retries=5)
    assert calls["n"] == 3
    assert load_rubric_cache() == {"q1": {"ok": True}}


def test_atomic_cache_update_raises_after_max_retries(tmp_cache_file, monkeypatch):
    monkeypatch.setattr("benchmax.rubrics.cache.time.sleep", lambda _: None)

    def always_fails(_):
        raise RuntimeError("persistent")

    with pytest.raises(RuntimeError, match="persistent"):
        atomic_cache_update(always_fails, max_retries=2)


def test_format_cached_rubrics_for_prompt_empty():
    assert _format_cached_rubrics_for_prompt(_empty_cache_entry()) is None


def test_format_cached_rubrics_for_prompt_nonempty():
    cache = _empty_cache_entry()
    cache["positive_rubrics"].append({"title": "P", "description": "pd"})
    cache["negative_rubrics"].append({"title": "N", "description": "nd"})
    out = _format_cached_rubrics_for_prompt(cache)
    assert "Positive Rubrics:" in out
    assert "- P: pd" in out
    assert "- N: nd" in out


def test_filter_and_cache_skips_zero_variance(tmp_cache_file):
    new = {"positive_rubrics": [{"title": "Flat", "description": "d"}]}
    filter_and_cache_rubrics("q1", new, "positive_rubrics", scores=[1.0, 1.0, 1.0])
    cache = load_rubric_cache()["q1"]
    assert cache["positive_rubrics"] == []


def test_filter_and_cache_keeps_top_3_by_std(tmp_cache_file):
    # Insert 4 rubrics with increasing variance; only top 3 (by std) survive.
    for i, scores in enumerate([[0, 0, 1], [0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1]]):
        title = f"R{i}"
        filter_and_cache_rubrics(
            "q1",
            {"positive_rubrics": [{"title": title, "description": "x"}]},
            "positive_rubrics",
            scores=scores,
        )
    kept = load_rubric_cache()["q1"]["positive_rubrics"]
    assert len(kept) == 3
    stds = [r["std"] for r in kept]
    assert stds == sorted(stds, reverse=True)


def test_filter_and_cache_updates_existing_rubric_in_place(tmp_cache_file):
    new = {"positive_rubrics": [{"title": "Same", "description": "v1"}]}
    filter_and_cache_rubrics("q1", new, "positive_rubrics", scores=[0, 1])
    first_std = load_rubric_cache()["q1"]["positive_rubrics"][0]["std"]

    new["positive_rubrics"][0]["description"] = "v2"
    filter_and_cache_rubrics("q1", new, "positive_rubrics", scores=[0, 0, 0, 1])
    kept = load_rubric_cache()["q1"]["positive_rubrics"]
    assert len(kept) == 1
    assert kept[0]["description"] == "v2"
    assert kept[0]["std"] != first_std
