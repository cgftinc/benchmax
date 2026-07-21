import pytest

from benchmax.rewards import Rubric, RubricCache


def _rubric(title: str, polarity: str = "positive") -> Rubric:
    return Rubric(title, "description", polarity=polarity)


def test_cache_is_caller_owned_and_prompt_isolation_is_stable():
    first, second = RubricCache(), RubricCache()
    first.consider("q", _rubric("varies"), [0, 1])
    assert [rubric.title for rubric in first.get("q").positive] == ["varies"]
    assert second.get("q").all == ()


def test_cache_rejects_non_discriminative_rubrics():
    cache = RubricCache()
    assert not cache.consider("q", _rubric("flat"), [1, 1, 1])
    assert cache.get("q").all == ()
    with pytest.raises(ValueError, match="finite"):
        cache.consider("q", _rubric("invalid"), [0, float("nan")])


def test_cache_keeps_highest_variance_per_polarity():
    cache = RubricCache(max_per_polarity=2)
    cache.consider("q", _rubric("low"), [0.4, 0.5])
    cache.consider("q", _rubric("high"), [0, 1])
    cache.consider("q", _rubric("medium"), [0.2, 0.8])
    cache.consider("q", _rubric("negative", "negative"), [0, 1])
    assert [rubric.title for rubric in cache.get("q").positive] == ["high", "medium"]
    assert [rubric.title for rubric in cache.get("q").negative] == ["negative"]


def test_get_does_not_expose_mutable_cache_internals():
    cache = RubricCache()
    cache.consider("q", _rubric("good"), [0, 1])
    selected = cache.get("q")
    cache.consider("q", _rubric("better"), [0, 0.5, 1])
    assert [rubric.title for rubric in selected.positive] == ["good"]
