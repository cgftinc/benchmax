"""The RAG seed's audited reward + retrieval gold-hit@k probe (`rag_main.py`).

Reward is exercised on fixtures with the judge monkeypatched; the probe on a fake
CorpusClient (no network, no corpus creation, no stdin)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import benchmax.cli.scaffold as scaffold_pkg
from benchmax.cli._project import _load_module_from_file, discover_env_class

_RAG_MAIN = Path(scaffold_pkg.__file__).parent / "rag_main.py"


@pytest.fixture
def rag_mod():
    return _load_module_from_file(_RAG_MAIN)


def _bare_env(mod):
    """A CustomSearchEnv instance without the network __init__, with just the attrs
    compute_reward reads (its own _canonicalize_id override stays live)."""
    cls = discover_env_class(mod)
    env = cls.__new__(cls)
    env._judge_model = "m"
    env._judge_base_url = "u"
    env._judge_timeout = 30.0
    env._judge_token_provider = lambda: "k"
    return env


def _judge(score):
    async def _fake(**kw):
        return {"score": score}

    return _fake


def _reward(
    env, mod, monkeypatch, answer_html, *, judge_score, gold_file="geography.md"
):
    monkeypatch.setattr(mod, "evaluate_single_rubric", _judge(judge_score))
    msgs = [{"role": "assistant", "content": answer_html}]
    task = {
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
        "reference_chunks": [{"metadata": {"file": gold_file}}],
    }
    return asyncio.run(
        env.compute_reward("r", msgs, task, termination_reason="finished")
    )


# ── reward: gating + the ungated retrieval_hit ──────────────────────────────────


def test_reward_wrong_answer_still_credits_retrieval(rag_mod, monkeypatch):
    """The core audit fix: a WRONG answer that cited the gold source earns
    retrieval_hit (UNGATED), but no gated secondary and no correctness."""
    env = _bare_env(rag_mod)
    r = _reward(
        env,
        rag_mod,
        monkeypatch,
        "<answer>Berlin [Source: geography.md]</answer>",
        judge_score=0.0,  # judged wrong
    )
    assert r["answer_correctness"] == 0.0
    assert r["citation_precision"] == 0.0  # gated → 0 on a wrong answer
    assert r["answer_length"] == 0.0  # gated → 0
    assert r["retrieval_hit"] > 0.0  # UNGATED — found + cited the gold source


def test_reward_no_answer_block_all_zero(rag_mod, monkeypatch):
    """No <answer> tag → all components 0, and the judge is never even called."""
    env = _bare_env(rag_mod)
    calls: list = []

    async def _rec(**kw):
        calls.append(1)
        return {"score": 1.0}

    monkeypatch.setattr(rag_mod, "evaluate_single_rubric", _rec)
    msgs = [{"role": "assistant", "content": "I think it's Paris but no answer tag"}]
    task = {"question": "Q", "ground_truth": "Paris", "reference_chunks": []}
    r = asyncio.run(env.compute_reward("r", msgs, task, termination_reason="finished"))
    assert set(r) == set(rag_mod.REWARD_KEYS)
    assert all(v == 0.0 for v in r.values())
    assert not calls  # short-circuited before the judge call


def test_reward_correct_cited_concise_all_positive(rag_mod, monkeypatch):
    env = _bare_env(rag_mod)
    r = _reward(
        env,
        rag_mod,
        monkeypatch,
        "<answer>Paris [Source: geography.md]</answer>",
        judge_score=1.0,
    )
    assert r["answer_correctness"] > 0
    assert r["retrieval_hit"] > 0
    assert r["citation_precision"] > 0  # cited only the gold → precision 1, gated ok
    assert r["answer_length"] > 0  # short + correct


def test_reward_correct_no_citation_no_retrieval_hit(rag_mod, monkeypatch):
    """A correct answer that cites nothing gets correctness but zero retrieval_hit —
    the boilerplate-answer analog (no evidence surfaced → no retrieval credit)."""
    env = _bare_env(rag_mod)
    r = _reward(env, rag_mod, monkeypatch, "<answer>Paris</answer>", judge_score=1.0)
    assert r["answer_correctness"] > 0
    assert r["retrieval_hit"] == 0.0


def test_reward_verbose_answer_zero_length_term(rag_mod, monkeypatch):
    env = _bare_env(rag_mod)
    long = "Paris is the capital of France. " * 40  # >> ANSWER_LENGTH_CAP chars
    r = _reward(
        env,
        rag_mod,
        monkeypatch,
        f"<answer>{long}[Source: geography.md]</answer>",
        judge_score=1.0,
    )
    assert r["answer_correctness"] > 0
    assert r["answer_length"] == 0.0  # length_score clamps to 0 above the cap


def test_reward_citation_matches_title_path_variant(rag_mod, monkeypatch):
    """_canonicalize_id matches a title-path citation to a bare-id gold (id-hash OR
    title-path): cited 'docs/Geography.md' vs gold 'geography'."""
    env = _bare_env(rag_mod)
    r = _reward(
        env,
        rag_mod,
        monkeypatch,
        "<answer>Paris [Source: docs/Geography.md]</answer>",
        judge_score=0.0,  # even wrong, retrieval_hit should fire on the match
        gold_file="geography",
    )
    assert r["retrieval_hit"] > 0.0


# ── probe: read-only, non-interactive gold-hit@k ────────────────────────────────


class _FakeCorpus:
    def __init__(self, id, name):
        self.id = id
        self.name = name


class _FakeChunk:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeSearchResult:
    def __init__(self, results):
        self.results = results


def _fake_client_cls(corpora, results_by_query, asearch_calls):
    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def list_corpora(self):
            return corpora

        async def asearch(self, corpus_id, query, limit):
            asearch_calls.append((corpus_id, query, limit))
            return _FakeSearchResult(results_by_query.get(query, []))

    return _FakeClient


def test_probe_gold_hit_over_synthetic_corpus(rag_mod, monkeypatch):
    calls: list = []
    corpora = [_FakeCorpus("cid-1", rag_mod.CORPUS_NAME)]
    results = {"capital of France?": [_FakeChunk({"file": "geography.md"})]}
    monkeypatch.setattr(
        rag_mod, "CorpusClient", _fake_client_cls(corpora, results, calls)
    )
    env = discover_env_class(rag_mod).__new__(discover_env_class(rag_mod))
    eval_rows = [
        {
            "question": "capital of France?",
            "reference_chunks": [{"metadata": {"file": "geography.md"}}],
        }
    ]
    out = asyncio.run(env.validate_probe(eval_rows))
    assert out["ok"] is True
    assert out["gold_hit_at_k"] == 1.0
    assert f"gold-hit@{rag_mod.PROBE_TOP_K}" in out["summary"]
    assert calls and calls[0][0] == "cid-1"  # searched the resolved corpus id


def test_probe_skips_when_corpus_not_ingested(rag_mod, monkeypatch):
    """No corpus matching CORPUS_NAME → skipped, and asearch is NEVER called (no
    corpus creation, no stdin prompt)."""
    calls: list = []
    monkeypatch.setattr(rag_mod, "CorpusClient", _fake_client_cls([], {}, calls))
    env = discover_env_class(rag_mod).__new__(discover_env_class(rag_mod))
    eval_rows = [
        {"question": "q", "reference_chunks": [{"metadata": {"file": "x.md"}}]}
    ]
    out = asyncio.run(env.validate_probe(eval_rows))
    assert out["ok"] is False
    assert "not ingested" in out["summary"]
    assert not calls  # never searched → never created / hung


def test_probe_skips_when_no_reference_chunks(rag_mod, monkeypatch):
    """Eval rows without gold reference_chunks → skipped before the client is even
    constructed (so an unreachable backend can't hang the probe)."""
    built: list = []

    class _Tripwire:
        def __init__(self, **kwargs):
            built.append(1)

    monkeypatch.setattr(rag_mod, "CorpusClient", _Tripwire)
    env = discover_env_class(rag_mod).__new__(discover_env_class(rag_mod))
    out = asyncio.run(env.validate_probe([{"question": "q"}]))
    assert out["ok"] is False
    assert "reference_chunks" in out["summary"]
    assert not built  # short-circuited before constructing the client
