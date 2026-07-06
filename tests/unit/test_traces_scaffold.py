"""The traces seed's comparative reward + row-shape probe (`traces_main.py`).

dataset_preprocess is exercised on the new `to_jsonl_dict` row shape
(`prompt_messages` / message-dict `ground_truth` / `init_rollout_args` lineage);
the reward on fixtures with the judge monkeypatched (no network); the probe on
good/off-shape rows; and the committed seed jsonl on the exact shape contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import benchmax.cli.scaffold as scaffold_pkg
from benchmax.cli._project import _load_module_from_file, discover_env_class

_SCAFFOLD_DIR = Path(scaffold_pkg.__file__).parent
_TRACES_MAIN = _SCAFFOLD_DIR / "traces_main.py"


@pytest.fixture
def traces_mod():
    return _load_module_from_file(_TRACES_MAIN)


def _bare_env(traces_mod):
    """A CustomTraceEnv without __init__, with just the judge attrs the reward reads."""
    cls = discover_env_class(traces_mod)
    env = cls.__new__(cls)
    env._judge_model = "m"
    env._judge_base_url = "u"
    env._judge_timeout = 42.0
    env._judge_token_provider = lambda: "k"
    return env


def _tool_call(name, arguments, cid="call_1"):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _row():
    """One `castform data traces` row in the to_jsonl_dict shape."""
    return {
        "prompt_messages": [
            {
                "role": "user",
                "content": "Cancel order #7 please.",
                "tool_calls": [],
                "tool_call_id": "",
                "name": "",
            }
        ],
        "ground_truth": {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("cancel_order", '{"order_id": "7"}')],
            "tool_call_id": "",
            "name": "",
        },
        "init_rollout_args": {
            "trace_id": "t-1",
            "turn_index": 0,
            "total_messages": 2,
            "scores": {"quality": 0.9},
            "raw_prompt": "[USER] Cancel order #7 please.",
        },
    }


# ── dataset_preprocess: the new to_jsonl_dict shape maps through ────────────────


def test_dataset_preprocess_maps_jsonl_row(traces_mod):
    cls = discover_env_class(traces_mod)
    row = _row()
    ex = cls.dataset_preprocess(row)
    # prompt_messages pass through (system_prompt is the "" placeholder → no prepend)
    assert ex["prompt_messages"] == row["prompt_messages"]
    # ground_truth rides in task as the recorded MESSAGE DICT, not a string
    assert ex["task"] == {"ground_truth": row["ground_truth"]}
    # trace lineage forwards through init_rollout_args
    assert ex["init_rollout_args"] == row["init_rollout_args"]
    assert ex["id"]  # canonical example id computed


def test_dataset_preprocess_handles_missing_ground_truth(traces_mod):
    cls = discover_env_class(traces_mod)
    row = _row()
    del row["ground_truth"]
    ex = cls.dataset_preprocess(row)
    assert ex["task"] == {"ground_truth": {}}  # shape kept, read defensively


def test_dataset_preprocess_prepends_pasted_system_prompt(traces_mod):
    """Once the detected system prompt is pasted in, it heads the chat prefix."""
    cls = discover_env_class(traces_mod)
    try:
        cls.system_prompt = "You are the support agent."
        ex = cls.dataset_preprocess(_row())
    finally:
        cls.system_prompt = ""
    assert ex["prompt_messages"][0] == {
        "role": "system",
        "content": "You are the support agent.",
    }


# ── _gt_text: the reference the judge compares against ──────────────────────────


def test_gt_text_tool_call_only(traces_mod):
    """A gold turn with empty content but tool calls must render the calls — an
    empty reference would zero every action row."""
    gt = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_tool_call("lookup_order", '{"order_id": "48219"}')],
    }
    text = traces_mod._gt_text(gt)
    assert 'tool_call: lookup_order({"order_id": "48219"})' == text


def test_gt_text_content_plus_tool_call(traces_mod):
    gt = {
        "role": "assistant",
        "content": "Refunding the duplicate now.",
        "tool_calls": [_tool_call("issue_refund", '{"invoice_id": "INV-1"}')],
    }
    text = traces_mod._gt_text(gt)
    assert text.startswith("Refunding the duplicate now.")
    assert 'tool_call: issue_refund({"invoice_id": "INV-1"})' in text


def test_gt_text_tolerates_flat_tool_calls_and_empty(traces_mod):
    # flat {name, arguments} (no nested "function") is tolerated at read time
    flat = {"content": "", "tool_calls": [{"name": "ping", "arguments": "{}"}]}
    assert traces_mod._gt_text(flat) == "tool_call: ping({})"
    assert traces_mod._gt_text({}) == ""


# ── compute_reward: judge success / failure, zeros short-circuits ────────────────


def _judge(score):
    async def _fake(**kw):
        return {"score": score}

    return _fake


_MSGS = [
    {"role": "user", "content": "Cancel order #7 please."},
    {"role": "assistant", "content": "I'll cancel order #7 right away."},
]


def test_reward_judge_success_threads_env_config(traces_mod, monkeypatch):
    env = _bare_env(traces_mod)
    captured: dict = {}

    async def _fake(**kw):
        captured.update(kw)
        return {"score": 1.0}

    monkeypatch.setattr(traces_mod, "evaluate_single_rubric", _fake)
    task = {"ground_truth": _row()["ground_truth"]}
    r = asyncio.run(env.compute_reward("r", _MSGS, task))
    assert r == {"ground_truth_match": 1.0}
    # the judge compares against the RENDERED gold turn (tool-call path here)
    assert "tool_call: cancel_order" in captured["ground_truth"]
    assert captured["question"] == "Cancel order #7 please."
    assert captured["timeout"] == 42.0  # env judge config threaded


def test_reward_partial_match_score(traces_mod, monkeypatch):
    env = _bare_env(traces_mod)
    monkeypatch.setattr(traces_mod, "evaluate_single_rubric", _judge(0.5))
    r = asyncio.run(
        env.compute_reward("r", _MSGS, {"ground_truth": _row()["ground_truth"]})
    )
    assert r == {"ground_truth_match": 0.5}


def test_reward_judge_failure_scores_zero_without_crashing(traces_mod, monkeypatch):
    env = _bare_env(traces_mod)

    async def _boom(**kw):
        raise RuntimeError("judge down")

    monkeypatch.setattr(traces_mod, "evaluate_single_rubric", _boom)
    r = asyncio.run(
        env.compute_reward("r", _MSGS, {"ground_truth": _row()["ground_truth"]})
    )
    assert r == {"ground_truth_match": 0.0}


def test_reward_empty_completion_zeros_judge_not_called(traces_mod, monkeypatch):
    env = _bare_env(traces_mod)
    calls: list = []

    async def _rec(**kw):
        calls.append(1)
        return {"score": 1.0}

    monkeypatch.setattr(traces_mod, "evaluate_single_rubric", _rec)
    msgs = [{"role": "user", "content": "q"}]  # no assistant turn at all
    r = asyncio.run(
        env.compute_reward("r", msgs, {"ground_truth": _row()["ground_truth"]})
    )
    assert r == {"ground_truth_match": 0.0}
    assert not calls  # short-circuited before the judge call


def test_reward_missing_ground_truth_zeros_judge_not_called(traces_mod, monkeypatch):
    """An empty gold turn ({} row) scores 0 rather than judging against nothing."""
    env = _bare_env(traces_mod)
    calls: list = []

    async def _rec(**kw):
        calls.append(1)
        return {"score": 1.0}

    monkeypatch.setattr(traces_mod, "evaluate_single_rubric", _rec)
    r = asyncio.run(env.compute_reward("r", _MSGS, {"ground_truth": {}}))
    assert r == {"ground_truth_match": 0.0}
    assert not calls


# ── validate_probe: row-shape + trace coverage, ok and failure shapes ────────────


def test_probe_ok_counts_rows_traces_and_tool_turns(traces_mod):
    env = discover_env_class(traces_mod).__new__(discover_env_class(traces_mod))
    r1 = _row()
    r2 = _row()
    r2["ground_truth"] = {"role": "assistant", "content": "Done!", "tool_calls": []}
    r2["init_rollout_args"] = dict(r1["init_rollout_args"], trace_id="t-2")
    out = asyncio.run(env.validate_probe([r1, r2]))
    assert out["ok"] is True
    assert out["rows"] == 2 and out["traces"] == 2
    assert out["tool_call_rows"] == 1
    assert "2 rows / 2 traces" in out["summary"] and "shape OK" in out["summary"]


def test_probe_flags_off_shape_rows(traces_mod):
    """The OLD wizard shape (string ground_truth / missing prompt_messages) must
    fail the probe with a pointer at `castform data traces`."""
    env = discover_env_class(traces_mod).__new__(discover_env_class(traces_mod))
    stale = {
        "prompt_messages": [{"role": "user", "content": "q"}],
        "ground_truth": "Paris",
    }
    out = asyncio.run(env.validate_probe([_row(), stale]))
    assert out["ok"] is False
    assert out["bad_rows"] == [1]
    assert "castform data traces" in out["summary"]


def test_probe_empty_dataset_skips(traces_mod):
    env = discover_env_class(traces_mod).__new__(discover_env_class(traces_mod))
    out = asyncio.run(env.validate_probe([]))
    assert out["ok"] is False
    assert "skipped" in out["summary"]


# ── the committed seed datasets honor the shape contract ─────────────────────────


def _load_seed(name):
    rows = []
    for line in (_SCAFFOLD_DIR / name).read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize(
    "name, expected",
    [("traces_train_dataset.jsonl", 6), ("traces_eval_dataset.jsonl", 3)],
)
def test_seed_datasets_parse_and_match_shape(traces_mod, name, expected):
    rows = _load_seed(name)
    assert len(rows) == expected
    for row in rows:
        assert set(row) == {"prompt_messages", "ground_truth", "init_rollout_args"}
        assert isinstance(row["prompt_messages"], list) and row["prompt_messages"]
        for m in row["prompt_messages"]:
            assert set(m) == {"role", "content", "tool_calls", "tool_call_id", "name"}
        gt = row["ground_truth"]
        assert isinstance(gt, dict) and gt["role"] == "assistant"
        args = row["init_rollout_args"]
        assert set(args) == {
            "trace_id",
            "turn_index",
            "total_messages",
            "scores",
            "raw_prompt",
        }
        assert isinstance(args["scores"], dict)
        assert isinstance(args["raw_prompt"], str) and args["raw_prompt"]
    # at least one gold turn is a pure tool call (empty content) so the reward's
    # _gt_text tool-call path is exercised end-to-end by the seed
    assert any(
        not r["ground_truth"]["content"] and r["ground_truth"]["tool_calls"]
        for r in rows
    )


def test_seed_datasets_pass_the_probe_and_preprocess(traces_mod):
    """Day-one contract: the committed seeds satisfy the env's own probe and map
    through dataset_preprocess without error."""
    cls = discover_env_class(traces_mod)
    env = cls.__new__(cls)
    rows = _load_seed("traces_eval_dataset.jsonl")
    out = asyncio.run(env.validate_probe(rows))
    assert out["ok"] is True
    for row in _load_seed("traces_train_dataset.jsonl") + rows:
        ex = cls.dataset_preprocess(row)
        assert ex["prompt_messages"] and ex["task"]["ground_truth"]
