"""The judge seed's 8-component gated reward + fixture-harness probe
(`judge_main.py`): parser forgiveness, the gating chain, exploit ceilings,
gold-rescore on the committed datasets, leakage/challenge guards, and the
deterministic data generator. Loaded by path — the seed is not (yet) in the
template registry."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import benchmax.cli.scaffold as scaffold_pkg
from benchmax.cli._project import _load_module_from_file, discover_env_class

_SCAFFOLD = Path(scaffold_pkg.__file__).parent
_JUDGE_MAIN = _SCAFFOLD / "judge_main.py"


@pytest.fixture
def judge_mod():
    return _load_module_from_file(_JUDGE_MAIN)


@pytest.fixture
def env(judge_mod):
    return discover_env_class(judge_mod)()


def _rows(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (_SCAFFOLD / name).read_text("utf-8").splitlines()
        if line.strip()
    ]


def _eval_rows() -> list[dict]:
    return _rows("judge_eval_dataset.jsonl")


def _multi_row(rows: list[dict]) -> dict:
    """A row with >= 2 gold assistant turns, so per-turn fractions are visible."""
    return next(r for r in rows if len(r["ground_truth"]["turn_guidelines"]) >= 2)


def _score(env, mod, row: dict, text: str) -> dict[str, float]:
    return asyncio.run(
        env.compute_reward(
            "r", [{"role": "assistant", "content": text}], mod._task_from_row(row)
        )
    )


def _mutated(row: dict, fn) -> str:
    g = json.loads(json.dumps(row["ground_truth"]))  # deep copy
    fn(g["turn_guidelines"])
    return json.dumps(g)


def _generic_reason(gu: dict) -> str:
    # guideline/category words only — the boilerplate exploit
    return (
        f"violates the {gu['guidance_category']} {gu['guidance_key']} "
        f"guideline at phase {gu['guideline_phase']}"
    )


# ── parser: strict + prose forgiveness + malformed ───────────────────────────


def test_parser_strict_prose_and_malformed(judge_mod):
    row = _eval_rows()[0]
    gold = json.dumps(row["ground_truth"])
    n = len(row["ground_truth"]["turn_guidelines"])

    assert len(judge_mod._parse_judgment(gold)) == n
    # forgiveness: surrounding prose is stripped when the outermost {...} parses
    assert len(judge_mod._parse_judgment(f"My audit:\n{gold}\nDone.")) == n
    # malformed / pure prose / wrong shapes → None (the schema gate)
    assert judge_mod._parse_judgment('{"turn_guidelines": [') is None
    assert judge_mod._parse_judgment("no json here at all") is None
    assert judge_mod._parse_judgment('{"turn_guidelines": "nope"}') is None
    # wrong-typed fields inside an entry → None
    bad = json.loads(gold)
    bad["turn_guidelines"][0]["guideline_used"]["is_violation"] = "false"
    assert judge_mod._parse_judgment(json.dumps(bad)) is None
    bad = json.loads(gold)
    bad["turn_guidelines"][0]["turn_index"] = True  # bool is not an index
    assert judge_mod._parse_judgment(json.dumps(bad)) is None


def test_reward_malformed_output_all_zero(judge_mod, env):
    row = _eval_rows()[0]
    r = _score(env, judge_mod, row, "I think turn 1 breaks the fee rule.")
    assert set(r) == set(judge_mod.REWARD_KEYS)
    assert all(v == 0.0 for v in r.values())  # schema=0 zeroes everything


# ── happy path: gold output pays every component in full ────────────────────


def test_gold_output_full_reward_every_component(judge_mod, env):
    row = _multi_row(_eval_rows())
    r = _score(env, judge_mod, row, json.dumps(row["ground_truth"]))
    for k in judge_mod.REWARD_KEYS:
        assert r[k] == pytest.approx(1.0), k
    assert sum(r.values()) == pytest.approx(judge_mod.MAX_TOTAL_REWARD)


def test_prose_wrapped_gold_scores_like_gold(judge_mod, env):
    row = _eval_rows()[0]
    wrapped = "Sure — here's the judgment:\n" + json.dumps(row["ground_truth"])
    r = _score(env, judge_mod, row, wrapped)
    assert r["schema"] == 1.0
    assert sum(r.values()) == pytest.approx(judge_mod.MAX_TOTAL_REWARD)


# ── the gating chain: violation_flag pays only on key+phase+supported reason ─


def test_wrong_key_gates_violation_flag_on_that_turn(judge_mod, env):
    row = _multi_row(_eval_rows())
    n = len(row["ground_truth"]["turn_guidelines"])
    keys = sorted({g["guidance_key"] for g in row["guidelines"]})
    gold0 = row["ground_truth"]["turn_guidelines"][0]["guideline_used"]
    other = next(k for k in keys if k != gold0["guidance_key"])

    def _set_key(labels):
        labels[0]["guideline_used"]["guidance_key"] = other

    r = _score(env, judge_mod, row, _mutated(row, _set_key))
    assert r["turn_key"] == pytest.approx((n - 1) / n)
    # gated: the flag on that turn pays 0 even though its is_violation is right
    assert r["violation_flag"] == pytest.approx((n - 1) / n)
    assert r["turn_category"] == 1.0  # category untouched
    assert r["turn_phase"] == 1.0


def test_unsupported_reason_gates_violation_flag_on_that_turn(judge_mod, env):
    row = _multi_row(_eval_rows())
    n = len(row["ground_truth"]["turn_guidelines"])

    def _boilerplate_first(labels):
        gu = labels[0]["guideline_used"]
        gu["judge_reason"] = _generic_reason(gu)

    r = _score(env, judge_mod, row, _mutated(row, _boilerplate_first))
    assert r["turn_key"] == 1.0  # the label itself is still right...
    assert r["violation_flag"] == pytest.approx((n - 1) / n)  # ...but unsupported
    assert r["reason_support"] == pytest.approx((n - 1) / n)


# ── user-turn labels: precision + support penalized ─────────────────────────


def test_user_turn_label_penalized(judge_mod, env):
    row = _multi_row(_eval_rows())
    user_idx = next(
        m["turn_index"] for m in row["turn_meta"] if m["role"] == "user"
    )

    def _add_user_label(labels):
        extra = json.loads(json.dumps(labels[0]))
        extra["turn_index"] = user_idx
        labels.append(extra)

    r = _score(env, judge_mod, row, _mutated(row, _add_user_label))
    assert r["coverage_recall"] == 1.0  # gold turns still all labeled
    assert r["coverage_precision"] < 1.0  # the user-turn label zero-contributes
    assert r["reason_support"] < 1.0  # extra label dilutes support


# ── the boilerplate exploit ceiling (U5): total ≤ 6.0 ────────────────────────


def test_generic_reason_exploit_ceiling(judge_mod, env):
    row = _multi_row(_eval_rows())

    def _all_generic(labels):
        for e in labels:
            e["guideline_used"]["judge_reason"] = _generic_reason(e["guideline_used"])

    r = _score(env, judge_mod, row, _mutated(row, _all_generic))
    assert r["reason_support"] <= 0.05
    assert r["violation_flag"] == 0.0  # gated to zero on every turn
    total = sum(r.values())
    assert total <= judge_mod.GENERIC_REASON_CEILING + 1e-9
    assert total == pytest.approx(judge_mod.GENERIC_REASON_CEILING, abs=0.05)


# ── coverage: missing + duplicate labels ─────────────────────────────────────


def test_missing_turn_scores_below_full(judge_mod, env):
    row = _multi_row(_eval_rows())
    n = len(row["ground_truth"]["turn_guidelines"])
    r = _score(env, judge_mod, row, _mutated(row, lambda ls: ls.pop()))
    assert r["coverage_recall"] == pytest.approx((n - 1) / n)
    assert sum(r.values()) < judge_mod.MAX_TOTAL_REWARD - 1e-6


def test_duplicate_turn_label_hits_precision(judge_mod, env):
    row = _multi_row(_eval_rows())

    def _dup(labels):
        labels.append(json.loads(json.dumps(labels[0])))

    r = _score(env, judge_mod, row, _mutated(row, _dup))
    assert r["coverage_recall"] == 1.0
    assert r["coverage_precision"] < 1.0


# ── gold-rescore on BOTH committed datasets: every row must pay 8.0 ──────────


def test_gold_rescore_full_reward_on_committed_datasets(judge_mod, env):
    for name in ("judge_train_dataset.jsonl", "judge_eval_dataset.jsonl"):
        rows = _rows(name)
        assert rows, name
        for row in rows:
            r = _score(env, judge_mod, row, json.dumps(row["ground_truth"]))
            assert sum(r.values()) == pytest.approx(judge_mod.MAX_TOTAL_REWARD), (
                f"{name}:{row['trace_id']} rescored {r}"
            )


# ── the probe: fixture harness + leakage + challenge guards ─────────────────


def test_probe_ok_on_committed_eval(judge_mod, env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no train file / challenges dir — still green
    out = asyncio.run(env.validate_probe(_eval_rows()))
    assert out["ok"] is True
    assert "gold-rescore 4/4" in out["summary"]
    assert "0 failures" in out["summary"]
    assert "leakage: none" in out["summary"]


def test_probe_leakage_guard_trips_on_poisoned_row(judge_mod, env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _eval_rows()
    rows[0]["prompt"] += '\nHint: {"is_violation": true} on turn 1.'
    out = asyncio.run(env.validate_probe(rows))
    assert out["ok"] is False
    assert any("is_violation" in leak for leak in out["leaks"])


def test_probe_leakage_guard_trips_on_reason_echo(judge_mod, env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _eval_rows()
    gold_reason = rows[0]["ground_truth"]["turn_guidelines"][0]["guideline_used"][
        "judge_reason"
    ]
    rows[0]["prompt"] += "\n" + gold_reason
    out = asyncio.run(env.validate_probe(rows))
    assert out["ok"] is False
    assert any("judge_reason" in leak for leak in out["leaks"])


def test_probe_challenge_overlap_trips(judge_mod, env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _eval_rows()
    cdir = tmp_path / judge_mod.CHALLENGES_DIR
    cdir.mkdir()
    (cdir / "audit.jsonl").write_text(
        json.dumps({"trace_id": rows[0]["trace_id"], "prompt": "x"}) + "\n", "utf-8"
    )
    out = asyncio.run(env.validate_probe(rows))
    assert out["ok"] is False
    assert "OVERLAP" in out["summary"]


def test_probe_challenge_disjoint_passes_and_is_counted(
    judge_mod, env, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cdir = tmp_path / judge_mod.CHALLENGES_DIR
    cdir.mkdir()
    (cdir / "audit.jsonl").write_text(
        json.dumps({"trace_id": "othercorp-000", "prompt": "x"}) + "\n", "utf-8"
    )
    out = asyncio.run(env.validate_probe(_eval_rows()))
    assert out["ok"] is True
    assert "1 rows / 1 file(s)" in out["summary"]


# ── renderer, preprocessing, generator, and the no-tools contract ────────────


def test_render_trace_readout_smoke(judge_mod, env):
    rows = _eval_rows()[:2]
    results = [
        asyncio.run(
            env.compute_reward(
                "r",
                [{"role": "assistant", "content": json.dumps(row["ground_truth"])}],
                judge_mod._task_from_row(row),
            )
        )
        for row in rows
    ]
    text = judge_mod.render_trace_readout(rows, results)
    assert rows[0]["trace_id"] in text and rows[1]["trace_id"] in text
    assert "phase" in text  # per-turn gold guideline line rendered
    assert "schema=1.00" in text  # per-component rewards rendered
    # results are optional — gold timeline alone still renders
    assert rows[0]["trace_id"] in judge_mod.render_trace_readout(rows)


def test_dataset_preprocess_maps_prompt_and_hidden_task(judge_mod):
    cls = discover_env_class(judge_mod)
    row = _eval_rows()[0]
    ex = cls.dataset_preprocess(row)
    assert ex["prompt_messages"][0] == {"role": "system", "content": cls.system_prompt}
    assert ex["prompt_messages"][1] == {"role": "user", "content": row["prompt"]}
    assert set(ex["task"]) == {"ground_truth", "turn_meta", "guidelines", "trace_id"}
    assert ex["task"]["ground_truth"] == row["ground_truth"]
    assert ex["id"]  # canonical id computed


def test_generate_data_deterministic_and_matches_committed(
    judge_mod, tmp_path, monkeypatch
):
    committed_train = (_SCAFFOLD / "judge_train_dataset.jsonl").read_bytes()
    committed_eval = (_SCAFFOLD / "judge_eval_dataset.jsonl").read_bytes()
    monkeypatch.chdir(tmp_path)
    assert judge_mod.generate_data(force=True)
    first = (
        Path(judge_mod.TRAIN_FILE).read_bytes(),
        Path(judge_mod.EVAL_FILE).read_bytes(),
    )
    assert judge_mod.generate_data(force=True)  # regenerate over existing files
    second = (
        Path(judge_mod.TRAIN_FILE).read_bytes(),
        Path(judge_mod.EVAL_FILE).read_bytes(),
    )
    assert first == second  # deterministic generator
    assert first == (committed_train, committed_eval)  # committed = exact output


def test_single_turn_no_tools_contract(judge_mod, env):
    assert asyncio.run(env.list_tools()) == []
    assert asyncio.run(env.run_tool("r", "anything", x=1)) == ""
