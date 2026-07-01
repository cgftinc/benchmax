"""Unit tests for the `castform runs` command group (slice 1.2).

Offline: the platform client is replaced with a fake returning canned JSON, so
these exercise the CLI formatting + mode-selection logic without a network. The
"output matches the web-app view" half of the gate is the staging fixture check.
"""

from __future__ import annotations

import argparse

from benchmax.cli import runs
from benchmax.platform.exceptions import AuthenticationError, TrainerError


class _FakeClient:
    def __init__(self, **canned):
        self.canned = canned
        self.calls: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def list_runs(self, **_k):
        if "raise" in self.canned:
            raise self.canned["raise"]
        return self.canned.get("runs", [])

    def get_run(self, run_id, **_k):
        return self.canned.get("run", {})

    def get_run_details(self, run_id):
        return self.canned.get("details", {})

    def get_run_scalars(self, run_id, mode):
        self.calls["scalars_mode"] = mode
        return self.canned.get("scalars", {})

    def get_environment_logs(self, run_id, **_k):
        return self.canned.get("logs", [])

    def get_rollout_summary(self, run_id, **_k):
        return self.canned.get("summary", [])

    def get_rollout_heatmap(self, run_id, prompt_message_id, **_k):
        self.calls["heatmap_example"] = prompt_message_id
        return self.canned.get("heatmap", [])

    def get_rollout_details(self, run_id, rollout_id):
        self.calls["details_rollout"] = rollout_id
        return self.canned.get("rollout_details", {})

    def get_rollout_mode_average(self, run_id, **_k):
        return self.canned.get("mode_average", {})


def _patch(monkeypatch, **canned) -> _FakeClient:
    client = _FakeClient(**canned)
    monkeypatch.setattr(runs, "trainer_client", lambda: client)
    return client


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_runs_list_table(monkeypatch, capsys):
    _patch(
        monkeypatch,
        runs=[
            {
                "id": "r1",
                "name": "alpha",
                "status": "active",
                "createdAt": "2026-06-16T00:00:00Z",
            }
        ],
    )
    assert runs._cmd_runs_list(_ns(json=False)) == 0
    out = capsys.readouterr().out
    assert "ID" in out and "STATUS" in out
    assert "r1" in out and "alpha" in out and "active" in out


def test_runs_list_empty(monkeypatch, capsys):
    _patch(monkeypatch, runs=[])
    assert runs._cmd_runs_list(_ns(json=False)) == 0
    assert "No runs" in capsys.readouterr().out


def test_runs_list_json(monkeypatch, capsys):
    _patch(monkeypatch, runs=[{"id": "r1"}])
    assert runs._cmd_runs_list(_ns(json=True)) == 0
    assert '"r1"' in capsys.readouterr().out


def test_runs_list_not_logged_in(monkeypatch, capsys):
    _patch(monkeypatch, **{"raise": AuthenticationError("nope", 401)})
    assert runs._cmd_runs_list(_ns(json=False)) == 1
    assert "login" in capsys.readouterr().err


def test_runs_list_server_error(monkeypatch, capsys):
    _patch(monkeypatch, **{"raise": TrainerError("boom", 500)})
    assert runs._cmd_runs_list(_ns(json=False)) == 1
    assert "boom" in capsys.readouterr().err


def test_runs_get(monkeypatch, capsys):
    _patch(
        monkeypatch,
        run={"id": "r1", "name": "alpha", "status": "complete", "isOwner": True},
    )
    assert runs._cmd_runs_get(_ns(run_id="r1", config=False, json=False)) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "complete" in out and "/train/r1" in out


def test_runs_status_with_progress(monkeypatch, capsys):
    _patch(
        monkeypatch,
        run={
            "status": "active",
            "totalSteps": 10,
            "latestActivityMessage": "step 4 done",
        },
        details={"latestStep": 4, "errorCount": 0},
    )
    assert runs._cmd_runs_status(_ns(run_id="r1", json=False)) == 0
    out = capsys.readouterr().out
    assert "active" in out and "4 / 9" in out and "step 4 done" in out


def test_runs_scalars_default_mode_prefers_train(monkeypatch, capsys):
    client = _patch(
        monkeypatch,
        details={"modes": ["eval", "train"]},
        scalars={"reward": [{"step": 1, "value": 0.5}]},
    )
    assert runs._cmd_runs_scalars(_ns(run_id="r1", mode=None, json=False)) == 0
    assert client.calls["scalars_mode"] == "train"  # train preferred over eval
    out = capsys.readouterr().out
    assert "mode=train" in out and "reward" in out


def test_runs_scalars_explicit_mode(monkeypatch, capsys):
    client = _patch(monkeypatch, scalars={"loss": [{"step": 2, "value": 0.1}]})
    assert runs._cmd_runs_scalars(_ns(run_id="r1", mode="eval", json=False)) == 0
    assert client.calls["scalars_mode"] == "eval"
    assert "loss" in capsys.readouterr().out


def test_runs_scalars_no_modes(monkeypatch, capsys):
    _patch(monkeypatch, details={"modes": []})
    assert runs._cmd_runs_scalars(_ns(run_id="r1", mode=None, json=False)) == 0
    assert "No scalars yet" in capsys.readouterr().out


def test_runs_logs(monkeypatch, capsys):
    _patch(
        monkeypatch,
        logs=[
            {
                "createdAt": "t0",
                "level": "ERROR",
                "content": "boom",
                "traceback": "Trace\nline",
            }
        ],
    )
    assert runs._cmd_runs_logs(_ns(run_id="r1", rollout_id=None, json=False)) == 0
    out = capsys.readouterr().out
    assert "ERROR" in out and "boom" in out and "Trace" in out


# --- stored-rollout commands (rollouts / rollout) -----------------------


def _rollouts_ns(**kw):
    base = dict(run_id="r1", mode="eval", example=None, limit=50, json=False)
    base.update(kw)
    return _ns(**base)


def _rollout_ns(**kw):
    base = dict(run_id="r1", rollout_id="ro1", dataset=None, view=False, json=False)
    base.update(kw)
    return _ns(**base)


def test_runs_rollouts_summary_table(monkeypatch, capsys):
    _patch(
        monkeypatch,
        summary=[
            {
                "promptMessageId": "ex1",
                "promptText": "where do I add the exception?",
                "rewardHistory": [
                    {"step": 0, "meanReward": 0.2},
                    {"step": 20, "meanReward": 0.7},
                ],
            }
        ],
        mode_average={"avg": 0.55},
    )
    assert runs._cmd_runs_rollouts(_rollouts_ns()) == 0
    out = capsys.readouterr().out
    assert "EXAMPLE ID" in out and "ex1" in out
    assert "0.7" in out  # latest mean reward (not the step-0 value)
    assert "0.55" in out  # mode average in the header


def test_runs_rollouts_example_heatmap(monkeypatch, capsys):
    client = _patch(
        monkeypatch,
        heatmap=[
            {"id": "roA", "step": 0, "totalReward": 0.1},
            {"id": "roB", "step": 20, "totalReward": 0.9},
        ],
    )
    assert runs._cmd_runs_rollouts(_rollouts_ns(example="ex1")) == 0
    out = capsys.readouterr().out
    assert client.calls["heatmap_example"] == "ex1"
    assert "roA" in out and "roB" in out
    assert "castform runs rollout r1" in out  # next-step hint


def test_runs_rollout_details_with_gold(monkeypatch, capsys, tmp_path):
    ds = tmp_path / "eval.jsonl"
    ds.write_text(
        '{"prompt": "where do I add the exception?", "ground_truth": "edit /etc/docker"}\n'
    )
    _patch(
        monkeypatch,
        rollout_details={
            "step": 139,
            "totalReward": 0.85,
            "promptMessages": [
                {"role": "user", "content": "where do I add the exception?"}
            ],
            "messages": [
                {"role": "user", "content": "where do I add the exception?"},
                {"role": "assistant", "content": "edit /etc/docker and reload"},
            ],
            "rewards": [
                {"name": "answer_correctness", "value": 1.0},
                {"name": "citation_recall", "value": 0.3},
            ],
        },
    )
    assert runs._cmd_runs_rollout(_rollout_ns(dataset=str(ds))) == 0
    out = capsys.readouterr().out
    assert "edit /etc/docker" in out  # gold, joined from local dataset
    assert "edit /etc/docker and reload" in out  # the model's answer
    assert "answer_correctness" in out and "citation_recall" in out
    assert "step 139" in out


def test_runs_rollout_json_attaches_gold(monkeypatch, capsys, tmp_path):
    ds = tmp_path / "eval.jsonl"
    ds.write_text('{"prompt": "Q?", "ground_truth": "GOLD"}\n')
    _patch(
        monkeypatch,
        rollout_details={
            "promptMessages": [{"role": "user", "content": "Q?"}],
            "messages": [{"role": "assistant", "content": "A"}],
            "rewards": [],
        },
    )
    assert runs._cmd_runs_rollout(_rollout_ns(dataset=str(ds), json=True)) == 0
    out = capsys.readouterr().out
    assert '"gold": "GOLD"' in out


def test_runs_rollout_gold_not_found_is_graceful(monkeypatch, capsys, tmp_path):
    _patch(
        monkeypatch,
        rollout_details={
            "promptMessages": [{"role": "user", "content": "Q?"}],
            "messages": [{"role": "assistant", "content": "A"}],
            "rewards": [],
        },
    )
    # dataset path that doesn't exist → no gold, but must not crash
    assert (
        runs._cmd_runs_rollout(_rollout_ns(dataset=str(tmp_path / "nope.jsonl"))) == 0
    )
    assert "not found locally" in capsys.readouterr().out


def test_gold_join_helpers(tmp_path):
    assert runs._user_prompt([{"role": "user", "content": "hi"}]) == "hi"
    assert (
        runs.final_answer(
            [{"role": "assistant", "content": "one"}, {"role": "user", "content": "q"}]
        )
        == "one"
    )
    idx = {"a b c": "GOLD"}
    assert runs._match_gold("a b c", idx) == "GOLD"  # exact
    assert runs._match_gold("prefix a b c suffix", idx) == "GOLD"  # containment
    assert runs._match_gold("unrelated", idx) is None
    assert runs._match_gold(None, idx) is None


def test_gold_index_reads_question_key(tmp_path):
    # Flagship RAG datasets key on 'question'/'answer' (no 'prompt'/'ground_truth').
    ds = tmp_path / "eval.jsonl"
    ds.write_text(
        '{"question": "what is X?", "answer": "X is Y", "reference_chunks": []}\n'
    )
    idx = runs._gold_index([str(ds)])
    assert idx == {
        "what is X?": "X is Y"
    }  # was empty before the fix (keyed on 'prompt')


def test_match_gold_longest_wins_no_substring_shadow(tmp_path):
    # When one question is a substring of another, the LONGER (more specific) wins,
    # and a bare substring can't shadow it.
    idx = {"reset password": "SHORT", "reset password on mobile": "LONG"}
    assert runs._match_gold("help: reset password on mobile please", idx) == "LONG"
    # reverse direction no longer matches (prompt shorter than the question)
    assert runs._match_gold("reset", idx) is None
