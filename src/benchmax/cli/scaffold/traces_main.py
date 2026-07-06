"""Trace-replay environment (written by `castform setup --template traces`).

Post-trains a model to imitate a recorded agent: each dataset row is ONE
assistant turn from a real trace — `prompt_messages` is the conversation before
that turn, `ground_truth` is the recorded assistant message (a DICT with
content + tool_calls, or `{}` when none survived) — and the model is scored on
reproducing that turn. Single-turn, no live tools: the recorded tools are not
re-executed here, so `list_tools` returns [] and the model answers in one shot.

Data provenance: the real train/eval jsonl come from `castform data traces`
(rows are the `{prompt_messages, ground_truth, init_rollout_args}` shape —
`TrainingExample.to_jsonl_dict`). That command also DETECTS the recorded
agent's system prompt — paste it into `system_prompt` below so training runs
against the same context the gold turns saw. The committed seed datasets are a
tiny synthetic customer-support corpus so `castform validate` runs on day one.

The whole run is reproducible from this file: the reward is below (ONE
comparative LLM-judge component — does the completion match the recorded turn
in action and substance?), and the `VALIDATE_CONFIG` / `LAUNCH_CONFIG` blocks
bake in the rollout budgets so `castform validate` / `castform launch` need no
extra flags (a CLI flag still overrides). `python main.py` drives the loop:
data → validate, then STOP (`launch` is an explicit, confirmed step — it
spends GPU credits).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

from benchmax import config
from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.reward_helpers import clip01, extract_completion_text
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.platform.client import TrainerClient
from benchmax.platform.credentials import as_token_provider, platform_bearer
from benchmax.platform.login import ensure_session
from benchmax.platform.training_run import upload_training_run
from benchmax.platform.validation import run_validate_probe, validate_env
from benchmax.rubrics.rubric import Rubric, evaluate_single_rubric

# Judge model for the ground-truth-match reward component (LLM, no GPU).
JUDGE_MODEL = "gpt-5.4-mini"

# Comparative rubric: the completion is judged AGAINST the recorded turn, never
# on an absolute quality scale — "same action and substance as the reference"
# is far more stable than an absolute 1–10 score.
GROUND_TRUTH_MATCH_RUBRIC = Rubric(
    title="Ground-truth match",
    description=(
        "Response matches the reference response in action and substance: it "
        "takes the same action (the same tool call with equivalent arguments "
        "when the reference calls a tool, a direct reply when the reference "
        "replies directly) and conveys the same key content."
    ),
    type="positive",
    score_map={
        0: "Different action, or contradicts the reference response.",
        0.5: "Same action, but the substance only partly matches (missing or "
        "wrong key details).",
        1: "Same action and same substance as the reference response.",
    },
)

REWARD_KEYS = ("ground_truth_match",)

logger = logging.getLogger(__name__)


def _gt_text(gt: dict[str, Any]) -> str:
    """Render the recorded assistant turn as comparable reference text: its
    content plus one compact ``tool_call: name(arguments)`` line per recorded
    call. A gold turn that only called a tool has empty content — the calls ARE
    its substance, so they must render or the judge gets an empty reference."""
    parts: list[str] = []
    content = str(gt.get("content") or "").strip()
    if content:
        parts.append(content)
    for tc in gt.get("tool_calls") or []:
        fn = tc.get("function") or tc  # OpenAI nested format; flat tolerated
        name = str(fn.get("name") or "").strip()
        if name:
            parts.append(f"tool_call: {name}({fn.get('arguments') or ''})")
    return "\n".join(parts)


def _last_user_text(messages: Messages) -> str:
    """The most recent user turn — the 'question' framing for the judge."""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


class CustomTraceEnv(BaseEnv):
    """Replay one recorded assistant turn per rollout, judged against the gold.

    Single-turn and tool-free: the recorded tools have no live backend here, so
    the model *declares* its action in text (the judge compares it to the gold
    turn's content + tool calls). Wire real tools only when you have a backend
    for `run_tool` — a fake tool layer validates green while every call errors.
    """

    # `castform data traces` DETECTS the recorded agent's system prompt (printed
    # on generation, or with --dry-run). PASTE IT HERE so training matches the
    # traces — an empty prompt grades the model on a different context than the
    # one the gold turns were produced under.
    system_prompt = ""

    # The gate component for `castform validate --reward-audit` scorecards.
    PRIMARY_REWARD_KEY = "ground_truth_match"

    # Extra pip deps the rollout sandbox needs (it bundles only main.py + benchmax).
    PIP_DEPENDENCIES: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Judge config lives in __init__, NOT at module level: the sandbox
        # UNPICKLES this env, so module-level code never runs there. config.*
        # resolves against the env's own domain inside the sandbox.
        self._judge_base_url = config.llm_url()
        self._judge_model = JUDGE_MODEL
        self._judge_token_provider = as_token_provider(None, platform_bearer)
        self._judge_timeout = 30.0

    async def list_tools(self) -> list[ToolDefinition]:
        return []  # trace replay: no live tools — the model answers in one turn

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> str:
        return ""  # unreachable while list_tools is []

    @classmethod
    def dataset_preprocess(cls, example: dict[str, Any], **kwargs: Any) -> Example:
        """Map one `castform data traces` row (`to_jsonl_dict` shape) to an Example.

        `prompt_messages` passes through as the chat prefix; `ground_truth` is
        the recorded assistant turn as a MESSAGE DICT (not a string — read its
        `content` / `tool_calls`), carried in `task` for compute_reward; trace
        lineage (`trace_id` / `turn_index` / `scores` / `raw_prompt`) rides
        along via `init_rollout_args`.
        """
        return make_example(
            prompt_messages=list(example.get("prompt_messages") or []),
            # A row can lack a gold turn (`{}`): keep the shape so the reward
            # reads it defensively instead of KeyError-ing mid-rollout.
            task={"ground_truth": example.get("ground_truth") or {}},
            system_prompt=cls.system_prompt,
            init_rollout_args=example.get("init_rollout_args"),
        )

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """ONE comparative component — the whole training signal, so edit it here.

        The judge scores the completion against the RECORDED turn (content +
        rendered tool calls) on a 0 / 0.5 / 1 match scale. Positive scores only;
        a judge failure scores 0 (never crashes the rollout).
        """
        zeros = {k: 0.0 for k in REWARD_KEYS}
        try:
            completion = extract_completion_text(messages).strip()
            if not completion:
                return zeros
            reference = _gt_text((task or {}).get("ground_truth") or {})
            if not reference:
                # No recorded gold turn on this row → nothing to compare; the
                # validate_probe flags such rows before any GPU is spent.
                return zeros
            try:
                result = await evaluate_single_rubric(
                    rubric=GROUND_TRUTH_MATCH_RUBRIC,
                    question=_last_user_text(messages),
                    ground_truth=reference,
                    response=completion,
                    model_name=self._judge_model,
                    base_url=self._judge_base_url,
                    api_key=self._judge_token_provider(),
                    timeout=self._judge_timeout,
                )
                match = clip01(result.get("score", 0.0))
            except Exception:
                logger.warning("[CustomTraceEnv] match judge failed; scoring 0")
                match = 0.0
            return {"ground_truth_match": match}
        except Exception:
            # A reward bug must not crash the rollout — score 0, but LOG it: a
            # silent all-zero reward is the hardest reward bug to diagnose.
            logger.exception("[CustomTraceEnv] compute_reward failed")
            return zeros

    async def validate_probe(self, eval_dataset):
        """Row-shape + trace-coverage check over the eval rows (no GPU, no
        network): every row must carry a non-empty `prompt_messages` list and an
        assistant `ground_truth` DICT — the `castform data traces` output shape
        this env reads. Catches the old wizard shape (string ground_truth /
        top-level trace_id) before a launch trains on unreadable rows."""
        rows = list(eval_dataset or [])
        if not rows:
            return {"ok": False, "summary": "skipped (no eval rows)"}
        bad: list[int] = []
        trace_ids: set[str] = set()
        tool_call_rows = 0
        for i, row in enumerate(rows):
            pm = row.get("prompt_messages")
            gt = row.get("ground_truth")
            if not (
                isinstance(pm, list)
                and pm
                and isinstance(gt, dict)
                and gt.get("role") == "assistant"
            ):
                bad.append(i)
                continue
            if gt.get("tool_calls"):
                tool_call_rows += 1
            args = row.get("init_rollout_args") or {}
            if args.get("trace_id"):
                trace_ids.add(str(args["trace_id"]))
        if bad:
            return {
                "ok": False,
                "summary": (
                    f"{len(bad)}/{len(rows)} rows off-shape (need non-empty "
                    "prompt_messages + an assistant ground_truth dict; e.g. rows "
                    f"{bad[:5]}) — regenerate with `castform data traces`"
                ),
                "bad_rows": bad,
            }
        return {
            "ok": True,
            "summary": (
                f"{len(rows)} rows / {len(trace_ids)} traces, shape OK "
                f"({tool_call_rows} tool-call gold turns)"
            ),
            "rows": len(rows),
            "traces": len(trace_ids),
            "tool_call_rows": tool_call_rows,
        }


# ── Run config — validate/launch read these so the run reproduces from this file
#    alone (a CLI flag still overrides). See `castform validate/launch --help`.
VALIDATE_CONFIG = {
    "examples": 4,  # a few real rollouts so the scorecard's rewards vary visibly
}

LAUNCH_CONFIG = {
    # Total tokens across the WHOLE rollout. Trace prompts replay every prior
    # turn (tool results included), so this budget scales with TRACE TURN COUNT,
    # not tool calls made here — 8192 covers long-ish transcripts; raise it for
    # long traces (a truncated rollout is dropped from the loss).
    "max_rollout_len": 8192,
    "num_epochs": 2,  # note the plural key; eval tends to peak before the overfit tail
    # "type": "simple",  # GPU pool (gpu4 for 4B / gpu8 for 35B); "simple-cpu" = smoke
}


# ── Runnable entrypoint ──────────────────────────────────────────────────────
# `python main.py [data|validate|launch|all]` drives the whole loop SDK-directly —
# no CLI needed, and this file stays the reproducible record of the run. Stages are
# isolable and skip work whose output already exists (`--force` to redo):
#
#   python main.py data       generate/refresh the datasets (skip if present)
#   python main.py validate   baseline on a real-rollout subset (no GPU)
#   python main.py launch     validate-gate, then train on GPUs (spends credits)
#   python main.py  (or all)  data → validate, then STOP (never auto-launches)
#
# Import-safe: this block runs ONLY under `python main.py`. When the castform CLI
# imports this file it execs under the "main" stem, not "__main__", so nothing here
# fires — `castform validate` / `launch` reuse the SAME SDK calls, no drift.

TRAIN_FILE = "train_dataset.jsonl"
EVAL_FILE = "eval_dataset.jsonl"
ENV_ARGS: dict[str, Any] = {}  # CustomTraceEnv constructor kwargs (none by default)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text("utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_name() -> str:
    return LAUNCH_CONFIG.get("name") or CustomTraceEnv.__name__.lower()


def generate_data(force: bool = False) -> bool:
    """Produce `train_dataset.jsonl` / `eval_dataset.jsonl`.

    Provenance: for the traces template the datasets come FROM your recorded
    agent traces (not inline), so this stage documents how and skips if present:
        export BT_API_KEY=...                              # Braintrust key
        castform data traces --project <name> --dry-run    # confirm prompt + tools
        castform data traces --project <name>              # → train/eval jsonl
    Rows are `{prompt_messages, ground_truth, init_rollout_args}`; paste the
    DETECTED system prompt into CustomTraceEnv.system_prompt, commit the jsonl,
    and re-run. (The committed seed rows are synthetic placeholders that make
    day-one validate work.)
    """
    have = Path(TRAIN_FILE).exists() and Path(EVAL_FILE).exists()
    if have and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    print(
        "data: the traces template builds its datasets from your recorded agent "
        "traces. Run:\n"
        "  export BT_API_KEY=...                              # Braintrust key\n"
        "  castform data traces --project <name> --dry-run    # confirm prompt + tools\n"
        "  castform data traces --project <name>              # -> train/eval jsonl\n"
        "then paste the detected system prompt into CustomTraceEnv.system_prompt,\n"
        "commit the jsonl, and re-run."
    )
    return have


def _print_scorecard(report: Any) -> None:
    """A minimal, SDK-direct scorecard (the CLI's `castform validate` prints a richer
    one). Reads the per-rollout reward dicts straight off the ValidationReport."""
    remote = getattr(report, "remote", None)
    if remote is None:
        print("validate: no remote rollouts ran (nothing to score)")
        return
    for ex in remote.examples:
        if ex.ok and ex.rewards is not None:
            total = sum(
                v
                for v in ex.rewards.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            print(f"  rollout {ex.index}: total={total:.3f}  {ex.rewards}")
        else:
            print(f"  rollout {ex.index}: FAILED — {ex.error}")
    group = getattr(remote, "group_reward", None)
    if group is not None and group.rewards:
        print(f"  group mean: {group.rewards}")
    print(f"validate: {'PASS' if report.ok else 'FAIL'}")


def validate() -> Any:
    """Baseline the env on a real-rollout subset (no GPU). Returns the report."""
    train = _load_jsonl(TRAIN_FILE)
    eval_ds = _load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else []
    report = validate_env(
        env_class=CustomTraceEnv,
        env_args=ENV_ARGS,
        train_dataset=train,
        eval_dataset=eval_ds or None,
        pip_dependencies=CustomTraceEnv.PIP_DEPENDENCIES or None,
        local=False,  # run the remote real-rollout subset (matches `castform validate`)
        remote_examples=VALIDATE_CONFIG.get("examples", 2),
        group_reward_samples=VALIDATE_CONFIG.get("group_samples", 2),
        llm_model=VALIDATE_CONFIG.get("model"),
        max_turns=VALIDATE_CONFIG.get("max_turns", 4),
        max_tool_calls=VALIDATE_CONFIG.get("max_tool_calls", 8),
    )
    _print_scorecard(report)
    # Env probe (row shape + trace coverage) — catches a stale/old-shape dataset
    # before GPU. Runs in-process, best-effort.
    probe = run_validate_probe(CustomTraceEnv, ENV_ARGS, eval_ds)
    if probe is not None:
        print(f"  probe: {probe.get('summary') or probe}")
    return report


def launch(assume_yes: bool = False) -> str | None:
    """Validate-gate, confirm, then upload + launch a GPU training run (spends
    credits). Returns the run id, or None if gated/aborted."""
    report = validate()  # cheap pre-flight — never spend GPU on a broken env
    if report is None or not report.ok:
        print(
            "launch: validate gate FAILED — fix the env before launching.",
            file=sys.stderr,
        )
        return None
    if not assume_yes:
        reply = (
            input(
                f"Launch '{_run_name()}' on GPUs — this spends credits. Continue? [y/N] "
            )
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("launch: aborted.")
            return None
    train = _load_jsonl(TRAIN_FILE)
    eval_ds = _load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else []
    uploaded = upload_training_run(
        env_class=CustomTraceEnv,
        train_dataset=train,
        eval_dataset=eval_ds,
        run_name=_run_name(),
        constructor_args=ENV_ARGS,
        pip_dependencies=CustomTraceEnv.PIP_DEPENDENCIES or None,
    )
    # LAUNCH_CONFIG feeds the launcher, minus the reserved keys: `name` is the run
    # name above; `type` is not a wire arg. The server rejects any unknown key.
    launcher_args = {
        k: v for k, v in LAUNCH_CONFIG.items() if k not in ("type", "name")
    }
    with TrainerClient() as client:
        run_id = client.launch_training_run(
            name=_run_name(),
            launcher_args=launcher_args or None,
            **dataclasses.asdict(uploaded),
        )
    print(f"launch: started run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this env: data → validate → launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
        help="Stage to run (default: all = data → validate, then STOP).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate datasets even if present."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)

    ensure_session()  # best-effort: no-op if a credential resolves

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        report = validate()
        ok = report is not None and report.ok  # non-zero exit on a failed baseline
    if args.stage == "launch":
        ok = launch(assume_yes=args.yes) is not None  # None = gated / aborted / failed
    # `all` / bare `python main.py` STOPS after validate — launch is never automatic
    # (it spends GPU credits); run `python main.py launch` to train.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
