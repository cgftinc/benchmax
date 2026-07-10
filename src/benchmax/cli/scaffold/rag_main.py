"""RAG search environment (written by `castform setup --template rag`).

Post-trains a model to answer questions by SEARCHING a corpus and citing its
sources. The search tool, system prompt, and dataset wiring come from `SearchEnv`
(the convenience base); THIS file spells out the reward inline in `compute_reward`
— the reward is the whole training signal, so it's here to read and edit, not
buried in the library. The heavy pieces (the correctness judge, the citation
matcher) stay as named helpers imported from the lib so this file stays short.

The reward is AUDITED (see `compute_reward`): correctness gates every secondary
component, `retrieval_hit` is UNGATED so citing gold is rewarded even on a wrong
answer, citations match by id-hash OR title-path, and brevity is a deterministic
length term (no second LLM call). `validate_probe` measures retrieval gold-hit@k
over the eval rows — a pre-GPU check the cheap rollout can't give you.

The whole run is reproducible from this file: the reward is above, and the
`VALIDATE_CONFIG` / `LAUNCH_CONFIG` blocks bake in the rollout budgets so
`castform validate` / `castform launch` need no extra flags (a CLI flag still
overrides). Audit the reward on real transcripts before a serious launch:
`castform validate --reward-audit`.

Data: `train_dataset.jsonl` / `eval_dataset.jsonl` with `{question, answer,
reference_chunks}` rows — generate them from your corpus with
`castform data qa-gen --corpus-name <CORPUS_NAME> --fast`. Build the corpus first
with `castform corpus ingest <folder> --name <CORPUS_NAME>`.

Footgun: do NOT pass `benchmax` in `local_modules` at launch — it re-imports the
package by value and breaks `issubclass(env, BaseEnv)`. benchmax is already on the
trainer image; only your own local modules need bundling.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

from benchmax import config
from benchmax.envs.postgres_search.search_env import (
    ANSWER_LENGTH_CAP,
    CORRECTNESS_RUBRIC,
    SearchEnv,
    canonicalize_source_id_loose,
    extract_answer_block,
    score_citations,
)
from benchmax.envs.reward_helpers import clip01, extract_completion_text
from benchmax.envs.types import Messages
from benchmax.platform.client import TrainerClient
from benchmax.platform.login import ensure_session
from benchmax.platform.training_run import upload_training_run
from benchmax.platform.validation import run_validate_probe, validate_env
from benchmax.rag.corpus.postgres.client import CorpusClient
from benchmax.rag.corpus.postgres.search import PostgresSearch
from benchmax.rubrics.rubric import evaluate_single_rubric

# The corpus to search. It must already exist on the Corpora backend — create it
# with `castform corpus ingest <folder> --name <CORPUS_NAME>`. Resolved by name at
# rollout time. (An existing name resolves without prompting; a non-existent name
# can block on an interactive corpus-cap prompt — ingest it first.)
CORPUS_NAME = "my-corpus"

# Judge model for the correctness reward component (LLM, no GPU).
JUDGE_MODEL = "gpt-5.4-mini"

# Search-call budget per rollout; the system prompt advertises this same number.
# Keep it <= 8 unless `castform launch --list-args` shows a higher launch tool-call
# cap. Each search = one turn + one tool call; the final answer is one extra TURN.
# The rollout budget below (VALIDATE_CONFIG / LAUNCH_CONFIG) is sized off this.
MAX_SEARCH_CALLS = 6

# Cap each search tool result so N searches don't blow the rollout token budget
# (~4 chars/token; 6 searches × 8000 ≈ 12k tokens, under a 16384 max_rollout_len).
MAX_TOOL_OUTPUT_CHARS = 8000

# Deterministic brevity cap (imported above; lib default 600): an answer at/above
# ANSWER_LENGTH_CAP chars earns no length bonus; shorter (still-correct) answers earn
# more. Replaces the LLM conciseness judge (which fired on ~2% of rollouts) with
# dense signal on every correct rollout.

# validate_probe: retrieval gold-hit@k over eval rows — k, and a cap on live searches.
PROBE_TOP_K = 10
PROBE_MAX_ROWS = 25

# ── Reward weights (all SUMMED into one scalar per rollout) ──────────────────
# `answer_correctness` is the GATE: every component EXCEPT `retrieval_hit` is
# × correctness, so brevity/precision can't be earned on a wrong answer.
# `retrieval_hit` is UNGATED — citing a gold source is rewarded even when the final
# answer is wrong (the audit found gating it killed the search-learning signal).
# Scale so correctness dominates.
W_CORRECTNESS = 1.0
W_RETRIEVAL_HIT = 0.3
W_CITATION_PRECISION = 0.3
W_LENGTH = 0.2

REWARD_KEYS = (
    "answer_correctness",
    "retrieval_hit",
    "citation_precision",
    "answer_length",
)

logger = logging.getLogger(__name__)


def _meta_file_id(metadata: dict[str, Any] | None) -> str:
    """Document-level id from a chunk's metadata — the key both the citation match
    and the gold-hit probe key off (mirrors the lib's reference-id extraction)."""
    md = metadata or {}
    return str(md.get("file") or md.get("file_path") or "").strip()


class CustomSearchEnv(SearchEnv):
    # The gate component for `castform validate --reward-audit` — secondaries are
    # judged for redundancy against it (not a hardcoded 'correctness' key).
    PRIMARY_REWARD_KEY = "answer_correctness"

    # Extra pip deps the rollout sandbox needs — `validate`/`launch` read this and
    # install it (the sandbox bundles only main.py + benchmax). Empty for the default
    # Postgres corpus; when you swap `search=` to a provider client, list its SDK
    # here (e.g. ["chromadb>=1.0.0", "snowballstemmer>=2.2.0"]) — or pass
    # `--provider <name>` to validate/launch and skip the bookkeeping.
    PIP_DEPENDENCIES: list[str] = []

    # Rendered once at class-definition so the dataset/prompt preprocessors read
    # the resolved value via `cls` (keep MAX_SEARCH_CALLS in sync with __init__).
    # To change the prompt, override SYSTEM_PROMPT_TEMPLATE (see SearchEnv).
    system_prompt = SearchEnv.render_system_prompt(
        corpus_description=f"the '{CORPUS_NAME}' corpus",
        max_search_calls=MAX_SEARCH_CALLS,
    )

    def __init__(self, **kwargs):
        super().__init__(
            # PostgresSearch is pickle-safe; the bearer is resolved per request,
            # nothing credential-shaped is frozen into the bundled env.
            search=PostgresSearch(CORPUS_NAME, base_url=config.platform_url()),
            judge_base_url=config.llm_url(),
            judge_model=JUDGE_MODEL,
            max_search_calls=MAX_SEARCH_CALLS,
            **kwargs,
        )

    @staticmethod
    def _truncate_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS, **_kw):
        # Scale the per-result char cap down so MAX_SEARCH_CALLS searches fit the
        # rollout token budget (the base default is larger; see MAX_TOOL_OUTPUT_CHARS).
        return SearchEnv._truncate_tool_output(text, max_chars=max_chars)

    def _canonicalize_id(self, source_id: str) -> str:
        """Match citations by id-hash OR title-path (the lib default — lowercase,
        drop any directory prefix and file extension, so 'docs/Geography.md',
        'geography.md' and a bare 'geography' canonicalize alike). Kept here as
        the seam to swap in a corpus-specific matcher."""
        return canonicalize_source_id_loose(source_id)

    def estimate_rollout_tokens(self) -> int:
        # Worst case: system prompt + N searches × the per-result char cap + the
        # answer, at ~4 chars/token. `castform launch` warns if this exceeds
        # LAUNCH_CONFIG's max_rollout_len (raise the budget or shrink the context).
        chars = (
            len(self.system_prompt or "")
            + MAX_SEARCH_CALLS * MAX_TOOL_OUTPUT_CHARS
            + ANSWER_LENGTH_CAP
        )
        return chars // 4

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """The reward — the whole training signal. Edit freely; audit with
        `castform validate --reward-audit` before launching.

        `answer_correctness` (0/1 from the judge rubric) is the GATE: every component
        EXCEPT `retrieval_hit` is × correctness, so brevity/precision can't be earned
        on a wrong answer. `retrieval_hit` is UNGATED — citing a gold source is
        rewarded even when the answer is wrong. Return positive scores only.
        """
        zeros = {k: 0.0 for k in REWARD_KEYS}
        try:
            # Strict extraction: no committed <answer> → "" → scores 0 (the model's
            # reasoning is never scored as the answer).
            answer = extract_answer_block(extract_completion_text(messages))
            if not answer.strip():
                return zeros
            t = task or {}
            reference_chunks = t.get("reference_chunks", [])

            # Correctness judge — ONE rubric call (no separate conciseness judge;
            # brevity is the deterministic answer_length term). A judge failure means
            # "not verified correct" (0) and must NOT zero the ungated retrieval
            # signal below, so it's caught locally.
            try:
                result = await evaluate_single_rubric(
                    rubric=CORRECTNESS_RUBRIC,
                    question=str(t.get("question") or t.get("prompt") or ""),
                    ground_truth=str(t.get("ground_truth") or ""),
                    response=answer,
                    model_name=self._judge_model,
                    base_url=self._judge_base_url,
                    api_key=self._judge_token_provider(),
                    timeout=self._judge_timeout,
                )
                correctness = clip01(
                    result.get("score", 0.0)
                )  # 0 / 1 (rubric score_map)
            except Exception:
                logger.warning("[CustomSearchEnv] correctness judge failed; scoring 0")
                correctness = 0.0

            # Citations: id-hash OR title-path match via _canonicalize_id (above).
            recall, precision = score_citations(
                answer, reference_chunks, canonicalize=self._canonicalize_id
            )
            # Deterministic brevity: shorter (still-correct) answers score higher.
            length_score = clip01(1.0 - len(answer) / ANSWER_LENGTH_CAP)

            return {
                "answer_correctness": W_CORRECTNESS * correctness,
                "retrieval_hit": W_RETRIEVAL_HIT * recall,  # UNGATED
                "citation_precision": W_CITATION_PRECISION * precision * correctness,
                "answer_length": W_LENGTH * length_score * correctness,
            }
        except Exception:
            # A reward bug must not crash the rollout — score 0, but LOG it: a
            # silent all-zero reward is the hardest reward bug to diagnose.
            logger.exception("[CustomSearchEnv] compute_reward failed")
            return zeros

    async def validate_probe(self, eval_dataset):
        """Retrieval gold-hit@k over the eval rows — proves the corpus actually
        surfaces the gold sources BEFORE spending GPU (a green rollout doesn't).

        Read-only + non-interactive: resolves the corpus by NAME via `list_corpora`
        and searches it; it NEVER creates a corpus or blocks on stdin (unlike the
        rollout search path). Skips gracefully when the corpus isn't ingested or the
        eval rows carry no gold `reference_chunks`."""
        rows = [r for r in (eval_dataset or []) if r.get("reference_chunks")]
        if not rows:
            return {
                "ok": False,
                "summary": "skipped (no eval rows with reference_chunks)",
            }
        client = CorpusClient(base_url=config.platform_url())
        # list_corpora() is synchronous — run it off the event loop so the outer probe
        # deadline (run_validate_probe's wait_for) can actually enforce its timeout.
        corpora = await asyncio.to_thread(client.list_corpora)
        corpus = next((c for c in corpora if c.name == CORPUS_NAME), None)
        if corpus is None:
            return {
                "ok": False,
                "summary": f"skipped (corpus {CORPUS_NAME!r} not ingested)",
            }

        rows = rows[:PROBE_MAX_ROWS]
        hits = 0
        for row in rows:
            gold = {_meta_file_id(rc.get("metadata")) for rc in row["reference_chunks"]}
            gold.discard("")
            if not gold:
                continue
            res = await client.asearch(
                corpus_id=corpus.id,
                query=str(row.get("question") or row.get("prompt") or ""),
                limit=PROBE_TOP_K,
            )
            retrieved = {
                _meta_file_id(getattr(c, "metadata", None)) for c in res.results
            }
            if gold & retrieved:
                hits += 1
        rate = hits / len(rows)
        return {
            "ok": rate > 0,
            "summary": f"gold-hit@{PROBE_TOP_K} = {rate:.2f} ({hits}/{len(rows)} rows)",
            "gold_hit_at_k": rate,
            "k": PROBE_TOP_K,
        }


# ── Run config — validate/launch read these so the run reproduces from this file
#    alone (a CLI flag still overrides). See `castform validate/launch --help`.

# A search env needs a turn/tool budget above the 4/8 default, or the rollout is
# truncated below MAX_SEARCH_CALLS. N searches → N+1 turns (a final answer turn) and
# N tool calls.
VALIDATE_CONFIG = {
    "max_turns": MAX_SEARCH_CALLS + 1,
    "max_tool_calls": MAX_SEARCH_CALLS,
    "examples": 6,  # a few real rollouts make --reward-audit's per-component read sharper
}

# The trainer ignores an env's recommended_max_*, so bake the budget here. NOTE
# max_tool_calls is NOT a launch knob (stays 8) — keep MAX_SEARCH_CALLS <= 8. The
# accepted arg set is `castform launch --list-args`; an unknown key here is skipped
# with a warning.
LAUNCH_CONFIG = {
    "max_turns": MAX_SEARCH_CALLS + 1,
    # Total tokens across the WHOLE rollout (all turns). MODEL-AWARE: 16384 is the
    # gemma-26B OOM ceiling; a dense 4B is content-hungry, so RAISE it (e.g. 24576).
    # Lower it for 26B. A rollout that hits the cap is truncated and DROPPED from loss.
    "max_rollout_len": 16384,
    "num_epochs": 2,  # eval peaks early then regresses; 2 keeps the best-eval region
    # Prefer the BEST-eval checkpoint over the last (eval regresses in the overfit
    # tail). Set it via the launcher when the server exposes the knob — see
    # `castform launch --list-args`; watch `castform runs scalars --mode eval`.
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
ENV_ARGS: dict[str, Any] = {}  # CustomSearchEnv constructor kwargs (none by default)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text("utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_name() -> str:
    return LAUNCH_CONFIG.get("name") or CustomSearchEnv.__name__.lower()


def generate_data(force: bool = False) -> bool:
    """Produce `train_dataset.jsonl` / `eval_dataset.jsonl`.

    Provenance: for the rag template the datasets are generated FROM your ingested
    corpus (not inline), so this stage documents how and skips if they're present:
        castform corpus ingest <folder> --name my-corpus
        castform data qa-gen --corpus-name my-corpus --fast
    Commit the resulting jsonl so the run reproduces from this repo. Re-generate with
    --force. (For a from-scratch env, replace this with the inline gen code.)
    """
    have = Path(TRAIN_FILE).exists() and Path(EVAL_FILE).exists()
    if have and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    print(
        "data: the rag template builds its datasets from your corpus. Run:\n"
        "  castform corpus ingest <folder> --name my-corpus\n"
        "  castform data qa-gen --corpus-name my-corpus --fast\n"
        "then commit the jsonl and re-run."
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
        env_class=CustomSearchEnv,
        env_args=ENV_ARGS,
        train_dataset=train,
        eval_dataset=eval_ds or None,
        pip_dependencies=CustomSearchEnv.PIP_DEPENDENCIES or None,
        local=False,  # run the remote real-rollout subset (matches `castform validate`)
        remote_examples=VALIDATE_CONFIG.get("examples", 2),
        group_reward_samples=VALIDATE_CONFIG.get("group_samples", 2),
        llm_model=VALIDATE_CONFIG.get("model"),
        max_turns=VALIDATE_CONFIG.get("max_turns", 4),
        max_tool_calls=VALIDATE_CONFIG.get("max_tool_calls", 8),
    )
    _print_scorecard(report)
    # Env probe (retrieval gold-hit@k) — proves what the cheap rollout can't. Runs
    # in-process, best-effort; None (nothing printed) unless validate_probe is set.
    probe = run_validate_probe(CustomSearchEnv, ENV_ARGS, eval_ds)
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
        env_class=CustomSearchEnv,
        train_dataset=train,
        eval_dataset=eval_ds,
        run_name=_run_name(),
        constructor_args=ENV_ARGS,
        pip_dependencies=CustomSearchEnv.PIP_DEPENDENCIES or None,
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
