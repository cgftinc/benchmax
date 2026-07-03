# Castform training project

You are driving a reinforcement-learning run with the `castform` CLI. Keep the
loop simple: tailor the seed env, make data, validate a cheap baseline on real
rollouts, then decide whether to iterate or spend GPU on launch.

## Required workflow

Every run follows the same stages:

```bash
castform setup        # 1. scaffold agent skills + project guides
castform data …       # 2. data — write your own jsonl, or generate (rag/traces)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

`castform setup` already wrote a runnable seed `main.py` plus tiny
`train_dataset.jsonl` / `eval_dataset.jsonl`, so `castform validate` should work on
day one. Tailor that seed instead of starting from scratch.

Load the matching skill before each stage:

| Stage | Skill |
|---|---|
| design/edit `main.py` | `.claude/skills/design-environment/SKILL.md` |
| create/upload data | `.claude/skills/generate-data/SKILL.md` |
| run/read `castform validate` | `.claude/skills/verify-environment/SKILL.md` |
| launch GPU training | `.claude/skills/launch-run/SKILL.md` |
| monitor/debug a run | `.claude/skills/view-progress/SKILL.md` |

The validation skill is mandatory before every `castform validate`; it defines the
scorecard and baseline report format. Follow each skill's handoff to the next
stage rather than jumping straight to the command.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install castform        # the CLI is the benchmax package, published as castform
castform login                 # browser sign-in; writes ~/.castform (no API key to manage)
```

For headless / CI, set `PLATFORM_API_KEY` (from `app.castform.dev/account/api-keys`)
instead of `castform login`.

## Project files (the convention `castform validate` / `launch` expect)

- `main.py` — defines your environment: a single `BaseEnv` subclass.
- `train_dataset.jsonl` / `eval_dataset.jsonl` — one JSON object per line; each
  needs at least a `prompt` (and usually a `ground_truth`).

`castform validate`/`launch` import the one `BaseEnv` subclass from `main.py` and
load those two files. Keep that layout.

## The loop

1. **Design the environment** with `design-environment`. `main.py` should expose one
   `BaseEnv` subclass with `list_tools`, `run_tool`, and `compute_reward` (plus
   optional `compute_group_reward`). No tools needed? Return `[]`.
2. **Make the data** with `generate-data`. Keep `train_dataset.jsonl` and
   `eval_dataset.jsonl` disjoint; each row needs the fields your reward reads from
   `task`.
3. **Baseline** with `verify-environment`, then `castform validate`. This is a
   cheap, real-rollout eval. Green means validate passed with sane, varying rewards
   and no reward errors.
4. **Review and decide.** If rewards are flat, too easy, too hard, or suspicious,
   fix reward/data/env and re-validate. If the baseline is faithful enough, launch.
<!-- rag:start -->
   For RAG, also run `castform validate --reward-audit`: green can still hide
   no-answer completions, broken retrieval, brittle citation matching, or a lenient
   judge. Use the retrieval `gold-hit@k` probe before spending GPU.
<!-- rag:end -->
   Keep the user posted while iterating. Each validate runs real remote rollouts
   (~30-60s each), so a full fix-and-re-validate loop can take 10+ minutes.
5. **Launch** with `launch-run`: `castform launch` validates, uploads, then starts
   GPU training. This spends credits.
6. **Monitor** with `view-progress`: use eval curves and stored rollouts, not only
   latest train reward. Train can rise while eval falls.

## Reward functions — get these right

Rewards are the training signal; robust rewards matter more than anything else.

- **Return positive scores.** Negative rewards destabilise training.
- **All reward components are SUMMED** into one scalar per rollout (the trainer
  adds the values of the dict `compute_reward` returns). Scale them so the sum is
  meaningful — a `{"correct": 1.0, "format": 0.1}` weights format at ~10%.
- **Gate secondary bonuses on the primary objective.** If the answer is wrong or
  missing, citation/style/brevity/tool-use rewards should usually pay `0` (or be
  multiplied by correctness). Otherwise the model can learn to bank bonuses while
  failing the task.
- **Prefer comparative rewards** for qualitative/LLM-judge scoring: compare the
  completion against `ground_truth`, or use `compute_group_reward` to *rank*
  completions within a group rather than score them absolutely. Ranking is far
  more stable than an absolute 1–10 judge score. A "finer" absolute LLM judge often
  collapses back to the same 0/1 decisions; use pairwise/listwise ranking when you
  need tie-breaking resolution.
<!-- rag:start -->
- **For RAG, audit the reward before launch** (`castform validate --reward-audit`).
  Extract only a committed `<answer>...</answer>` block (never score the model's
  whole reasoning as an answer), match citations by **id-hash OR title-path** across
  corpus source formats, keep an **ungated** `retrieval_hit` (credit finding gold
  even on a wrong answer), and prefer a deterministic length term over an LLM
  conciseness judge.
<!-- rag:end -->

## Dependencies

The trainer runs your env in an isolated image. Declare every non-benchmax
dependency:

- external PyPI packages → `pip_dependencies=["httpx", ...]`
- local modules → `local_modules=[scoring_utils]` — pass the **imported module
  objects**, not their names as strings.

`castform launch` reads these from `--pip` and bundles `main.py` automatically; if
you call the SDK directly, pass them to `upload_training_run`.

## Gotchas that silently cost you (verified against the trainer)

- **Per-rollout config goes in `__init__`, never at module level.** The sandbox
  **unpickles** your env — module-level code does **not** run there — so anything a
  rollout needs (clients, resolved config, budgets) must be set in `__init__` or it
  won't exist at rollout time.
- **`max_turns` defaults to 4, `max_tool_calls` to 8.** A multi-turn env that
  needs more is silently truncated, and the trainer does **not** consult an env's
  `recommended_max_*` (it never passes the env class to the limit resolver). Set the
  budget explicitly: `castform validate --max-turns N --max-tool-calls N` (both
  settable) and `castform launch --set max_turns=N`. ⚠ At launch `max_tool_calls` is
  **not** a `--set` knob (stays 8), so a tool-heavy env that makes more than 8 tool
  calls is capped in training — keep its per-rollout tool-call count ≤ 8 unless
  `--list-args` shows a higher cap.
- **The launch token budget is `max_rollout_len`** (total tokens across the whole
  rollout, all turns) — **not** `max_response_len`. The server rejects unknown
  arg names. `castform launch --list-args` shows the live, accepted set. Set it
  generously: a rollout that hits the budget is truncated and **dropped from the
  loss**, so too small a value silently wastes rollouts. `castform launch` warns
  before the confirm if the estimated rollout exceeds the budget.
<!-- rag:start -->
  For search envs, budget for `MAX_SEARCH_CALLS × MAX_TOOL_OUTPUT_CHARS`; a large
  per-search result string across many turns can hit the cap even when each single
  tool response looks safe.
<!-- rag:end -->
- **`ok=false` + `remote_ran=true` + `examples=[]` is an infra failure**, not a
  model verdict — a worker/startup problem swallowed the rollouts. Preserve the
  output and **retry**; never report it as a baseline result.
- **Held-out eval is `castform validate --train eval_dataset.jsonl`** (validate the
  eval split as the train set), **NOT** a `--eval` flag.
- **Model-id split:** the validate path names a model like `qwen3.5-4b`; the launch
  training model (`--set model=`) uses the HF-style id like `Qwen/Qwen3.5-4B`. Don't
  cross them.
- **Companion-server envs** (an env that talks to a separate game/sim server, e.g.
  a Showdown-style env) need that server provisioned alongside the rollout — the
  `SkypilotProvisioner` pattern. This is manual today and is the biggest
  env-authoring footgun; see the `verify-environment` skill.

## First-party use-cases

If the task is **search/RAG** (post-training an LLM to use a search tool over a
corpus) or **traces** (training from collected production agent traces), castform
has first-party support — see `castform.com/docs/rag` and `.../traces`.
