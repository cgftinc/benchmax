# Model router: plan notes

Deliverable: model router trained on codebase data. Tasks mined from repo
history with codeprobe, rollouts collected via harbor, router trained to
predict per-model outcomes. Blog post + maybe OSS weights.

Out of scope: refined training recipe, first-class mode-routing support,
web demo, deployed router.

## What we have

- Four standalone scripts, all validated: `convert_to_harbor.py` (codeprobe
  task dirs -> harbor tasks), `gate_tasks.py` (oracle/nop task filter),
  `audit_trajectories.py` (cheat detection), `baseline_router.py` (zero-shot
  prompted router). Operational usage lives in README.md.
- click: 20 converted harbor tasks, 19 gated trainable (1 nop_pass,
  ffcc7494, a flaky-test-fix PR whose tests pass at base).
- **Full 19-task x 7-route matrix collected** (~141 scored real-agent trials
  in `harbor_runs/`, all audited clean): claude-code with haiku / sonnet /
  opus / fable, codex with luna / terra / sol. ~$114 notional for the 99
  fill trials. This is the raw dataset.
- Mining validated on click, pytest, fastify, river. river confirms the
  postgres-sidecar tier costs ~1-2s extra boot, nothing more.
- Local auth works for collection without raw API keys (claude-code via
  keychain OAuth, codex via auth.json).

## What to watch out for

- **Label noise is real on frontier tasks.** Two (task, model) pairs flipped
  across attempts: terra/97210f24 (1 then 0) and opus/ec822695 (0 then 1).
  Both were discriminating tasks, i.e. exactly where routing signal lives.
  Easy and unsolvable rows stayed consistent. Handled by the split-asymmetric
  k in the collection protocol below, not by raising k everywhere.
- **Network allowlist is mandatory; instructions are not sufficient.** terra
  once solved a task by fetching upstream and applying the real fix (the
  local leak guard held, the public network defeated it). With an explicit
  Rules section active, terra reformed but opus still attempted a
  `pip download` of a released version. Deterrence by instruction is
  probabilistic; egress control is not.
  Caveat: harbor's enforcement needs Linux nftables, so macOS Docker
  Desktop cannot run allowlist tasks. Collect on Modal/Linux.
- **Do not trust codeprobe's test-file classification.** It is name
  heuristics, and it tagged `src/click/testing.py` (the fix itself) as a
  test file. Overlaying that handed the verifier the solution and caused 2
  of 3 original nop false-positives. The converter now filters to real test
  paths only, never `src/**`.
- **Yield is capped by the scan window.** codeprobe scans only the most
  recent `count*8` merged PRs, so 20 tasks is click's yield at default
  quality gates. More data means more repos, not deeper history, unless we
  patch the window.
- CodeRouterBench is not usable as training data: ~97% repurposed
  single-turn snippet benchmarks (HumanEval+/MBPP+/LeetCode with synthetic
  bugs), wrong distribution for agentic repo-level routing.

## THE main thing: codeprobe's verifier does not use the PR's tests

At verification time codeprobe checks out the base commit, applies only the
agent's diff, and runs the mined pytest command against the *base-state*
test files. Test files the PR added do not exist yet (pytest errors); test
files the PR modified run in their old version (they pass without the fix).
Base-state tests are only a regression signal, which is why codeprobe
weights `test_passed` at just 0.2 in its checklist reward.

Our fix (SWE-bench test-patch style): at convert time snapshot the PR-era
test files (`git show <gt_commit>:<path>`) into the harbor task's `tests/`
dir, and have test.sh overlay them before pytest. The agent never sees
/tests, so `test_passed` can dominate the reward. Validated end to end on
river f2512a7e: ground truth PASSes in 3.2s, base code with PR-era tests
overlaid FAILs on exactly the changed behavior.

Related converter constraint: the leak guard must be a fresh `git init` plus
`git fetch --depth N origin <base_sha>`, then drop the remote. A naive clone
leaves the fix reachable via `git log --all`.

## Repo candidates

Merge style does not matter (the gh API path covers squash merges). What
matters is PR narrative discipline, test culture, and 2026 merge activity
(checked via gh search, 2026-07-30).

Farm roster (decided 2026-07-30): depth-first, most tasks from a small set of
high-quality Python repos, all on the one pip converter path. Target ~300
mined -> ~260-280 gated including click's 19.

| repo    | 2026 merges | --count | note |
|---------|-------------|---------|------|
| pytest  | 332         | 60      | best measured mining quality (q=0.75) |
| pydantic| 298         | 60      | needs --pip extras (dirty-equals etc.) |
| numpy   | 884         | 60      | needs --apt build-essential; meson compile tax; freshest tasks |
| sympy   | 262         | 50      | symbolic-math domain |
| xarray  | 175         | 50      | needs --pip hypothesis |

Bench (if a repo underperforms the gate): pylint (231), dask (146).
No codeprobe patch needed: --count's "(3-20)" is help text only, no
IntRange or downstream clamp; search window = count*8 straight into
gh pr list.

Skipped: pandas/scikit-learn (slow suites), fastapi (docs/i18n churn),
jinja (0 merges in 2026), rich (32), flask (15), requests (chore-heavy).
django (491, postgres-backed) stays skipped for now; targeted verify
commands may void the slow-suite objection - free to revisit via mining.

Later waves (need converter per-repo profiles: base image, install cmd,
verify normalizer, optional sidecar): TS single-package fastify/zod/hono
(fastify mined verify cmds are jest-style but its runner is borp - the
normalizer must fix them); Go + postgres sidecar river/sqlc (river: 5 tasks
mined at q=1.00, sidecar pattern validated). Postgres-only repos are tier-2
cheap (postgres:16-alpine sidecar boots ~1s via compose). TS monorepos
(vitest 679, typescript-eslint 349) and Rust wait: monorepo task-to-package
mapping unvalidated; Rust unit tests live inside impl files so the file-copy
test overlay would clobber agent edits (needs hunk-level overlay).
Ruby/Rust also blocked at the miner: codeprobe supports Python/Go/JS-TS only.

## Collected data: the click matrix (19 tasks x 7 routes, 2026-07-30)

Scope: this dataset is too small to judge whether one routing method beats
another. Its job is to exercise the pipeline end to end; method comparisons
wait for the scaled collection.

mean reward per (task, route); fractional cells are k=2 attempts that
disagreed. The known-tainted terra/b67832c2 cheat trial is excluded (still on
disk in `harbor_runs/2026-07-28__21-20-33/`; the dataset builder must filter
by audit).

```
task        haiku sonnet  opus fable  luna terra   sol
19fd4d6e        0      0     0     0     0     0     0
333c28d7        0      0     0     0     0     0     0
3bb230dc        1      0     0     0     0     0     0
5ee8e312        1      1     1     1     1     1     1
76552ff1        0      1     0     0     0     0     0
831c8f09        0      0     0     0     0     0     0
8c95c73b        0      0     1     1     0     0     0
97210f24        0      0     1     0     0   0.5     0
b67832c2        0      0     0     0     0     0     0
c3535905        0      0     1     1     0     0     0
c4802104        0      0     0     0     0     0     0
c943271a        0      0     0     0     0     0     0
cd9bdd96        1      1     1     1     1     1     1
cfa01eeb        1      0     1     0     0     0     0
d959898d        0      0     1     0     0     0     0
e0d1678e        0      1     0     1     1     1     1
ec822695        0      0   0.5     1     0     0     0
fc6c7c47        1      1     1     1     1     0     0
fe3ad76e        0      0     1     1     1     0     1
```

Per-route: haiku 26.3% at $0.50/task, sonnet 26.3% at $1.01, opus 50.0% at
$1.77, fable 42.1% at $2.74, luna 26.3% at $0.22, terra 18.4% at $0.30, sol
21.1% at $1.09. Oracle 68.4%.

Headroom picture: **6/19 solved by nobody, 2/19 by everybody, 11/19
discriminate** (up from 3/9 at the 8-task sanity check). Notable:

- **Opus is not a superset anymore**: 3bb230dc is solved only by haiku,
  76552ff1 only by sonnet, e0d1678e by five routes but not opus. Oracle
  68.4% vs always-opus 50.0% - a real quality gap opened at breadth, on top
  of the cost play.
- Caution: the single-solver cells are k=1, exactly where label noise lives.
  The test-split k=5 pass re-adjudicates them before anything is claimed.
- Fable is expensive, not better: 42.1% at $2.74 vs opus 50.0% at $1.77.
- luna is the cost floor: matches haiku's solve rate at $0.22/task.

Zero-shot prompted routers cannot capture the discriminating tasks. A sweep
of 7 router models (haiku/sonnet/opus/fable, luna/terra/sol) x 19 tasks
found: weaker routers hedge to the mid-tier model, stronger routers commit
to the cheap one (router IQ mostly buys cost-confidence), nobody routes to
sol, and on fe3ad76e *every* router picked a model that fails it. The
bottleneck is information about per-model quirks, not router reasoning.

## P0-P2 shakedown results (click, 2026-07-30; 4 test tasks - patterns, not
verdicts)

- v0 zero-shot: 0% test solve for ALL 7 router models (19/28 picks hedged to
  sonnet, which solves no test task). The naive router is systematically
  wrong, not just weak.
- P1 profile: lifts 5/7 router models 0% -> 25%; picks shift from
  sonnet-hedging to opus-escalation. Router capability barely matters -
  terra-as-router is the best row (25% at $2.47, cheapest by 2x). Weak
  model + information beats strong model without it.
- P1 answer-sheet diagnostic (routing TRAIN tasks, whose outcomes are in the
  profile): 70% vs oracle 73%, opus 50%. Found every unique solver; the only
  misses are the 4 unsolvable tasks (escalated to opus). Mechanism proven.
  But cost discipline stays mediocre even with answers (6/11 hits picked the
  cheapest solver; all-pass tasks still sent to opus).
- Train 70% vs test 25% is the project gap in one number: with per-task
  answers the prompted router is near-oracle; without them it cannot
  generalize. That is what P3 volume + P4 training are for.
- P2 kNN: 25% at $1.28; neighbour similarities 0.01-0.26 (noise), i.e. it
  degenerates to base rates at this scale, as predicted.

## Metrics (agreed 2026-07-30)

Measure pass rate and what it cost. One table, one row per policy, following
Agent-as-a-Router (arXiv:2606.22902):

| policy | solve rate | $/task | Perf/$ |

Three reference rows frame the router:

- **always-cheapest** (currently luna): the floor.
- **always-frontier** (opus): what people actually do today, and the real
  alternative to our router. The most important comparison.
- **oracle** (best route per task): the ceiling.

Quality efficiency = where the router lands in that band on both axes: how
close to oracle quality, at how much less than frontier spend.

Cost is fully loaded: **router call + whatever it routes to**. A prompted
router burns an LLM call per task, and that goes in the $/task column or we
are not paying for our own method.

Evaluation is on a held-out task split, ordered temporally (train = earlier
PRs, eval = later), which matches deployment and makes trace-search leakage
guards automatic. The split over click's 19 tasks is not yet fixed.

Watch-outs:

- Never sort by Perf/$: it is a ratio, so always-cheapest wins it while
  solving half as much (luna 6.4 vs opus 1.5 on the 19x7 matrix).
- Report the **routable subset separately** (tasks where routes disagree).
  Full-set numbers mostly measure "did you pick the cheap model when it did
  not matter".
- Oracle = max p_hat per task, cheapest route among ties. Credit that
  route's actual p_hat, never 1.0: binarizing the oracle while the
  always-<route> rows average p_hat inflates the ceiling by pure artifact.
  "Cheapest route clearing a bar" is a deployable POLICY, not a ceiling; it
  trades quality for cost and can score below the frontier.
- The band was degenerate at 8 tasks (oracle tied always-frontier), but at
  19 tasks it is not: oracle 68.4% vs opus 50.0%. Re-check at every scale
  jump, since it decides what the post can claim.
- AIQ and the cost-quality convex hull are deferred until we have enough
  tasks for a real lambda sweep; at 4 routes the curve is ~4 points.

## Collection protocol (agreed 2026-07-30)

Attempts per (task, route) differ by split:

- **Train split: k=1.** Maximize task count. A binary outcome is already an
  unbiased sample of p, and a learner pools across tasks, so more tasks beat
  more attempts at equal spend.
- **Test split: k=5.** The oracle is a max over routes, and a max over noisy
  estimates is biased upward, which would inflate the ceiling every published
  number is measured against. Depth on ~20% of tasks is cheap.

Cost of k=1 training data: some flipped labels, landing disproportionately on
discriminating tasks, since that is where the variance is.

## Prior art worth knowing

- **Ramp SWE-Bench** (labs.ramp.com/swebench): 80 private tasks mined from
  Ramp's own production PRs, repos reconstructed at base commits, gold
  patches held out, sandbox-validated fail->pass, plus a "model ladder" to
  discard no-signal tasks. That is our pipeline, built privately. Strong
  validation of the approach. Their public router claim is a single number: 30% cost cut, no
  performance loss (EWMA + Thompson sampling, not a trained predictor).
- **Faros AI** (211 real repo tasks x 6 model+harness routes): the single
  best route was optimal on only 84/211 (40%), and best-vs-worst per task
  averaged 43 points. Best headroom evidence for routing on real repos.
  Their caution: routing alone was not enough.
- **RouterBench**: its own KNN/MLP routers *lose* to the Zero router (a
  weighted coin flip between two models) on MBPP, the coding task, and sit
  inside noise elsewhere. That is the null result we have to beat.

## Router design

Framing: this is a COST play. Match frontier quality for less money, not
beat it on quality. (At 8 tasks the oracle tied always-opus, pure cost play;
the 19-task matrix opened an 18-point quality gap too, but pending k=5
confirmation of the single-solver cells, cost stays the primary framing.)

Cost accounting: record per-trial token counts (input / cache / output)
alongside cost_usd. Harbor already emits all four, so this is about not
discarding them. Dollars are derived and expire when prices change or a
route has no vendor price; tokens do not. Reported tables still use dollars.
Acceptable workaround if we skip per-task cost: a hardcoded $/route, which
is enough to rank routes since the cost ordering is stable.

Rungs:

- v0, done: zero-data prompted router (the 7-router sweep). Informative
  floor: nobody routed to sol, every router missed fe3ad76e.
- v1: prompted router plus a per-route strengths-and-quirks profile derived
  from the collected traces. Directly tests whether v0's failure was
  missing information rather than weak reasoning.
  The profile MUST be built from the train split only, and generated from
  train-split stats rather than hand-written by us, or the comparison is
  dead and we are the router.
- kNN over the request text, averaging the best route among neighbours.
  Baseline only: does a dumb model already work?
- Trained router. Variants below.

We will not run the grid. Start with the easiest rung, keep what is clearly
powerful, drop what is clearly bad, and only ablate properly once the dataset
is big enough for an ablation to mean anything.

### Training variants

**Router input:** the user request plus cheap pre-solve metadata (repo,
language, and similar); never anything derived from the solution (PR diff,
files-changed, patch size), which leaks the answer.

**Base model:** something small and cheap. Exact pick still open.

**Tools available** (in escalation order):

1. Nothing but the router input above.
2. Add grep / search over the codebase.
3. Add search over past traces. Leakage guards, enforced by how the index is
   built rather than by query-time filtering:
   - The retrieval index contains train-split traces only, so no eval task
     can ever be retrieved.
   - During training, exclude the current task from retrieval (leave-one-out).
     Otherwise the router learns to depend on finding its own trace, which
     will not exist at eval time.
   - Use a temporal split (train = earlier PRs, eval = later). Then "no
     future information" is automatic instead of a filter to remember. This
     supersedes the "simple train/test" note in Metrics.

Harbor can execute the router's own tool calls, since router and collection
operate over the same data. That also makes a tool-using router a harbor task,
scored the same way.

**Output form:**

- (a) One pass, predict every route's score.
- (b) Emit the single most efficient route (value judgement baked into the
  weights, so lambda cannot be swept afterwards).
- (c) Per (task, route) inference of score + token count; run once per route.
  Route identity is an input, so this extends to routes not seen in training.

**Token count form:** the router emits the count as text, so the variants are
a plain number vs bucket labels (exponential, Fibonacci, whatever). Try both;
a plain number estimate may be fine. Buckets have two structural advantages:
a single-token label gives a readable distribution to take an expectation
over, and they sidestep the noisy tokenization of long digit strings. Note
that greedy decoding of a number yields roughly the mode, not the mean the
decision rule wants.

**Decision rule:** try both, since they are post-processing over the same
predictions and cost nothing extra:

- argmax of score - lambda * cost; sweeping lambda traces the cost-quality
  curve.
- cheapest-adequate: cheapest route whose score clears a threshold tau. The
  deployable one-sentence story (Ramp's framing). Only meaningful if scores
  are calibrated, so check calibration on the val split.

**Training method:** no tools and a direct prediction -> SFT. Tool use -> RL.
Reward for a routing decision is a lookup into the collected matrix, so
training and sweeping are cheap; only held-out validation costs agent runs.

### Partial-data robustness (open question)

In production there is no complete counterfactual matrix, only whatever route
each task actually ran on. Notes:

- Training survives ragged coverage for (c), needs masked losses for (a), and
  is impossible for (b), whose label requires the counterfactuals.
- Evaluation does not survive: the oracle is uncomputable, and logged routes
  are confounded because whatever picked the route picked it for a reason. A
  deployment would need a randomized slice for unbiased evaluation.
- Measurable cheaply: mask our own matrix to 75 / 50 / 25% of cells, retrain,
  and plot the degradation. No new agent runs needed.

## Phases (agreed 2026-07-30)

Modular: every stage is a standalone file, stages talk through files on disk
(`dataset.jsonl`, `picks.jsonl`, the table). Each phase adds one row to the
same scoreboard - no phase invents its own metric - and ends with an artifact
the supervisor can eyeball plus an explicit go/kill question.

- **P0 - scoreboard on click.** `build_dataset.py` + `scoreboard.py`,
  temporal split frozen. Tests the plumbing and that the split does not put
  all discriminating tasks in one half. Infrastructure, no kill signal.
- **P1 - v1 profile router.** Per-route profile generated from train-split
  stats, injected into the prompted router; router cost included. Question:
  does trace-derived information fix v0's blindness? Kill: lands at random.
  Artifact: one row + a readable picks file.
- **P2 - kNN.** Question: does a trivial model match P1? If yes, prompting
  adds nothing over retrieval.
- **P3 - breadth collection.** The spend gate (explicit approval): N repos,
  k=1 train / k=5 test, Modal with allowlist. Question: does the
  oracle-vs-frontier gap survive scale? Everything before P3 is pipeline
  verification; everything after is claims.
- **P4 - trained SFT (variant c: route-as-input, score + token count).**
  Token head on/off is the ablation; both decision rules and the lambda
  sweep read off the same predictions. Kill: does not beat P1.
- **P5 - tools/RL**, only if P4's failures look like missing information
  rather than missing capacity.

P1/P2 run on click now purely as pipeline shakedown, then re-run unchanged on
the P3 dataset. Method-vs-method judgment happens once, at scale.

## P3 prep decisions (2026-07-30; collection NOT started)

- Route pool: cut fable (dominated: worse than opus at 1.5x price). Six
  routes remain: haiku / sonnet / opus, luna / terra / sol.
- Auth: keep subscription creds shipped via --ae (same plumbing as local
  docker); throttling accepted. OAuth tokens expire mid-farm, so long runs
  re-mint between batches.
- Concurrency: small on purpose - ~2 concurrent claude + 2 concurrent codex.
  At that scale Modal buys no parallelism; its only remaining value is
  allowlist enforcement.
- **Decided: collect on local docker** (Rules + audit only; audit has caught
  2/2 real cheats plus one attempt). Modal's only remaining value at our
  concurrency is allowlist enforcement; skipping it for now.
- Modal validated anyway (2026-07-30, codex terra on cd9bdd96, allowlist
  task): works end to end with copied subscription creds, reward 1.0,
  speed and cost a wash vs local (88s vs 79s warm; agent setup faster on
  Modal, agent execution slower). Available as a drop-in (`-e modal`) if
  enforcement becomes worth it; enforcement-blocks-anything probe never run.
- Distribution: `harbor publish` (task datasets, public/private) and
  `harbor upload` (run results) cover sharing; decide visibility timing
  vs the blog post.
- Converter's Dockerfile template is Python-only (pip install -e .); TS
  repos (fastify/zod) need a second template - keep P3 pure Python.
- Order: farm tasks with codeprobe first (no model spend; fixes true
  per-repo yield and the cost estimate), then the Modal one-task check,
  then collection.

## Farm execution log (2026-07-31)

Mining: all 5 repos mined deep in one night, 280 tasks, ~$33 LLM spend
(numpy 60 / pytest 60 / pydantic 60 / sympy 50 / xarray 50). --count>20
confirmed working (no cap). Stale-clone gotcha: `gh pr list` returns merge
SHAs the local clone lacks; fetch all branch heads before mining (backport
branches too), else those PRs are silently skipped.

Converter went through real hardening (recipes in README): per-repo
--apt/--install-cmd; submodule init in the leak-guard (numpy vendors meson;
submodules are external repos, no leak); empty-overlay tasks skipped as
unverifiable (16 dropped); bare-`pytest` verify commands pointed at the
overlay files. Repo-specific traps that cost gate re-runs: pydantic
editable+extras pip resolution picks a mismatched pydantic-core (two-step
install); pytest needs SETUPTOOLS_SCM_PRETEND_VERSION (no tags in the
leak-guarded checkout); xarray needs h5py explicitly + pytest-mypy-plugins
pinned <4 (per-commit extras/flag drift). Smoke-test recipe per repo BEFORE
gating: build one image, run `<verify_command> --collect-only -q`.

Initial gate yields: pytest 45/60, pydantic 23/54, xarray 37/49, numpy
23/51, and sympy 39/50: 167 passes from 264 converted tasks. The recovery
audit below subsequently promoted 29 more without weakening the verifier.

Collection (decided, running): local docker, 6 routes, k=1, 2 claude + 2
codex lanes. Infra: docker VM memory raised 8->14 GiB after exit-137 OOM
kills corrupted ~25% of trials AND could silently flip gate verdicts
(oracle killed = fail). Orchestration is gap-fill: `fill_missing.py` computes
(task, model) pairs lacking a completed clean trial across base and suffixed
run directories (max 3 attempts each) and safely materializes exact fill
sets. `run_family.sh` loops until nothing is missing, optionally waiting for
an explicit producer sentinel when the farm is still growing. `run_gates.sh`
accepts repository arguments and delegates idempotent farm promotion to
`gate_tasks.py --promote-to`, including its manifest merge. All state is
recomputed from disk, so restarts are free. Auth: claude OAuth token
read fresh from the macOS keychain each round (they expire; harbor masks
secrets in persisted config.json, so configs are NOT an auth source);
codex just needs CODEX_FORCE_AUTH_JSON=1 (harness copies local creds).

Data-integrity gotcha found during collection: Harbor still runs the verifier
after an agent/API exception, so an auth or quota failure on an untouched
checkout can have a numeric reward of 0. These are infrastructure, not model
failures. `fill_missing.py` now requires a clean numeric result before marking
a cell done, and `build_dataset.py` independently drops exception-bearing
trials. `AgentTimeoutError` is retained because the fixed agent time budget is
part of the benchmark. Infrastructure attempts still count toward the retry
cap so an exhausted subscription cannot loop forever; retired cells remain
absent and can be reopened with a higher cap after access is restored.

After the Claude subscription expired, 818 quota/auth trial directories
(`ApiRateLimitError` or zero-token `UnknownApiError`) were moved out of the
farm runs to Trash. They contained no usable model outcomes and would only
consume the retry cap. This reopens 83 Opus, 111 Sonnet, and 155 Haiku cells
for clean backfill after Claude access is restored.

Provisional prototype set (2026-08-01): all three Codex routes have clean
coverage on all 167 gated tasks. Opus has 84 numeric outcomes, but one
(`27aaf685`) is audit-tainted because it downloaded the target Pydantic
package from PyPI, leaving an audited rectangular set of **83 tasks x 4
routes = 332 unique cells** (Opus / Sol / Terra / Luna). The artifact is
`dataset_opus_codex_83.jsonl`; its repo mix is pytest 41 / xarray 27 /
pydantic 15, so it has no numpy or sympy coverage and is selection-biased.
Use it for pipeline and router prototyping only until Claude backfill lands.
On this set: Opus solves 41/83, Sol 38/83, Terra 37/83, Luna 37/83, and the
per-task oracle solves 44/83. Only 10/83 tasks discriminate among routes.
The temporal test slice has just 2 discriminating tasks, making current
P1/P2 test scores too small for method claims.

Prototype split decision: use per-repository chronology rather than one
global timestamp cut. Each repo contributes its older half to train and
newer half to test. Because all three repo counts are odd, the middle task
from each repo goes to train. This uses every task and produces 43 train / 40
test: train is pydantic 8 / pytest 21 / xarray 14, test is pydantic 7 /
pytest 20 / xarray 13. Pass `--split-strategy repo-temporal` consistently to
`profile_router.py`, `knn_router.py`, and `scoreboard.py`.

P0/P1 prototype rerun with `gpt-5.6-terra` as the router model on the 40-task
repo-temporal test half: the naive prior-only router solved 21/40 (52.5%) at
$34.97 selected-agent cost; the train-profile/stats router solved 20/40
(50.0%) at $6.43. Router-call cost is unavailable from subscription-backed
`codex exec`, so these totals include the selected coding agent only. Naive
picks were Terra 16 / Sol 16 / Luna 7 / Opus 1; stats picks were Luna 26 /
Terra 14. Always-Luna solves 21/40 for $2.55, so both prompted routers are
currently dominated. On the six route-disagreement test tasks, naive solves
2 and stats solves 1; treat this as a prototype diagnostic, not a method
claim.

Sanity trials passed: codex/terra solved a pydantic task (r=1.0, $0.13);
claude/sonnet made a genuine near-miss attempt on a pytest task (244
passed, 1 failed on the PR's behavior test, $0.52).

## Task-pool recovery and expansion (2026-08-01)

Re-audited all 97 post-conversion gate rejections against their raw oracle
logs. The main recoverable causes were verifier commands that passed fixture
or source files directly to pytest, dependency drift in xarray, and NumPy
tasks built with a newer pytest than their base commit's exact requirements
pin. Pydantic also needed Hypothesis, but most remaining Pydantic failures
depend on a ground-truth-era compiled `pydantic-core`; forcing those through
without rebuilding the core would make the benchmark environment less
faithful, not more useful. The two SymPy errors were a nondiscriminating task
and a 608-module verifier timeout, so neither was promoted.

Recovery result after a k=1 screen and independent k=3 confirmation:

| repo | newly recovered |
|---|---:|
| pytest | 9 |
| pydantic | 2 |
| xarray | 6 |
| numpy | 12 |
| sympy | 0 |
| **total** | **29** |

The gated pool was therefore **196 tasks**, up from 167. All 29 were copied
into `harbor_tasks/farm`; the farm received its own 196-entry pass manifest,
so it is directly consumable by `build_dataset.py`. Future gates can use
`--promote-to harbor_tasks/farm`, which copies only passes and merges their
verdicts idempotently.

Expansion wave result (2026-08-02): duplicate-safe depth mining added 231 raw
tasks with zero overlap against the original task IDs: pytest 71, xarray 80,
and SymPy 80. Codex enriched all 71 pytest and 80 xarray tasks plus the newest
50 SymPy tasks. Conversion and k=1 screening retained 33 pytest, 46 xarray,
and 40 SymPy candidates. Thirty of the SymPy candidates were selected for the
k=3 target, leaving ten screened candidates as reserve. Independent k=3
confirmation promoted 33 pytest, 46 xarray, and 30 SymPy tasks: **109 new
tasks**, bringing the self-contained farm to **305 pass-only tasks**. Current
composition is pytest 87 / xarray 89 / SymPy 69 / NumPy 35 / Pydantic 25.

The wave also replaced the hand-copied install commands with strict TOML
environment profiles for the five repositories. `convert_to_harbor.py
--profile <repo>` validates the profile schema, records profile provenance,
resolves `{pytest_version}` from each historical base commit, and still allows
explicit CLI overrides for experiments. Real conversion tests regenerated one
task per repository and matched the already k=3-gated reference artifacts.

## Next step

Complete Sol/Terra/Luna collection over the 305-task farm, retrying only clean
infrastructure gaps, then rebuild a rectangular Codex matrix. After Claude
access resets, backfill the missing Claude cells and revisit test-split k=5
runs and a proper compiled-core Pydantic environment if its marginal yield is
still worth the setup cost.
