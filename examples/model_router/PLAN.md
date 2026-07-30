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
matters is PR narrative discipline and test culture. Validated or shortlisted:
click, pydantic, pytest, flask/jinja, rich (py); fastify, zod (ts);
caddy, cli/cli (go). Skipped: pandas/django/scikit-learn (slow suites),
fastapi (docs/i18n churn in PRs).

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

## Next step

Collect breadth so the variants above have data to run on.
