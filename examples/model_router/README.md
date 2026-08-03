# model_router

Model router for coding agents, trained on tasks mined from a codebase's own
history. Design decisions, findings, phasing, and watch-outs live in
[PLAN.md](PLAN.md); this file is the how-to-run guide.

Every stage is a standalone script; stages communicate through files on disk:

```
mine -> convert -> gate -> collect -> build_dataset -> router rung -> scoreboard
        (tasks)  (manifest) (trials)  (dataset.jsonl)  (picks.jsonl)   (table)
```

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python 'codeprobe>=0.13,<0.14' 'harbor>=0.18,<0.19'
gh auth status    # mining needs an authenticated gh for PR narratives
```

## Steps (click as the worked example)

**1. Mine tasks from repo history** (full clone; shallow can't be mined):

```bash
git clone https://github.com/pallets/click
(cd click && ../.venv/bin/codeprobe mine . --goal quality --count 20 --no-interactive)
```

To use the saved Codex CLI account for instruction enrichment instead of
Codeprobe's Anthropic/OpenAI API backends, mine with `--no-llm` and run the
tracked postprocessor before conversion:

```bash
# One-task preview: writes codex_enrichment/<id>.json, changes no task files.
python enrich_with_codex.py numpy/.codeprobe/tasks \
    --task-id 5e403264 --out-dir codex_enrichment

# After inspecting a preview, enrich every still-raw task atomically.
python enrich_with_codex.py numpy/.codeprobe/tasks \
    --all --in-place --out-dir codex_enrichment

# Resume an exact mined-ID batch without repeating completed tasks.
python enrich_with_codex.py numpy/.codeprobe/tasks \
    --task-id <id-a> --task-id <id-b> --in-place --resume \
    --out-dir codex_enrichment
```

The postprocessor invokes `codex exec` in an isolated read-only temporary
directory with a strict output schema, uses saved Codex authentication, and
records model, reasoning effort, runtime, token usage, prompt hash, and the
rendered instruction in each preview artifact. Calls batch five tasks by
default to amortize agent-input overhead; a failed multi-task batch retries
each task singly. `--all` skips tasks already stamped `llm` or `codex`, and
`--resume` provides the same behavior for an explicit `--task-id` list. Set
`CODEX_BIN`, `CODEPROBE_CODEX_MODEL`, or pass `--model` when a non-interactive
shell needs an explicit binary/model.

**2. Convert to harbor tasks** (leak-guarded checkout, PR-era test overlay):

```bash
python convert_to_harbor.py numpy harbor_tasks/numpy \
    --profile numpy \
    --agent-network public    # 'allowlist' (default) on Linux/Modal;
                              # macOS Docker can't enforce network policies
```

The tracked profiles under `environment_profiles/` encode the validated base
image, repository URL, OS packages, install commands, dependency pins, fetch
depth, and timeouts for pytest, Pydantic, NumPy, SymPy, and xarray. A profile
name resolves next to this script; an explicit TOML path also works. Generated
`task.toml` files retain the profile name as provenance.

```bash
for repo in pytest pydantic numpy sympy xarray; do
    python convert_to_harbor.py "$repo" "harbor_tasks/$repo" \
        --profile "$repo" --agent-network public
done
```

Explicit `--repo-url`, `--base-image`, `--fetch-depth`, `--apt`,
`--install-cmd`, or `--post-install-cmd` values override the selected profile,
so a one-off experiment does not require editing the stable recipe.
`{pytest_version}` in either install command resolves from each task's base
commit `uv.lock` or exact requirements pin rather than today's package index.

Smoke-test a new repo's recipe before gating: build one task image, then
`docker run --rm <img> sh -c '<verify_command> --collect-only -q'`.

**3. Gate tasks** (oracle must pass, nop must fail; writes `manifest.json`):

```bash
python gate_tasks.py harbor_tasks/click -k 3 \
    --promote-to harbor_tasks/farm
```

If a Harbor or Docker outage leaves `error` verdicts, rerun only the missing
repetitions while preserving completed oracle/nop rewards:

```bash
python gate_tasks.py harbor_tasks/click -k 3 \
    --resume-errors \
    --promote-to harbor_tasks/farm
```

`--promote-to` copies only newly passing tasks and merges their pass records
into the farm's own `manifest.json`. It is idempotent, so resumed or repeated
gate batches can safely target the same farm and the farm can be passed
directly to `build_dataset.py`.

**4. Collect outcomes** (one run per route; repeat over models):

```bash
.venv/bin/harbor run -p harbor_tasks/click -a claude-code -m claude-opus-5 \
    -n 1 --jobs-dir harbor_runs/<label> \
    --ae CLAUDE_CODE_OAUTH_TOKEN=... --ae CLAUDE_FORCE_OAUTH=1
# codex routes: -a codex -m gpt-5.6-sol --ae CODEX_FORCE_AUTH_JSON=1 (needs -m)
```

For resumable collection over the shared farm, use the tracked family runner:

```bash
./run_family.sh codex
MODELS='gpt-5.6-sol gpt-5.6-terra' CONCURRENCY=2 \
    MAX_ATTEMPTS=3 ./run_family.sh codex
```

It scans both the base and suffixed `harbor_runs/farm-<model>*` directories,
accepts only clean numeric outcomes (while retaining `AgentTimeoutError` as a
real benchmark result), and rebuilds each fill directory from the exact
remaining gaps. Set `WAIT_FOR=logs/gates.done` only when a separate producer
will deliberately grow the farm and create that sentinel; otherwise the
runner exits as soon as its current farm is complete.

To gate and promote one or more converted repositories sequentially:

```bash
./run_gates.sh pytest xarray sympy
```

`K`, `CONCURRENCY`, `FARM`, and `JOBS_ROOT` are optional environment
overrides. Promotion goes through `gate_tasks.py --promote-to`, so the farm
manifest and task directories are updated together instead of by a separate
hand-copy step.

**5. Build the dataset** (joins task metadata + gate verdicts, drops
audit-tainted trials, stamps merge dates for the temporal split):

```bash
python build_dataset.py harbor_runs --tasks-dir harbor_tasks/click --out dataset.jsonl
```

For a rectangular prototype matrix, repeat `--require-route MODEL` for every
route that must be present. Tasks missing any required clean, audited trial
are excluded explicitly instead of being left for downstream tools to drop.

**6. Run a router rung** (each rung emits a picks file):

```bash
python baseline_router.py harbor_tasks/click --router-model claude-sonnet-4-6  # v0 zero-shot
python profile_router.py dataset.jsonl   # P1: prompt + train-split track record
python knn_router.py dataset.jsonl -k 3  # P2: TF-IDF nearest-neighbour, no LLM
```

`baseline_router.py` also accepts a dataset JSONL, discovers its exact route
pool, and routes the selected split. Set `CODEX_BIN` when a non-interactive
runner's PATH differs from the interactive shell (for example, an fnm-managed
npm installation would otherwise be shadowed by a stale Homebrew binary).

For the provisional multi-repo dataset, use `--split-strategy repo-temporal`
on the router and scoreboard commands. It splits each repository into older
train and newer test halves, assigning the extra task to training when that
repository has an odd task count. This prevents cross-time leakage within a
repository without letting different repository date ranges dominate the
split.

**7. Score everything on the one shared table:**

```bash
python scoreboard.py dataset.jsonl --split test --picks <picks.jsonl>
```

`audit_trajectories.py <jobs_dir> --package click` scans any run for cheat
commands on demand; `build_dataset.py` applies the same audit automatically.

## Files

| File | Role |
|---|---|
| `convert_to_harbor.py` | Data prep. Leak-guarded Dockerfile (fetch base commit only), PR-test overlay verifier (codeprobe's own verifier never restores PR tests), oracle `solve.sh` = PR diff, network policy + anti-cheat rules in every task |
| `environment_profiles/*.toml` | Validated, declarative per-repository build/test recipes consumed by `convert_to_harbor.py --profile` |
| `enrich_with_codex.py` | Optional post-mining instruction enrichment through the saved Codex CLI account; schema-validated previews by default, atomic task updates only with `--in-place` |
| `gate_tasks.py` | Task filter. Keeps a task only if oracle k runs all pass and nop k runs all fail; manifest.json records verdicts |
| `fill_missing.py` | Clean-coverage scanner across repeated Harbor waves; optionally materializes the exact remaining task set for a model |
| `run_family.sh` | Restart-safe Codex/Claude collection loop over the shared farm, with configurable models, concurrency, retry cap, and optional producer sentinel |
| `run_gates.sh` | Sequential multi-repository gate runner using the gate's idempotent promotion path |
| `audit_trajectories.py` | Cheat detector. Scans executed commands (not raw text) for upstream fetches / package-index installs of the target repo |
| `build_dataset.py` | Flattens trials into `dataset.jsonl`: joins task.toml metadata + manifest verdicts, drops tainted/ungated trials loudly, resolves merge dates, optionally requires a complete route set |
| `baseline_router.py` | Zero-shot prompted router (v0), adapted from Agent-as-a-Router (arXiv:2606.22902); accepts a task directory or dataset and builds a prior-only prompt for its exact route pool |
| `profile_router.py` | P1: prompted router + per-route track record generated from the train split |
| `knn_router.py` | P2: TF-IDF kNN over instructions; no LLM, the trivial baseline |
| `scoreboard.py` | The one policy table: always-\<route\>, random, router (from picks), oracle ceiling; global or per-repo temporal split; routable subset reported separately |

## Untracked working state (regenerable or dataset-store material)

- `pytest/ pydantic/ xarray/ numpy/ sympy/` - 511 mined tasks across
  full repo clones (`.codeprobe/tasks` inside)
- `harbor_tasks/farm/` - 305 k=3-gated tasks plus its self-contained pass
  manifest
- `harbor_tasks/recover-*/` - reproducible k=1 screens and k=3 recovery
  confirmations
- `harbor_runs/` - raw gate and model-attempt trial directories
- `dataset*.jsonl` / `router_sweep/` - flattened prototype datasets and picks
- `logs/` - run logs

## Known gotchas (details in PLAN.md)

- `harbor run` takes ONE `-p` path; use a dataset dir for subsets, and give
  concurrent jobs separate `--jobs-dir`s (timestamped names collide).
- Network allowlist enforcement requires Linux (Modal maps it natively to
  sandbox egress params); macOS Docker Desktop fails harbor's kernel probe,
  so local runs are `--agent-network public` + Rules + audit only.
- codeprobe mining scans only the most recent `count*8` merged PRs.
  `--count`'s "(3-20)" is help text only - larger counts work and scan
  proportionally deeper history.
- Instruction-only anti-cheat deters some models, not all (opus attempted
  `pip download` despite the Rules); the allowlist is the real defense.
