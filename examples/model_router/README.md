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

**2. Convert to harbor tasks** (leak-guarded checkout, PR-era test overlay):

```bash
python convert_to_harbor.py click harbor_tasks/click \
    --repo-url https://github.com/pallets/click \
    --agent-network public    # 'allowlist' (default) on Linux/Modal;
                              # macOS docker can't enforce network policies
```

**3. Gate tasks** (oracle must pass, nop must fail; writes `manifest.json`):

```bash
python gate_tasks.py harbor_tasks/click -k 3
```

**4. Collect outcomes** (one run per route; repeat over models):

```bash
.venv/bin/harbor run -p harbor_tasks/click -a claude-code -m claude-opus-5 \
    -n 1 --jobs-dir harbor_runs/<label> \
    --ae CLAUDE_CODE_OAUTH_TOKEN=... --ae CLAUDE_FORCE_OAUTH=1
# codex routes: -a codex -m gpt-5.6-sol --ae CODEX_FORCE_AUTH_JSON=1 (needs -m)
```

**5. Build the dataset** (joins task metadata + gate verdicts, drops
audit-tainted trials, stamps merge dates for the temporal split):

```bash
python build_dataset.py harbor_runs --tasks-dir harbor_tasks/click --out dataset.jsonl
```

**6. Run a router rung** (each rung emits a picks file):

```bash
python baseline_router.py harbor_tasks/click --router-model claude-sonnet-4-6  # v0 zero-shot
python profile_router.py dataset.jsonl   # P1: prompt + train-split track record
python knn_router.py dataset.jsonl -k 3  # P2: TF-IDF nearest-neighbour, no LLM
```

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
| `gate_tasks.py` | Task filter. Keeps a task only if oracle k runs all pass and nop k runs all fail; manifest.json records verdicts |
| `audit_trajectories.py` | Cheat detector. Scans executed commands (not raw text) for upstream fetches / package-index installs of the target repo |
| `build_dataset.py` | Flattens trials into `dataset.jsonl`: joins task.toml metadata + manifest verdicts, drops tainted/ungated trials loudly, resolves merge dates |
| `baseline_router.py` | Zero-shot prompted router (v0), adapted from Agent-as-a-Router (arXiv:2606.22902) |
| `profile_router.py` | P1: prompted router + per-route track record generated from the train split |
| `knn_router.py` | P2: TF-IDF kNN over instructions; no LLM, the trivial baseline |
| `scoreboard.py` | The one policy table: always-\<route\>, random, router (from picks), oracle ceiling; temporal split; routable subset reported separately |

## Untracked working state (regenerable or dataset-store material)

- `click/ pytest/ fastify/ river/` - mined repo clones (`.codeprobe/tasks` inside)
- `harbor_tasks/click/` - 20 converted tasks + `manifest.json` (19 pass)
- `harbor_runs/` - collected trials (the raw dataset: 19 tasks x 7 routes)
- `dataset.jsonl` - flattened trials
- `router_sweep/` - v0 zero-shot picks from 7 router models x 19 tasks
- `logs/` - run logs

## Known gotchas (details in PLAN.md)

- `harbor run` takes ONE `-p` path; use a dataset dir for subsets, and give
  concurrent jobs separate `--jobs-dir`s (timestamped names collide).
- Network allowlist enforcement requires Linux (Modal maps it natively to
  sandbox egress params); macOS Docker Desktop fails harbor's kernel probe,
  so local runs are `--agent-network public` + Rules + audit only.
- codeprobe mining scans only the most recent `count*8` merged PRs
  (count caps at 20); deeper history needs a codeprobe patch.
- Instruction-only anti-cheat deters some models, not all (opus attempted
  `pip download` despite the Rules); the allowlist is the real defense.
