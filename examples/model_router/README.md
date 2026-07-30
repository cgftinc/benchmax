# model_router

Train a model router for coding agents on repo-native data: mine tasks
from a repository's merged-PR history with
[codeprobe](https://github.com/sjarmak/codeprobe), convert them into
[harbor](https://pypi.org/project/harbor/) tasks, collect per-model
rollout outcomes, and train a router to predict per-model success
probability and token usage. See [PLAN.md](PLAN.md) for the agreed
design, findings so far, and watch-outs.

## Pipeline (all standalone scripts)

```
codeprobe mine            # tasks from repo history (needs gh auth)
convert_to_harbor.py      # codeprobe task dirs -> harbor task dirs
gate_tasks.py             # filter: oracle k=3 must pass, nop k=3 must fail
harbor run                # collect (task x model x attempt) outcomes
audit_trajectories.py     # flag rollouts that fetched the ground truth
baseline_router.py        # zero-shot prompted-router baseline (ACRouter-style)
```

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python codeprobe 'harbor>=0.18,<0.19'
gh auth status   # codeprobe mining needs authenticated gh for PR narratives
```

## Per-repo flow (click as the worked example)

```bash
git clone https://github.com/pallets/click        # full clone; shallow can't be mined
(cd click && ../.venv/bin/codeprobe mine . --goal quality --count 20 --no-interactive)

python convert_to_harbor.py click harbor_tasks/click \
    --repo-url https://github.com/pallets/click \
    --agent-network public          # 'allowlist' (default) on Linux/Modal;
                                    # macOS docker can't enforce phase policies

python gate_tasks.py harbor_tasks/click -k 3      # writes manifest.json

# collect outcomes (repeat per agent/model; k = attempts per task)
.venv/bin/harbor run -p harbor_tasks/click -a claude-code -m claude-sonnet-4-6 \
    -k 3 -n 4 --jobs-dir harbor_runs/<label> \
    --ae CLAUDE_CODE_OAUTH_TOKEN=... --ae CLAUDE_FORCE_OAUTH=1
# codex agents: -a codex -m <model> --ae CODEX_FORCE_AUTH_JSON=1 (needs -m)

python audit_trajectories.py harbor_runs/<label> --package click
python baseline_router.py harbor_tasks/click --router-model claude-sonnet-4-6
```

Each trial's `result.json` carries the full training tuple: reward,
n_input/cache/output tokens, cost_usd, and phase timings; trajectories
live under the trial dir.

## What each file is

| File | Role |
|---|---|
| `convert_to_harbor.py` | Data prep. Leak-guarded Dockerfile (fetch base commit only), PR-test overlay verifier (codeprobe's own verifier never restores PR tests), oracle `solve.sh` = PR diff, network policy + anti-cheat rules in every task |
| `gate_tasks.py` | Task filter. Keeps a task only if oracle k runs all pass and nop k runs all fail; manifest.json records verdicts (pass / oracle_fail / nop_pass / flaky) |
| `audit_trajectories.py` | Post-run cheat detector. Scans executed commands (not raw text) for upstream fetches / package-index installs of the target repo |
| `baseline_router.py` | Zero-shot prompted router adapted from Agent-as-a-Router (arXiv:2606.22902); supports claude and codex CLIs as router engines |

## Untracked working state (regenerable or dataset-store material)

- `click/ pytest/ fastify/ river/` - mined repo clones (`.codeprobe/tasks` inside)
- `harbor_tasks/click/` - 20 converted tasks + `manifest.json` (19 pass) + `router_picks.jsonl`
- `harbor_runs/` - all collected trials (the raw dataset: ~68 rollouts as of 2026-07-29)
- `router_sweep/` - zero-shot picks from 7 router models x 19 tasks
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
