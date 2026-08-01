# GitLab handbook BM25 gateway A/B

This experiment compares the text rollout path without the standalone TITO
Gateway against the current gateway path. It holds the corpus, dataset, model,
prompt, tools, rewards, context budget, rollout shape, optimizer settings, and
five-epoch schedule constant.

The two Castform snapshots are intentionally immutable:

- `pre_harbor`: `466e947c25eb137e1778d5f8a33c87cc906b729c`
- `post_harbor`: `1aed982f2621d87438b3ad095818440a4ea930c5`

Their scheduler-visible job names are:

- `gitlab-bm25-ab-pre-harbor-no-gateway`
- `gitlab-bm25-ab-post-harbor-gateway-pre-harbor-effective-3-tools`
- `gitlab-bm25-ab-post-harbor-gateway-optimized-pre-harbor-effective-3-tools`

The post-Harbor name deliberately calls out its compatibility budget. The
historical trainer accepted `max_tool_calls=4` but stopped on `>= 4` before
executing the fourth search. The current gateway correctly interprets a limit
of four as four executed calls, so this replacement post-Harbor arm keeps the
shared prompt at "up to 4 searches" while enforcing an effective gateway budget
of three. This isolates gateway behavior from the historical off-by-one.

The historical task is copied from the pre-Harbor snapshot's `simple.yaml`. The
gateway task is copied from the current snapshot's `simple.yaml`. Both are
submitted through `POST /v1/train/runs/internal/launch`; the public customer
Benchmax-version floor is not involved.

## Inputs

- GitLab handbook commit: `3078d0213524f8ca0c0e3a70680a21929a9f65ff`
- Corpus: `gitlab-handbook-bm25-3078d0213524-staging`
- Expected chunks: `31,665`
- Chunking: min `1024`, max `2048`, overlap `128`
- Dataset revision: `efafe62fe5da43904e46f90c8031f95b3304bce6`
- Dataset: `256` train rows / `64` eval rows, with pinned SHA-256 values
- Model: `Qwen/Qwen3.5-4B`
- Context: `16,384` total tokens
- Turns/advertised searches/effective executed searches: `5` / `4` / `3`
- Train/eval samples per prompt: `9` / `9`
- Epochs: `5`
- Learning rate: `4e-6`

`experiment.json` is the machine-readable source of truth.

## Prepare and verify

Every executable owns its dependencies through PEP 723 metadata and runs in an
isolated uv environment.

```bash
uv run --script prepare.py
uv run --script pre_harbor/build_bundle.py
uv run --script post_harbor/build_bundle.py
uv run --script verify_parity.py
```

`prepare.py` reuses and verifies the existing corpus. It refuses to append to an
existing corpus. Use `--ingest` only when the named corpus does not exist.

The two bundle formats cannot be shared: each bundle must be built by the
Benchmax version embedded in its trainer snapshot. Local environment support
modules are captured into each bundle, so neither training job imports code from
another example directory.

The linked historical example's `max_rollout_len` was stale by the pre-Harbor
snapshot: Platform accepted `max_context_len` and mapped it to Slime's
`rollout_max_context_len`. These scheduler tasks bypass `LAUNCH_CONFIG` and set
`ARG_ROLLOUT_MAX_CONTEXT_LEN=16384` plus the matching eval value directly. For
this text-only Qwen run, the historical trainer enables its integrated TITO
path, which enforces that total-context limit.

## Upload once

After both bundles exist:

```bash
uv run --script upload_assets.py
```

This uploads one content-addressed dataset prefix. The pre-Harbor task addresses
the two JSONL files directly; the gateway task addresses their containing
prefix. The generated blob paths are recorded in
`artifacts/uploaded_assets.json`.

## Inspect and launch

The launcher is dry-run-only unless `--launch` is supplied:

```bash
uv run launch_ab.py
```

For real submission, set `CASTFORM_AUTH_TOKEN` and `WANDB_API_KEY`, then submit
both arms to the same pool. Platform injects the scheduler-owned `GIT_TOKEN`;
the launcher supplies staging's `github-https-v1` private-workdir descriptor
using the same immutable SHA as the task:

```bash
uv run launch_ab.py --arm both --pool gpu4 --launch
```

Launching both arms is sequential. If the first succeeds and the second is
rejected, the first remains scheduled and its run/job IDs are printed.

### Post-Harbor optimized runner

``post_harbor/task_optimized.yaml`` is a derivative of
``simple-optimized.yaml`` from the same immutable post-Harbor Castform
snapshot.  It retains the post-Harbor environment bundle, dataset, five-epoch
A/B hyperparameters, 16,384-token context budget, and effective three-tool
compatibility limit.  It is a separate runner-implementation experiment, not
an interchangeable replacement for the standard gateway A/B arm.

Inspect or submit it through the same scheduler path:

```bash
uv run launch_ab.py --arm post_harbor --task-variant optimized
uv run launch_ab.py --arm post_harbor --task-variant optimized --pool gpu4 --launch
```

The internal route strips `resources` and `setup` from both copied task files;
the selected scheduler pool owns those settings. The immutable `workdir`,
environment, secrets, and `run` block remain from the corresponding snapshot.

## Interpretation

This is a Harbor-boundary regression test, operationally labeled
"gateway vs. no gateway." Harbor and the current snapshot contain changes
beyond the gateway, so a result can implicate the boundary but cannot alone
prove gateway causality.
