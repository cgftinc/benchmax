# Castform router training

A small, separate training path for the runtime in `../castform-router`:

```text
approved Git repos -> task seeds -> measured backend rollouts -> Qwen JSONL -> LoRA
```

It intentionally has no UI, workflow simulator, tracing system, cloud launcher, or
credential store. Git authentication uses the customer's existing SSH agent or Git
credential helper; secrets do not belong in the project file.

## 1. Describe repositories and backends

Copy `project.example.json`. Repositories may be SSH/HTTPS URLs or local Git paths.
Pin `revision` to the branch, tag, or commit the training run should inspect. Backend
names must match the aliases configured in `../castform-router/litellm/config.yaml`.

## 2. Mine task seeds

```bash
uv run castform-router-training mine project.json --output run --limit-per-repo 100
```

This clones each approved repository and writes `run/tasks.jsonl` from recent
non-merge commit subjects and bodies. Enterprises can replace that file with vetted
issue or pull-request tasks using the same fields: `task_id` and `task` are required.
No source code is sent to a model during this step.

## 3. Run every task on every backend

The evaluator/rollout system writes one row per attempt to `outcomes.jsonl`:

```json
{"task_id":"api:abc123","backend":"small-route","success":true,"input_tokens":120000,"cache_read_tokens":80000,"output_tokens":6000}
```

Repeat rows are supported. Success targets use a Beta(1,1) posterior mean, so one
success/failure becomes `0.6667`/`0.3333` rather than false certainty at `1`/`0`.
Exact mean token counts stay in audit-only label metadata; the learned response uses
stable input, cache-read, and output bands. A task is excluded unless every configured
backend has at least one result. This package deliberately does not fabricate labels
from Git history.

## 4. Build the scorer dataset

```bash
uv run castform-router-training dataset project.json \
  --tasks run/tasks.jsonl --outcomes outcomes.jsonl --output run/train.jsonl
```

The output is chat-format JSONL matching the runtime scorer contract. Costs are not
included in model input; the runtime policy owns cost-based selection.

### Scaffold from the shared Harbor corpus

The `castform-ai/model-router` repository already contains audited Harbor outcomes.
Build a pinned repo-temporal train/eval workspace directly from a local checkout:

```bash
uv run castform-router-training scaffold-model-router /path/to/model-router \
  --output run/model-router-sft \
  --model claude-haiku-4-5 \
  --model claude-opus-5 \
  --model claude-sonnet-4-6 \
  --model gpt-5.6-luna \
  --model gpt-5.6-sol \
  --model gpt-5.6-terra
```

Add `--model deepseek-v4-flash` only after its full 324-task farm is published.
Full-matrix filtering intentionally excludes tasks missing any selected model, so a
partial DeepSeek sample would reduce the entire SFT set to those sampled tasks. The
DeepSeek route ID is `claude-code/deepseek-v4-flash@deepseek` even though it uses the
Claude Code harness through DeepSeek's Anthropic-compatible endpoint.

The restart-safe DGX collector and systemd unit live on the model-router branch
`codex/deepseek-farm`.

## 5. Fine-tune Qwen

```bash
uv sync --extra training
uv run castform-router-training train --dataset run/train.jsonl \
  --model Qwen/Qwen3.5-0.8B --output run/qwen-router-lora
```

Only assistant target tokens contribute to loss; task/request prompt tokens are
masked. The result is a LoRA adapter and tokenizer. Serve the adapter behind LiteLLM's
`castform-router-qwen` alias, validate on held-out tasks, and only promote it after it
beats the existing scorer on routing success and cost.
