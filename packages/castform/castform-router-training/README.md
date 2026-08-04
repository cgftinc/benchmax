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
{"task_id":"api:abc123","backend":"small-route","success":true}
```

Repeat rows are supported. Success probabilities are the observed success rate per
backend. A task is excluded unless every configured backend has at least one result.
This package deliberately does not fabricate labels from Git history.

## 4. Build the scorer dataset

```bash
uv run castform-router-training dataset project.json \
  --tasks run/tasks.jsonl --outcomes outcomes.jsonl --output run/train.jsonl
```

The output is chat-format JSONL matching the runtime scorer contract. Costs are not
included in model input; the runtime policy owns cost-based selection.

## 5. Fine-tune Qwen

```bash
uv sync --extra training
uv run castform-router-training train --dataset run/train.jsonl \
  --model Qwen/Qwen3.5-0.8B --output run/qwen-router-lora
```

The result is a LoRA adapter and tokenizer. Serve the adapter behind LiteLLM's
`castform-router-qwen` alias, validate on held-out tasks, and only promote it after it
beats the existing scorer on routing success and cost.
