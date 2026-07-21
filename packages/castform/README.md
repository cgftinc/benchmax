# Castform

Castform is the SDK and small companion CLI for the Castform training platform.
It depends on the platform-independent `benchmax` runtime and adds authentication,
validation, uploads, launches, run inspection, hosted corpus/RAG libraries, trace
processing and project scaffolding.

Python 3.12 is required.

## Login profiles

Production is the built-in default, so normal users only run `castform login`.
Internal and self-hosted deployments use named profiles:

```bash
castform login --profile staging --domain castform.dev
castform profile activate staging
castform profile list
```

Use `castform --profile prod <command>` for a one-command override. Self-hosted
profiles can pass `--api-url`, `--auth-url`, `--llm-url`, and `--app-url` to
`castform login` when their services do not follow the standard subdomain
layout. Profile routing is stored in `~/.castform/config.toml`; login sessions
remain in the protected `~/.castform/credentials.json`.

## Start a standalone project

```bash
uv tool install castform
castform login
castform setup --dir my-environment
cd my-environment
uv sync
uv run python main.py
```

`castform setup` writes a `pyproject.toml`, a runnable `main.py`, small seed
datasets and script-focused agent skills. Bare `main.py` prepares data and runs
local validation with two siblings; it never launches training. After reviewing
the result, launch explicitly:

```bash
uv run python main.py launch
```

The generated script makes every boundary visible:

1. prepare or reference data with normal Python library calls;
2. materialize an example through `Environment.create_dataset`, then call
   `validate_environment` for the real two-sibling environment contract;
3. call `dump_bundle(..., pip_dependencies=[...])` with explicit remote deps;
4. call `upload_training_run(bundle=...)` with that exact bundle and only the
   dataset splits that should be uploaded;
5. ask for cost confirmation, then call `TrainerClient.launch_training_run`.

The CLI does not duplicate that orchestration. Use it for `login`, `setup`,
`doctor`, `guide`, `runs` inspection and cancelling a run with `castform stop`.
`castform whoami` and `castform logout` manage the active session, and
`castform with-auth -- <cmd>` runs an external command with the profile's
credential injected as `CASTFORM_AUTH_TOKEN`.

## Optional libraries

Add only the features the project uses:

```bash
uv add 'castform[rag]'
uv add 'castform[traces]'
```

Vector-store integrations ship as their own extras: `castform[chroma]`,
`castform[pinecone]`, `castform[turbopuffer]`.

Corpus ingestion, QA generation and trace preparation are Python library
workflows. Keep their calls in the project's data stage so the preparation is
reviewable and reproducible. Remote rollout imports must also be listed explicitly
in the bundle's `pip_dependencies`; project dependencies and rollout dependencies
are related but serve different runtimes.

Dataset upload is optional. Omit `train_dataset` and/or `eval_dataset` when the
environment resolves that split at runtime, such as through Harbor or Git. Passing
an empty list is explicit and uploads an empty JSONL; it does not mean “omit.”

## License

Apache 2.0 © 2026 CGFT Inc.
