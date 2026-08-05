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
uv tool install -U castform
castform login
castform setup --dir my-environment
cd my-environment
```

`castform setup` writes coding-agent guidance and script-focused skills. It does
not generate environment code or choose between `BaseEnv`, `HarborEnv`, RAG, or
another task shape. Ask your agent to inspect the maintained
[Benchmax examples](https://github.com/castform-ai/benchmax/tree/main/examples),
choose the closest example, and adapt its project structure and `main.py`.

Once the agent has created the project, bare `main.py` should prepare data and
run local validation with two siblings; it must never launch training. After
reviewing the result, launch explicitly:

```bash
uv sync
uv run python main.py
uv run python main.py launch
```

The resulting script should make every boundary visible:

1. prepare or reference data with normal Python library calls;
2. materialize an example through `Environment.create_dataset`, then call
   `validate_environment` for the real two-sibling environment contract;
3. call `dump_bundle(..., pip_dependencies=[...])` with explicit remote deps;
4. call `upload_assets(bundle=...)` with that exact bundle and only the
   dataset splits that should be uploaded;
5. ask for cost confirmation, then call `TrainerClient.launch_training_run`.

The CLI does not duplicate that orchestration. Use it for `login`, `setup`,
`doctor`, `guide`, `runs` inspection and cancelling a run with `castform stop`.
`castform whoami` and `castform logout` manage the active session.

## Optional libraries

Add only the optional features the project uses:

```bash
uv add 'castform[rag]'
```

Trace adapters and `castform.traces.TracesPipeline` ship with the base
`castform` package; no trace-specific extra is required.

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
