# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in
the `benchmax` repository.

## Commands

```bash
# Install with all extras for development
uv sync --all-extras

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Run unit tests (fast, no external services)
uv run pytest tests/unit

# Run a single test
uv run pytest tests/unit/path/to/test_file.py::test_function

# Run integration tests (requires real API credentials)
uv run pytest -m integration

# Add a dependency
uv add <package>
```

Line length is 100. Ruff rules: E, F, I, N, W, UP. Python 3.12 required.

## Layout

```
src/benchmax/
├── envs/             # core — env primitives + built-in envs
│   ├── base_env.py / types.py / tracking.py
│   ├── mcp/                  ([mcp] extra)
│   ├── excel/, crm/          (env-specific extras)
│   ├── math/, wikipedia/
│   └── postgres_search/      (search envs, [rag])
├── prompts/          # core — Hermes/OpenAI tool prompt rendering
├── bundle/           # core — env class bundling + validation
├── platform/         # core — HTTP clients for the Castform platform
├── multi_model/      # core — multi-LLM provider helpers
├── rubrics/          # core — reward rubric primitives
├── config.py         # core — BASE_DOMAIN-driven URL resolution
├── rag/              ([rag] extra) — chunkers, corpus backends, qa_generation, preprocess
└── traces/           ([traces] extra) — agent trace import + processing
```

The platform-API client code (`StorageClient`, `TrainerClient`,
`RolloutClient`) lives in `benchmax.platform` and is used both internally
(by `benchmax.rag.qa_generation`) and by the higher-level
[`castform-sdk`](../castform-sdk/) package.

## URL configuration

Every platform URL routes through `benchmax.config`:

```python
from benchmax import config
config.platform_url()    # https://api.castform.com  (clients add /v1/...)
config.llm_url()         # https://llm.castform.com/v1
```

Rollouts are reached through the platform API (`platform_url()` +
`/v1/rollout/stream`), which gates the API key and proxies to rollout-service —
there is no separate rollout URL.

Override via env vars: `CASTFORM_BASE_DOMAIN`, `CASTFORM_PLATFORM_URL`,
`CASTFORM_LLM_URL`. **Do not** hardcode platform URLs in source — always go
through `config`.

## Design principles

These apply across all of `benchmax`, especially the data-prep code
(`rag/`, `traces/`) which runs on diverse customer corpora and trace
formats.

### No hardcoded language or cultural assumptions

No English stop word lists, locale-specific date formats, or currency
symbols baked into library code. Use universal patterns (e.g. `\d+` for
all numbers) instead of domain-specific regex. If a heuristic only works
for English natural-language text, expose a threshold and let the
algorithm handle the rest — it doesn't belong in the library otherwise.

### Design the composition layer, not just the leaves

When building a set of related functions (filters, adapters, processing
steps), design how they compose before implementing the individual
pieces. A pipeline runner, combined result type, or chaining mechanism
should exist from day one. Without it, every consumer writes their own
ad-hoc accumulator.

### Fail loudly, never silently reorder or fix

If a user provides an invalid configuration (e.g. wrong ordering of
pipeline steps), raise an error explaining the constraint. Don't silently
reorder or "correct" — the caller won't learn the constraint and the
behavior becomes invisible.

### Structured metadata over string parsing

Return types that carry structured metadata (e.g.
`DropReason(filter, reason, detail)`) instead of encoding information
into strings that consumers must parse with `startswith()` hacks.

### Shared utilities need their own tests

Extracting shared code (HTTP retry, tokenization, etc.) is only valuable
if the shared utility has dedicated tests covering its edge cases. Tests
that exercise it transitively through callers don't count — they only
test the happy path.

### Preserve caller expectations

Functions in a pipeline should preserve input ordering unless there's a
documented reason not to. If a function must reorder (e.g. for
determinism), document it explicitly.

### External API code must be tested against real APIs

Unit tests with mocked HTTP responses cannot validate query formats,
column names, auth flows, pagination behavior, or rate limit handling.
Mocks return whatever you tell them to — they don't catch invalid SQL,
wrong endpoint paths, or missing fields.

This applies to all external integrations: trace adapters (Braintrust,
Langfuse), corpus backends (Postgres, Turbopuffer, Pinecone, Chroma), the
Castform platform API, and any future provider.

Any PR that changes fetch logic, query construction, column lists,
pagination, or retry behavior must pass integration tests before merge.
**This is non-negotiable.**

Reviewers: if a PR touches provider API interaction and the diff contains
no integration test changes, that is a red flag. Do not approve without
verifying the new behavior works against the real API.

Integration tests use `@pytest.mark.integration` and are skipped in CI.
Run locally: `uv run pytest -m integration` (credentials loaded from
`.env.test`).

## Adding a new RAG corpus backend

1. Create `src/benchmax/rag/corpus/<backend>/` with at minimum:
   - `source.py` — implements the `ChunkSource` protocol
   - `client.py` (if needed) — HTTP/SDK wrapper
   - `filter_mapper.py` (if applicable) — translates predicate AST → backend's filter DSL
2. Add an opt-in extra in `pyproject.toml`:
   `<backend> = ["<sdk-package>>=X"]`
3. Add unit tests in `tests/unit/rag/corpus/<backend>/`. Mark anything
   that hits the real backend with `@pytest.mark.integration`.
4. Add at least one integration test that exercises the live API end-to-end.

## Adding a new trace provider

1. Create `src/benchmax/traces/<provider>/` with `adapter.py` implementing
   the `TraceAdapter` protocol (`connect`, `list_projects`, `fetch_traces`).
2. Register in `traces/registry.py`.
3. Add unit + integration tests under `tests/unit/traces/<provider>/`.
