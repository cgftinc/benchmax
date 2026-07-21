# BenchMax workspace

This workspace contains two independently installable Python distributions:

- `packages/benchmax`: the platform-independent environment runtime.
- `packages/castform`: the Castform SDK and CLI, which depends on BenchMax.

Standalone environment projects live under `examples/`.

```text
examples ──> benchmax
    │
    └──────> castform        # only when platform features are used

castform ──> benchmax
benchmax -X-> castform
```

BenchMax owns only the portable runtime: environment execution, ordered dataset
types, rewards, stable identities and bundle construction. Castform owns login,
platform clients, validation, uploads, launches, hosted corpus/RAG workflows and
project scaffolding. Each example is a standalone Python 3.12 project with its
dependencies declared in its own `pyproject.toml`.

Start a new Castform project with `castform setup`; the generated `main.py` is
the executable workflow for data preparation, local validation, bundling, upload
and an explicitly confirmed launch.

## Tests

Run each distribution independently so their same-named test packages do not
collide during collection:

```bash
uv run --project packages/benchmax pytest -c packages/benchmax/pytest.ini packages/benchmax/tests
uv run --project packages/castform pytest -c packages/castform/pytest.ini packages/castform/tests
uv run pytest tests/architecture
```

The final command runs the workspace-level package-boundary tests.
