<picture>
  <img alt="BenchMax" src="./packages/benchmax/static/benchmax.png" width="100%">
</picture>

# BenchMax workspace

This workspace contains two independently installable Python distributions:

- `packages/benchmax`: the platform-independent environment runtime.
- `packages/castform`: the Castform SDK and CLI, which depends on BenchMax.

Standalone environment projects live under `examples/`.

## Tests

Run each distribution independently so their same-named test packages do not
collide during collection:

```bash
uv run --project packages/benchmax pytest -c packages/benchmax/pytest.ini packages/benchmax/tests
uv run --project packages/castform pytest -c packages/castform/pytest.ini packages/castform/tests
uv run pytest
```

The final command runs the workspace-level package-boundary tests.
