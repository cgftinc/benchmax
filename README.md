# BenchMax

BenchMax is a Python SDK for defining and training reinforcement-learning
environments: tools, rewards and datasets as code. The `castform` package adds
the Castform platform on top: login, validation, uploads and GPU launches.

Python 3.12 is required.

## Install

```bash
uv add castform        # or: pip install castform
```

Installing `castform` pulls in `benchmax`. Install `benchmax` alone when you
only need the environment runtime without any platform integration.

## Get started

```bash
castform setup
```

This signs you in and scaffolds a project whose `main.py` owns the whole
workflow: bare `python main.py` prepares data and validates the environment
locally (no launch), and `python main.py launch` bundles, uploads and starts a
training run after an explicit confirmation.

Working examples live under [`examples/`](examples/), from a single-turn math
env to multimodal tool use and third-party Harbor harnesses. For the API,
see the [BenchMax guide](packages/benchmax/README.md) and the
[Castform guide](packages/castform/README.md).

## Development

Run each distribution's tests independently so their same-named test packages
do not collide during collection:

```bash
uv run --project packages/benchmax pytest -c packages/benchmax/pytest.ini packages/benchmax/tests
uv run --project packages/castform pytest -c packages/castform/pytest.ini packages/castform/tests
uv run pytest tests/architecture
```

The final command runs the workspace-level package-boundary tests.
