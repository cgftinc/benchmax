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

`castform setup --template sft` scaffolds the env-less variant instead:
supervised fine-tuning over `{"messages": [...]}` rows, where
`python main.py validate` is a purely local dataset check and there is no
environment or reward to write.

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
uv run --project packages/benchmax pytest tests/architecture
```

The final command runs the workspace-level package-boundary tests. Keep every
invocation `--project`-scoped and pass the paths explicitly: a bare `uv run`
syncs every workspace member, including the heavyweight `examples/*` ones, and
`pytest` with no path argument picks up the root `pytest.ini` and silently runs
only the architecture tests.
