# benchmax

benchmax is a python sdk for defining and training reinforcement-learning environments: tools, rewards and datasets as code.

extend [`BaseEnv`](packages/benchmax/src/benchmax/envs/base/env.py) for a simple model-and-tool loop, or use [`HarborEnv`](packages/benchmax/src/benchmax/envs/harbor/env.py) to train with arbitrary harnesses and sandboxes through harbor.

the `castform` package adds dataset generation helpers, environment validation, and launching training jobs.

python 3.12 is required.

## install

```bash
uv tool install -U castform
# or
pip install -U castform
```
this will install both castform and benchmax

check that the cli is available:

```bash
castform --version
```

## get started

```bash
castform setup
```

this signs you in and installs coding-agent guidance. it does not generate an
environment; ask your agent to choose and adapt the closest maintained example.

working examples live under [`examples/`](examples/)

see the [benchmax guide](packages/benchmax/README.md) and the [castform guide](packages/castform/README.md).
