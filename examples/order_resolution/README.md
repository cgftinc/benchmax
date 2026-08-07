# Order resolution

This example models a post-purchase support agent that can cancel an unfulfilled item,
change a pre-handoff shipping address, or replace an unfulfilled item with an in-stock,
same-price variant. The model receives typed business tools; it never receives raw SQL or
database credentials.

## Threat model

Each benchmark, validation, or training run uses an expiring Neon child branch. Every rollout
gets a separate `world_id` inside that branch, and every operational key and query is scoped to
that world. Mutations are atomic, idempotent business commands with receipts and audit events.

The hosted prototype intentionally bundles one pooled runtime DSN because BenchMAX 0.2.3 does
not provide general secret injection for hosted validation. That DSN must belong to a random,
low-privilege role created only on the disposable child branch. It must never be written to a
manifest, report, trace, or log. The direct admin DSN and Neon API key stay on the launcher's
machine and must never enter an environment bundle.

Generated manifests and reports live under `artifacts/` and are ignored. Held-out evaluation
rows, raw Olist inputs, credentials, and `.env` files are also ignored. Olist calibration is
optional, local-only, and never supplies task text or ground-truth actions.

## Commands

Run commands from the BenchMAX repository root unless noted otherwise:

```bash
uv sync --all-groups
uv run python examples/order_resolution/main.py preflight \
  --manifest examples/order_resolution/artifacts/implementation.json
uv run ruff check examples/order_resolution
```

`preflight` verifies the prepared superproject base, the pinned and initialized BenchMAX
gitlink, the Python/package versions, the audited `BaseEnv` contract, and both repositories'
allowlisted working-tree changes before it records the implementation manifest.

The live Neon and model-backed gates use the dedicated project configured by the ignored root
`.neon.env` file:

```bash
uv run python examples/order_resolution/main.py setup-neon
uv run pytest examples/order_resolution/tests/test_neon_integration.py -m integration -q
CASTFORM_PROFILE=default uv run python examples/order_resolution/main.py validate --hosted
CASTFORM_PROFILE=default uv run python examples/order_resolution/main.py baseline \
  --manifest examples/order_resolution/artifacts/baseline.json
uv run python examples/order_resolution/main.py probe-signal \
  --manifest examples/order_resolution/artifacts/baseline.json
uv run python examples/order_resolution/main.py report \
  --manifest examples/order_resolution/artifacts/baseline.json
uv run python examples/order_resolution/main.py verify-report \
  examples/order_resolution/artifacts/baseline.json
uv run python examples/order_resolution/main.py demo --frozen-cases
uv run python examples/order_resolution/main.py branches --assert-clean
```
