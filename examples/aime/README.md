# aime

AIME competition math (`aime/aime@latest` via Harbor) solved inside Modal
sandboxes by an offline-installed Mini-SWE agent — the upstream
mini-swe-agent loop replicated with only the Python standard library, so
sandbox setup needs no apt, PyPI, or wheel transfer.

Purpose: the fast Harbor smoke — runtime dataset resolution, sandboxed
agents calling the model through the run tunnel, and independent
per-rollout failure settlement.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/aime
# credentials: Modal from ~/.modal.toml
uv run python main.py             # data (Harbor resolve) → validate: two real Modal trials (no launch)
uv run python main.py launch      # train on GPUs (asks first; spends credits)
```

Sandbox payload (uploaded per trial by `aime_agent.py`): `mini_swe_probe.py`
(the agent loop), `castform_model.py` (stdlib model client), and
`run_mini_castform.py` (upstream-config driver).
