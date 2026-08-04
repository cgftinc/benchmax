# Castform LiteLLM router

One routing flow:

```text
task -> Qwen scorer -> cheapest adequate backend -> LiteLLM -> response
```

Call LiteLLM with the model `castform-auto`. The pre-call hook extracts the task,
asks the configured Qwen model for one success probability per backend, selects
the cheapest backend at or above `CASTFORM_ROUTER_QUALITY_THRESHOLD`, and rewrites
the model to that LiteLLM alias. LiteLLM then sends the original request to the
selected backend and returns its response unchanged.

The scorer never sees backend costs. If no backend reaches the threshold, the
highest-scored backend is used. If scoring fails, `CASTFORM_AUTO_FALLBACK_MODEL`
is used.

## Configuration

Start LiteLLM with `litellm_config.yaml` after setting the model/provider variables
referenced there. The defaults expect `small-route`, `medium-route`, and
`large-route`. Override the catalog with:

```bash
export CASTFORM_AUTO_BACKENDS_JSON='[
  {"name":"fast","model":"provider/model-a","provider":"provider","estimated_cost_usd":0.05},
  {"name":"strong","model":"provider/model-b","provider":"provider","estimated_cost_usd":0.20}
]'
export CASTFORM_AUTO_FALLBACK_MODEL=strong
export CASTFORM_ROUTER_MODEL_NAME=castform-router-qwen
export CASTFORM_ROUTER_MODEL_BASE_URL=http://localhost:4000
export CASTFORM_ROUTER_QUALITY_THRESHOLD=0.84
```

Each backend `name` must also exist as a LiteLLM `model_name`. Send a normal
OpenAI-compatible request:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"castform-auto","messages":[{"role":"user","content":"Fix the parser"}]}'
```

Router details are attached to LiteLLM request metadata under `castform_router`.
There is no UI, workflow/project generator, harness-specific routing, session
pinning, or tracing subsystem.
