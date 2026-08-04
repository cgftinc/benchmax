# Castform router

One inference-time flow:

```text
task -> Qwen scorer -> cheapest adequate policy -> LiteLLM backend -> response
```

Clients send a normal OpenAI-compatible request to LiteLLM using the public model
`castform-auto`. The LiteLLM pre-call hook extracts the task, asks Qwen for one
success probability per configured backend, and selects the least expensive backend
at or above `CASTFORM_ROUTER_QUALITY_THRESHOLD`. It rewrites the request model to
that backend alias; LiteLLM dispatches the original request and returns the backend
response unchanged.

The scorer never sees backend costs. If no backend reaches the threshold, the policy
uses the highest-scored backend. If scoring fails, it uses
`CASTFORM_AUTO_FALLBACK_MODEL`.

## Layout

```text
castform-router/
├── castform_router/        # scorer contract, policy, and LiteLLM callback
├── litellm/
│   ├── config.yaml         # public alias, scorer alias, and backend aliases
│   ├── Dockerfile          # LiteLLM plus the local callback package
│   ├── compose.yaml        # one-service local deployment
│   └── .env.example
└── tests/
```

The offline repository and fine-tuning pipeline lives separately in
`../castform-router-training`.

## Run with Docker Compose

```bash
cp litellm/.env.example litellm/.env
# Edit model endpoints and credentials in litellm/.env.
docker compose --env-file litellm/.env -f litellm/compose.yaml up --build
```

The image installs this Python package, allowing LiteLLM to import
`castform_router.litellm_callback.castform_auto_router`. The callback intercepts
only `castform-auto`; its request to the `castform-router-qwen` alias passes through
without being routed again.

The default catalog contains `small-route`, `medium-route`, and `large-route`.
Every `name` in `CASTFORM_AUTO_BACKENDS_JSON` must have a matching `model_name` in
`litellm/config.yaml`. Add or remove aliases in both places when changing the
catalog.

Send a request after the health check passes:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"castform-auto","messages":[{"role":"user","content":"Fix the parser"}]}'
```

Router details are attached to the internal LiteLLM request metadata under
`castform_router`. There is no UI, workflow generator, harness-specific routing,
session pinning, or tracing subsystem.
