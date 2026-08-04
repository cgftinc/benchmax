# Use Castform from an existing coding harness

This mode assumes the user has already chosen the harness. Castform routes only
the model deployment underneath it:

```text
Codex / Claude Code / OpenAI-compatible client
  -> stable Castform alias on LiteLLM
  -> local Qwen 0.8B scores compatible backends
  -> deterministic Castform cost-quality policy
  -> selected LiteLLM backend alias
  -> provider
```

The stable aliases are:

| Caller | LiteLLM model |
| --- | --- |
| Codex CLI | `castform-auto-codex` |
| Claude Code | `castform-auto-claude` |
| OpenAI-compatible client, OpenCode, or local app | `castform-auto-open` |

The harness-specific alias matters. It prevents the router from considering a
backend that has not been validated with that harness protocol.

## 1. Start Qwen and LiteLLM

```bash
cd /Users/jasonwong/Desktop/benchmax/packages/castform/litellm-router

docker desktop enable model-runner --tcp=12434
docker model pull hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M

export CASTFORM_ROUTER_UPSTREAM_MODEL='openai/huggingface.co/unsloth/qwen3.5-0.8b-gguf:Q4_K_M'
export CASTFORM_ROUTER_UPSTREAM_BASE_URL=http://model-runner.docker.internal/engines/v1
export CASTFORM_ROUTER_UPSTREAM_API_KEY=not-needed
export CASTFORM_ROUTER_MODEL_LABEL='Qwen 3.5 0.8B Q4_K_M'
export CASTFORM_ROUTER_MODEL_STATUS=stock_untrained

docker compose up -d --force-recreate
uv run python scripts/smoke_test.py
```

The smoke test verifies Chat Completions, OpenAI Responses, Anthropic Messages,
Qwen's strict JSON shape, the model rewrite, and session pinning.

## 2. Replace the mock backends

The zero-key stack deliberately routes selected backends to `mock-upstream`.
Copy `examples/in-harness.env.example` to `.env`, insert the provider model IDs
and API credentials you actually use, and restart Compose:

```bash
cp examples/in-harness.env.example .env
# Edit .env. Do not commit it.
docker compose up -d --force-recreate litellm trace-ui
uv run python scripts/smoke_test.py
```

Only list a model under a harness in `compatible_harnesses` after testing its
streaming and tool-call behavior with that harness. A model being reachable by
LiteLLM does not prove that it implements the protocol details Codex or Claude
Code relies on.

ChatGPT and Claude subscription sessions cannot be used by LiteLLM as provider
API credentials. Gateway-routed calls are billed to the API credentials in the
LiteLLM deployment.

## 3. Codex CLI

Create a separate Codex profile so your normal setup remains unchanged:

```toml
# ~/.codex/castform.config.toml
model = "castform-auto-codex"
model_provider = "castform"

[model_providers.castform]
name = "Castform via LiteLLM"
base_url = "http://localhost:4000/v1"
env_key = "LITELLM_API_KEY"
wire_api = "responses"
env_http_headers = {
  "x-castform-session-id" = "CASTFORM_SESSION_ID",
  "x-castform-trace-id" = "CASTFORM_TRACE_ID"
}
```

Start a routed Codex session:

```bash
export LITELLM_API_KEY=sk-local-dev
export CASTFORM_SESSION_ID="codex-$(uuidgen)"
export CASTFORM_TRACE_ID="trace-$(uuidgen)"
codex --profile castform
```

Codex sends Responses API calls to LiteLLM. The first call is scored by Qwen;
later calls with the same session ID reuse the selected backend.

## 4. Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=sk-local-dev
export ANTHROPIC_CUSTOM_MODEL_OPTION=castform-auto-claude
export ANTHROPIC_DEFAULT_OPUS_MODEL=castform-auto-claude
export ANTHROPIC_DEFAULT_SONNET_MODEL=castform-auto-claude
export ANTHROPIC_DEFAULT_HAIKU_MODEL=castform-auto-claude
export CLAUDE_CODE_SUBAGENT_MODEL=castform-auto-claude
export CASTFORM_SESSION_ID="claude-$(uuidgen)"
export CASTFORM_TRACE_ID="trace-$(uuidgen)"
export ANTHROPIC_CUSTOM_HEADERS="$(printf 'x-castform-session-id: %s\nx-castform-trace-id: %s' "$CASTFORM_SESSION_ID" "$CASTFORM_TRACE_ID")"

claude --model castform-auto-claude
```

Claude Code sends Anthropic Messages calls to the same LiteLLM server. Its
model picker may not discover a non-Claude alias automatically, which is why
the custom model option is set explicitly. Pinning all three built-in tiers and
the subagent model prevents background Haiku or subagent calls from bypassing
the Castform alias.

## 5. OpenAI-compatible local Llama or OpenCode-style client

Point the client at LiteLLM, not directly at Ollama or Model Runner:

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY=sk-local-dev
export OPENAI_MODEL=castform-auto-open
```

For a direct protocol check:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H 'Authorization: Bearer sk-local-dev' \
  -H 'Content-Type: application/json' \
  -H "x-castform-session-id: open-$(uuidgen)" \
  -H "x-castform-trace-id: trace-$(uuidgen)" \
  -d '{
    "model": "castform-auto-open",
    "messages": [{"role": "user", "content": "Explain this repository."}]
  }'
```

If “open llama” means a local Llama backend, configure `llama-route` in `.env`
and include it in `CASTFORM_AUTO_ROUTES_JSON`. If it means OpenCode, create an
OpenAI-compatible provider in OpenCode using the same base URL, API key, and
`castform-auto-open` model name.

## 6. Inspect the decision

Use the trace ID exported for the harness session:

```bash
curl "http://localhost:3000/api/traces/$CASTFORM_TRACE_ID" | jq
```

The important stages are:

```text
harness.model_request_received
litellm.scorer_request_received
router.candidates_scored
policy.route_selected
litellm.model_rewritten
provider.request_received
```

To force one backend for diagnosis without disabling the gateway, send an
`x-castform-route` header containing a configured gateway alias such as
`glm-route` or `claude-route`.
