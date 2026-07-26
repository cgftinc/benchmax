# Harvey

Trains `Qwen/Qwen3.5-35B-A3B` on
[`harveyai/lab:latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest)
using Harvey's native agent loop, Modal sandboxes, and a GPT judge through
Castform's OpenAI-compatible endpoint.

From the BenchMax workspace root:

```bash
uv sync
cd examples/harvey

export MODAL_TOKEN_ID='<modal-token-id>'
export MODAL_TOKEN_SECRET='<modal-token-secret>'

export HARVEY_JUDGE_MODEL='openai/gpt-5.4-nano'
export OPENAI_API_KEY='<dedicated-castform-api-key>'
export OPENAI_BASE_URL='https://llm.castform.com/v1'
export OPENAI_API_BASE='https://llm.castform.com/v1'

# Required by a legacy harveyai/lab placeholder; unused by the GPT judge.
export ANTHROPIC_API_KEY='unused-for-openai-judge'
export HARVEY_VERIFIER_ENV_VARS='OPENAI_API_KEY,OPENAI_BASE_URL,OPENAI_API_BASE,ANTHROPIC_API_KEY'

uv run python main.py launch
```

The launcher uses an existing Castform session or starts interactive login. It
does not require `castform with-auth`.
