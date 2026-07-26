# Harvey

Trains `Qwen/Qwen3.5-35B-A3B` on
[`harveyai/lab:latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest)
using Harvey's native agent loop, Modal sandboxes, and the dataset's Claude
Sonnet judge.

From the BenchMax workspace root:

```bash
uv sync
cd examples/harvey

export MODAL_TOKEN_ID='<modal-token-id>'
export MODAL_TOKEN_SECRET='<modal-token-secret>'
export ANTHROPIC_API_KEY='<anthropic-api-key>'

uv run python main.py launch \
  --judge-model 'anthropic/claude-sonnet-4-6' \
  --verifier-env-var ANTHROPIC_API_KEY

# Cost-efficient GPT judge alternative: replace ANTHROPIC_API_KEY and the
# command above with the commented configuration below. harveyai/lab declares
# ANTHROPIC_API_KEY even when RewardKit is overridden to use another provider.
#
# export OPENAI_API_KEY='<dedicated-castform-api-key>'
# export OPENAI_BASE_URL='https://llm.castform.com/v1'
# export OPENAI_API_BASE='https://llm.castform.com/v1'
# export ANTHROPIC_API_KEY='unused-for-openai-judge'
#
# uv run python main.py launch \
#   --judge-model 'openai/gpt-5.4-nano' \
#   --verifier-env-var OPENAI_API_KEY \
#   --verifier-env-var OPENAI_BASE_URL \
#   --verifier-env-var OPENAI_API_BASE \
#   --verifier-env-var ANTHROPIC_API_KEY
```

The launcher reads each `--verifier-env-var` value from the current environment,
so secrets do not appear in command-line arguments. It uses an existing
Castform session or starts interactive login; `castform with-auth` is not
required.
