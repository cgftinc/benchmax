# Harvey

Trains `Qwen/Qwen3.5-35B-A3B` on
[`harveyai/lab:latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest)
using Harvey's native agent loop, Modal sandboxes, and the dataset's Claude
Sonnet judge.

From the benchmax workspace root:

```bash
uv sync
cd examples/harvey

# Import the castform Modal profile into this shell (requires jq).
modal_config="$(MODAL_PROFILE=castform uv run modal config show --no-redact)"
export MODAL_TOKEN_ID="$(jq -r .token_id <<<"$modal_config")"
export MODAL_TOKEN_SECRET="$(jq -r .token_secret <<<"$modal_config")"
unset modal_config

export ANTHROPIC_API_KEY='<anthropic-api-key>'

uv run python main.py launch \
  --judge-provider anthropic \
  --judge-model 'anthropic/claude-sonnet-4-6'

# Cost-efficient GPT judge alternative: replace ANTHROPIC_API_KEY and the
# command above with the commented configuration below.
#
# export OPENAI_API_KEY='<dedicated-castform-api-key>'
# export OPENAI_BASE_URL='https://llm.castform.com/v1'
# export OPENAI_API_BASE='https://llm.castform.com/v1'
#
# uv run python main.py launch \
#   --judge-provider openai \
#   --judge-model 'openai/gpt-5.4-nano'
```
