# Live Harbor + Modal test

`test_harbor_modal_live.py` exercises the real path from `HarborEnv` through a
Harbor trial, a Modal sandbox, Mini-SWE-Agent, an OpenAI-compatible endpoint,
and the task verifier. Normal test runs exclude it.

Required inputs:

- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
- `BENCHMAX_HARBOR_API_KEY` or `CASTFORM_API_KEY`
- `BENCHMAX_RUN_HARBOR_MODAL_LIVE=1`

The endpoint defaults to `https://llm.castform.dev/v1` and the model defaults
to `qwen3.5-4b`. Override them with `BENCHMAX_HARBOR_BASE_URL` and
`BENCHMAX_HARBOR_MODEL`.

Harbor 0.18.0's Mini-SWE installer omits a LiteLLM runtime extra. Until a
release includes the upstream fix, run against its fixed commit:

```bash
uv run \
  --with 'harbor[modal] @ git+https://github.com/harbor-framework/harbor.git@16a510cecbda385d9d98b50d5096d7c36378f95a' \
  pytest -m integration tests/integration/harbor/test_harbor_modal_live.py -v
```

The verifier is deterministic, so this test does not require a judge model. It
uses the model endpoint directly; TITO streaming and accounting require a
separate trainer-integrated test.
