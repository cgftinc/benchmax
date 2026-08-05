# Get started with Harbor

This project is shaped for a Harbor package: Harbor resolves the task dataset,
creates the sandbox, runs the harness, and executes the verifier. Castform uploads
the environment bundle and supplies the model session used by the harness.

```bash
uv sync
uv run python main.py data --dataset <org/package>
uv run python main.py validate --dataset <org/package> \
  --modal-token-id <id> --modal-token-secret <secret>
uv run python main.py launch --dataset <org/package> \
  --modal-token-id <id> --modal-token-secret <secret>
```

If the packaged verifier needs credentials, inspect its configuration and pass
each requirement explicitly, for example
`--verifier-env OPENAI_API_KEY=<key>`. Do not read ambient environment variables
inside the environment constructor.

## Work with your agent

Ask your agent to inspect the task package, then start from the closest maintained
example under <https://github.com/castform-ai/benchmax/tree/main/examples>.

Use Harbor when the source is a Harbor package or its records describe executable
tasks with sandbox/build configuration, a named harness, and an in-sandbox
verifier. Use `BaseEnv` when the records are ordinary prompts/examples and your
Python environment owns the model loop and reward directly.

Keep the stock `TrialAgentConfig(name="mini-swe-agent")` when it fits. AIME shows
an offline-installed Mini-SWE variant for unreliable per-sandbox installation;
Harvey shows a task-specific native harness. Add custom harness code only for a
similarly concrete reason.

The script follows the maintained example shape:

- `_constructor_args(args)` builds explicit, serializable configuration once;
- the same dictionary constructs the local env and the uploaded bundle;
- Harbor data remains runtime-managed, so `upload_assets` gets no JSONL dataset;
- validation prints static, local, and remote warnings/errors before launch.

Harnesses may request `max_tokens` or `max_completion_tokens`, but validation warns
because Castform can clamp the effective cap. Do not set trainer-owned sampling
controls such as `temperature`, `top_p`, penalties, `seed`, or `stop`. A green
validation also requires real model calls, compatible tracked history, verifier
execution, and scorable rewards—not merely a completed process.

Run `uv run pytest tests` for the structural seed tests, then add task-specific
tests for required verifier environment, sandbox configuration, and reward output.
