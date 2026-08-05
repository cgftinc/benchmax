---
name: design-environment
description: Design a benchmax environment, its ordered dataset, tools, and explicit reward shape for a Castform project.
---

# Design an environment

Before coding, inspect the maintained [examples](https://github.com/castform-ai/benchmax/tree/main/examples), choose the closest task shape, and follow its `README.md`, project structure, and `main.py`. The examples are the source of truth for current APIs and patterns; do not recreate an environment from a scaffold snippet.

Use `BaseEnv` for the standard OpenAI-compatible chat and tool loop. Use
`HarborEnv` for a Harbor dataset/package or tasks that ship their own instruction,
sandbox, and verifier. Start with `aime` for packaged tasks or `harvey` for a
custom harness. Do not infer Harbor from a judge, tool, or Dockerfile alone.
Extend `Environment` directly only for another genuinely different rollout loop.

Do not normalize every task into one project layout. Preserve the selected
example's division between `main.py`, environment, harness, data, and search
modules, changing only what the task requires.

Build each `Example` with a stable ID, normally
`canonical_example_id(payload)`. `prompt_messages` is the reserved BaseEnv payload
field; all other fields become `rollout.example_args`. Put system messages in
`prompt_messages` rather than ambient module state.

`Dataset` is an ordered base class, not a cleaning pipeline. The environment owns
the runtime representation and can store lightweight references in payloads.
Preparation, cleaning and QA generation belong in the project data script.

## Reward contract

- Return named reward components from successful individual and group reward hooks.
- Return finite numbers and make correctness the dominant signal.
- Let judge, model, tool and sandbox operational failures propagate through the
  typed runtime path. benchmax logs them and returns no rewards with a
  non-`finished` termination reason.
- Do not catch a judge failure and report it as a legitimate score.
- Programming, malformed-result and configuration errors should remain loud.

Override `compute_group_rewards` only when scoring genuinely depends on successful
siblings. Failed siblings are excluded from group-relative scoring, and a failed
group judge zeroes otherwise-successful siblings without cancelling the group.

## Optional tools

`BaseEnv` supplies no tools by default. A tool-using environment returns standard
OpenAI tool schemas from `list_tools` and dispatches them in `run_tool`:

```python
async def run_tool(self, rollout_id: str, tool_name: str, **tool_args):
    if tool_name != "lookup":
        raise ValueError(f"unknown tool: {tool_name}")
    return await self.lookup(tool_args["query"])
```

Keep clients pickle-safe. Use `InjectedAuth` for calls through the Castform LLM endpoint so Castform supplies the current session credential. Use explicit `StaticBearerAuth` for a user-managed external endpoint; never read Castform credentials from benchmax environment code.

## Model-request ownership

Treat model sampling as trainer-owned. A harness may request an output ceiling
with `max_tokens` or `max_completion_tokens`; static validation emits a warning
because Castform may clamp that ceiling to the remaining context budget. Do not
set `temperature`, `top_p`, `top_k`, penalties, `seed`, or `stop` in agent or
nested model kwargs. Static validation rejects them instead of allowing a later
training failure. It also rejects unsupported controls such as `n > 1`, forced
`tool_choice`, logprobs, and non-text response formats.

## Review before handoff

1. Test dataset identity and split ordering.
2. Unit-test empty, wrong, partial and correct completions against every reward key.
3. Exercise tool errors and judge errors and confirm zero rewards plus an explicit
   termination reason and log.
4. Load **verify-environment** and run the real two-sibling validation.
5. Record every remote runtime import in `RUNTIME_DEPENDENCIES` for **launch-run**.

For Harbor, copy the sandbox credential flow and provider dependency constraint
from the selected maintained example.
