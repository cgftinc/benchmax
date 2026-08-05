---
name: launch-run
description: Review, bundle, upload, and explicitly launch a Castform GPU training run from the project script.
---

# Launch a run

Use this only after **verify-environment** reports a believable green baseline.
Launching spends GPU credits. The workflow lives in `main.py`, not a CLI launch
command:

```bash
uv run python main.py launch
```

Do not pass `--yes` unless the user has already explicitly authorized the cost.
Never launch after `validate_environment` reports a static or runtime sampling
or history-contract error. Review output-cap warnings (`max_tokens` or
`max_completion_tokens`) and confirm any effective clamp is intentional.

## Required ordering

Accept user configuration as explicit `main.py` arguments, normalize it once in
`_constructor_args(args)`, and reuse that dictionary for local construction and
`dump_bundle`. Avoid ambient `os.environ` reads in environments, tools, rewards,
and harness configuration.

Read the script and confirm that the launch action does all of the following
in order:

1. builds one `Bundle` with `dump_bundle`;
2. passes that exact object to `upload_assets(bundle=bundle, ...)`;
3. validates the uploaded assets (locally and in the hosted sandbox) and stops
   on failure;
4. asks the human to confirm a credit-spending GPU launch;
5. passes the same uploaded paths to `TrainerClient.launch_training_run` — the
   run trains on precisely what was validated.

The upload helper must not silently rebundle the environment, and launch must
not re-upload.

Dataset upload is explicit and optional. Supply `train_dataset` and/or
`eval_dataset` only for splits Castform should upload. Omit them for data resolved
by the environment at runtime (for example Harbor- or Git-managed data). Do not
use an empty list as an omission sentinel: it uploads an empty JSONL file.

## Dependencies

`RUNTIME_DEPENDENCIES` is explicit and authoritative for the remote rollout
runtime:

```python
bundle = dump_bundle(
    CustomEnv,
    constructor_args=constructor_args,
    pip_dependencies=RUNTIME_DEPENDENCIES,
)
```

List every external package imported while the environment, tools or rewards run.
Do not copy the whole project dependency list automatically: data-preparation and
development packages may not belong in the rollout image. benchmax captures local
modules under the environment project automatically. Source from another project
must be explicit: use `local_modules=` to capture it, or list its installed
distribution in `pip_dependencies` to keep it as a remote reference.

For Harbor, add the selected provider extra explicitly, such as
`harbor[modal]>=0.18,<0.19` or `harbor[daytona]>=0.18,<0.19`.

## Launch configuration

Review `LAUNCH_CONFIG` in source. In particular:

- `max_context_tokens` is the whole-rollout prompt-plus-response token budget;
- keep trainer turn/tool limits compatible with the environment's own limits;
- start with modest epochs and judge the eval curve, not only train reward;
- use `TrainerClient.list_launch_args()` when you need the live accepted schema
  instead of guessing an argument name.

<!-- rag:start -->
For search environments, budget for repeated tool output across turns. Confirm
the rollout bundle includes the runtime search client but not large local corpus-
preparation dependencies unless the environment imports them.
<!-- rag:end -->

## Credentials

Use `InjectedAuth` for model and judge calls through the Castform LLM endpoint so the hosted runtime supplies the current Castform credential. User-managed external endpoints use explicit `StaticBearerAuth`. Harbor sandbox credentials are currently explicit constructor inputs. Review static credentials before bundling and limit their scope.

## Handoff

Record the run ID printed by the script, then load **view-progress**. If upload or
launch fails, preserve the error, correct the script or credentials, and rerun the
smallest failed stage. Never bypass a failed validation gate.
