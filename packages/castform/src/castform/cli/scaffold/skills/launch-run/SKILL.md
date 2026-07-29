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

## Required ordering

Read the script and confirm that `launch()` does all of the following in order:

1. calls the same local `validate()` gate used during iteration;
2. stops if either sibling did not finish;
3. asks the human to confirm a credit-spending GPU launch;
4. builds one `Bundle` with `dump_bundle`;
5. passes that exact object to `upload_training_run(bundle=bundle, ...)`;
6. passes the returned paths to `TrainerClient.launch_training_run`.

The upload helper must not silently rebundle the environment.

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
    constructor_args=ENV_ARGS,
    pip_dependencies=RUNTIME_DEPENDENCIES,
)
```

List every external package imported while the environment, tools or rewards run.
Do not copy the whole project dependency list automatically: data-preparation and
development packages may not belong in the rollout image. BenchMax captures local
modules under the environment project automatically. Source from another project
must be explicit: use `local_modules=` to capture it, or list its installed
distribution in `pip_dependencies` to keep it as a remote reference.

For Harbor, add the selected provider extra explicitly, such as
`harbor[modal]>=0.18,<0.19` or `harbor[daytona]>=0.18,<0.19`.

## Launch configuration

Review `LAUNCH_CONFIG` in source. In particular:

- `max_rollout_len` is the whole-rollout token budget, not one response;
- keep trainer turn/tool limits compatible with the environment's own limits;
- start with modest epochs and judge the eval curve, not only train reward;
- use `TrainerClient.list_launch_args()` when you need the live accepted schema
  instead of guessing an argument name.

<!-- rag:start -->
For search environments, budget for repeated tool output across turns. Confirm
the rollout bundle includes the runtime search client but not large local corpus-
preparation dependencies unless the environment imports them.
<!-- rag:end -->

## SFT launch (and why it cannot land yet)

An env-less SFT project (see design-environment) launches with the same command
and the same confirmation discipline, but there is no bundle to build. Confirm
that its `launch()` does all of the following in order:

1. loads the train/eval pair once and runs the local `validate()` gate on it;
2. stops if any row carries a per-message `weight` and
   `LAUNCH_CONFIG["allow_experimental_weights"]` is not true;
3. stops if `castform.platform.client.SFT_LAUNCH_SUPPORTED` is `False`;
4. asks the human to confirm a credit-spending GPU launch;
5. calls `upload_sft_run(train=..., eval=..., run_name=...)`;
6. passes the returned paths to `TrainerClient.launch_sft_run`.

Steps 2 and 3 are independent gates, and both sit ahead of the upload on
purpose: a refused launch must not leave a dataset orphaned in storage behind an
API that cannot accept it.

**The live platform does not accept env-less SFT launch args yet.**
`SFT_LAUNCH_SUPPORTED` is `False` as of writing, so step 3 stops every SFT
launch before it uploads. The upload and launch path is fully implemented and
tested — it has nowhere to land until platform support ships. Do not tell a user
an SFT project is launch-ready today; track `SFT_LAUNCH_SUPPORTED` for when that
flips. Note that the flag gates the *scaffold*, not the SDK: code calling
`upload_sft_run` directly can still upload and then be rejected by the server at
launch. Keep the gate in the script.

On the wire, `launch_sft_run` nests `training_mode` and the dataset paths inside
`args`, which is where the platform reads them; a top-level `training_mode` is
silently ignored and would fall through to an RL run. `LAUNCH_CONFIG` carries
`training_mode` and an optional per-run `model` for that reason — they are wire
args, not local knobs — while `name`, `type` and `allow_experimental_weights`
are resolved locally and never forwarded.

## Credentials

Model and judge credentials must resolve per request (`InjectedAuth` for named
runtime providers); never freeze a temporary token in the bundle. Harbor sandbox
credentials are currently explicit constructor inputs. Review them before
bundling and limit their scope; a reference-injection design is deferred.

## Handoff

Record the run ID printed by the script, then load **view-progress**. If upload or
launch fails, preserve the error, correct the script or credentials, and rerun the
smallest failed stage. Never bypass a failed validation gate.
