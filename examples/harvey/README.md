# harvey

a harbor environment built with [`HarborEnv`](../../packages/benchmax/src/benchmax/envs/harbor/README.md) that trains on [`harveyai/lab@latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest) legal-work tasks using harvey's native harness loop in modal sandboxes.

## example task

each trial gives the agent a legal task over a set of documents; the dataset's rubric judge scores the deliverable:

```
task: Review the documents in /workspace/documents and draft the requested
      analysis.
agent: harvey's harness loop reads the documents with tools and writes the
       deliverable to /workspace/output
judge: the dataset's rubric judge scores the output
```

## launch training

```bash
cd examples/harvey

# extract modal credentials from the castform profile
export MODAL_TOKEN_ID=$(MODAL_PROFILE=castform uv run python -c "from modal.config import config; print(config['token_id'])")
export MODAL_TOKEN_SECRET=$(MODAL_PROFILE=castform uv run python -c "from modal.config import config; print(config['token_secret'])")
echo "modal token id: $MODAL_TOKEN_ID (secret: ${#MODAL_TOKEN_SECRET} chars)"

uv run python main.py launch \
  --modal-token-id $MODAL_TOKEN_ID \
  --modal-token-secret $MODAL_TOKEN_SECRET \
  --judge-provider anthropic \
  --judge-model 'anthropic/claude-sonnet-4-6' \
  --judge-api-key "$ANTHROPIC_API_KEY"

# no anthropic key? use the gpt judge through the castform endpoint that
# matches your active profile (llm.castform.dev for staging, llm.castform.com
# for prod):
#
# uv run python main.py launch \
#   --modal-token-id $MODAL_TOKEN_ID \
#   --modal-token-secret $MODAL_TOKEN_SECRET \
#   --judge-provider openai \
#   --judge-model 'openai/gpt-5.4-nano' \
#   --judge-api-key "$CASTFORM_API_KEY" \
#   --judge-base-url 'https://llm.castform.dev/v1'

# if iterating on the env, validate first: replace `launch` with `validate` above
```

the dataset resolves through harbor at trainer runtime, so launch only uploads the environment bundle, validates it, then asks for confirmation before spending credits (pass `--yes` to skip). all credentials are mandatory arguments and are bundled into the constructor args so trainer-side trials can reach modal and the judge.

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end. both locations run real modal sandbox trials.

## collect AutoCompact SFT data

`collect` runs only the Harbor training split. An online judge may replace the
base model's proposed action with `compact()`, repair its summary, and repair
the first continuation. Every accepted compaction event becomes three
call-level records; ordinary turns are retained in the raw trajectory but are
not exported for SFT.

```bash
uv run python main.py collect --yes \
  --output-dir ./harvey-autocompact-data \
  --model Qwen/Qwen3.5-35B-A3B \
  --modal-token-id "$MODAL_TOKEN_ID" \
  --modal-token-secret "$MODAL_TOKEN_SECRET" \
  --judge-provider anthropic \
  --judge-model 'anthropic/claude-sonnet-4-6' \
  --judge-api-key "$ANTHROPIC_API_KEY"
```

Use `--max-examples 1` for a smoke run and `--resume` to skip completed
per-rollout shards. The output includes raw trajectories, deterministic
`train.jsonl`/`eval.jsonl`, passed-only views, and `manifest.json`. The split
is by Harvey task id, so a compaction triplet can never cross splits. The
separate 126-task Harbor evaluation split is not read.

Launch SFT from an existing collection with the same credentials:

```bash
uv run python main.py launch-sft --yes \
  --output-dir ./harvey-autocompact-data \
  --modal-token-id "$MODAL_TOKEN_ID" \
  --modal-token-secret "$MODAL_TOKEN_SECRET" \
  --judge-provider anthropic \
  --judge-model 'anthropic/claude-sonnet-4-6' \
  --judge-api-key "$ANTHROPIC_API_KEY"
```

SFT uses the uploaded call-level JSONL directly. Prompt assistant messages are
loss-masked; only the single assistant completion in each record contributes
to loss.

## environment

```python
class HarveyLabHarborEnv(HarborEnv):
    def __init__(..., harness=None):
        super().__init__(
            dataset=DatasetConfig(name="harveyai/lab", ref="latest"),
            trial=HarborTrialTemplate(
                agent=harness or harvey_harness(),
                environment=TrialEnvironmentConfig(type=EnvironmentType.MODAL),
                verifier=TrialVerifierConfig(env=verifier_env),
            ),
            sandbox_credentials=ModalCredentials(...),
        )
```

the verifier env carries the judge credentials and model. harbor resolves the dataset, runs each rollout as a sandboxed trial, and settles the reward with the dataset's judge.

## harness

the agent harness lives in [`harness/`](harness/): harvey's native loop adapted to run inside a harbor environment. at launch, `main.py` sparse-clones harvey's LAB harness tree (a pinned ref; override with `HARBOR_HARVEY_GIT_URL` / `HARBOR_HARVEY_GIT_REF`) and captures it into the bundle (~0.7 MB), so authoring needs git and github access once but trial hosts need neither.

the harness is the default, not a requirement. pass any `BundledHarborAgent` as `harness=` to run a different agent loop against the same dataset and judge:

```python
env = HarveyLabHarborEnv(
    ...,
    harness=BundledHarborAgent(
        config=TrialAgentConfig(import_path="my_agent:MyAgent"),
        source=BundledAgentSource.from_directory(Path("my-harness"), files=("my_agent.py",)),
    ),
)
```
