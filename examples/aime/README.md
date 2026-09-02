# aime

a harbor environment built with [`HarborEnv`](../../packages/benchmax/src/benchmax/envs/harbor/README.md) where an offline-installed mini-swe agent solves AIME competition math inside Modal or prepared Cloudflare sandboxes.

## example task

each trial drops the agent into a sandbox with an AIME problem; the verifier scores the final answer all-or-nothing:

```
task: Find the number of ordered pairs (a, b) of positive integers with
      a + b = 1000 where neither a nor b has a zero digit.
agent: runs shell commands in the sandbox, e.g. a python one-liner to
       enumerate the pairs
answer: 738
```

## launch training

### Modal

```bash
cd examples/aime

# extract modal credentials from the castform profile
export MODAL_TOKEN_ID=$(MODAL_PROFILE=castform uv run python -c "from modal.config import config; print(config['token_id'])")
export MODAL_TOKEN_SECRET=$(MODAL_PROFILE=castform uv run python -c "from modal.config import config; print(config['token_secret'])")
echo "modal token id: $MODAL_TOKEN_ID (secret: ${#MODAL_TOKEN_SECRET} chars)"

uv run python main.py launch --modal-token-id $MODAL_TOKEN_ID --modal-token-secret $MODAL_TOKEN_SECRET

# if iterating on the env, validate first
uv run python main.py validate --modal-token-id $MODAL_TOKEN_ID --modal-token-secret $MODAL_TOKEN_SECRET
```

### Cloudflare Standard-2

All 60 tasks in `aime/aime@latest` have the same Dockerfile-only image, so the
Cloudflare option is a prepared-image deployment rather than a dynamic builder.
Deploy [`cloudflare/`](cloudflare/) once, then use the returned Worker URL and
the bearer key supplied through `SANDBOX_API_KEY`. Docker must be running, and
Wrangler must be authenticated for the target account (for automation, export
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`):

```bash
cd examples/aime/cloudflare
npm install
# This creates and immediately deploys a Worker version with the secret.
npx wrangler secret put SANDBOX_API_KEY
# The explicit deploy builds and pushes the prepared container image; Wrangler
# preserves the secret set above.
npm run deploy

cd ..
export CLOUDFLARE_SANDBOX_API_URL=https://benchmax-aime-sandbox.<subdomain>.workers.dev
export CLOUDFLARE_SANDBOX_API_KEY=<same-bearer-key>

uv run python main.py validate --sandbox-provider cloudflare
uv run python main.py launch --sandbox-provider cloudflare
```

The Worker uses Standard-2 containers, disables the warm pool, and sleeps idle
containers after 30 minutes. The adapter verifies every task Dockerfile before
starting a prepared container; a changed image or extra build-context file
fails closed.

the dataset (aime/aime@latest) resolves through harbor at trainer runtime, so launch only uploads the environment bundle, validates it, then asks for confirmation before spending credits (pass `--yes` to skip). credentials for the selected sandbox provider are bundled into the constructor args.

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end. both locations run real trials with the selected sandbox provider.

## environment

```python
class AimeMiniSweHarborEnv(HarborEnv):
    def __init__(..., sandbox_provider="modal", harness=None):
        super().__init__(
            dataset=DatasetConfig(name="aime/aime", ref="latest"),
            trial=HarborTrialTemplate(
                agent=harness or mini_swe_harness(),
                environment=(
                    TrialEnvironmentConfig(type=EnvironmentType.MODAL)
                    if sandbox_provider == "modal"
                    else TrialEnvironmentConfig(
                        import_path="cloudflare_environment:AimeCloudflareEnvironment"
                    )
                ),
                verifier=TrialVerifierConfig(),
            ),
            sandbox_credentials=...,
        )
```

harbor resolves the dataset, runs each rollout as a sandboxed trial, and settles the reward with the dataset's verifier.

## why the custom harness

this is operational hardening, not an AIME requirement. Harbor's stock mini-swe installer ran apt and PyPI setup in every fresh sandbox, taking 40–90 seconds and sometimes exceeding its setup timeout. the harness in [`harness/`](harness/) instead prefetches wheels on the trial host and installs them offline. use `TrialAgentConfig(name="mini-swe-agent")` when stock installation is reliable; pass a `BundledHarborAgent` only when a custom harness is actually needed:

```python
env = AimeMiniSweHarborEnv(
    sandbox_credentials=...,
    harness=BundledHarborAgent(
        config=TrialAgentConfig(import_path="my_agent:MyAgent"),
        source=BundledAgentSource.from_directory(Path("my-harness"), files=("my_agent.py",)),
    ),
)
```
