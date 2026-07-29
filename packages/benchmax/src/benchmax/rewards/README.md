# rewards

`benchmax.rewards` contains reusable scoring primitives for `BaseEnv` and direct `Environment` implementations. Harbor environments normally use their harness verifier and RewardKit instead.

## deterministic helpers

deterministic helpers extract completion text and score properties such as citations, overlap, tool usage, and search budgets without making another model request.

## judges and rubrics

`Judge` describes the model endpoint used to evaluate an output. `Rubric` defines a named criterion, its polarity, and its allowed scores.

the scoring helpers can evaluate one rollout independently or compare a completed sibling group:

```python
from benchmax.auth import InjectedAuth
from benchmax.envs import BaseEnv
from benchmax.rewards import Judge, Rubric, score_rubrics


judge = Judge(
    model="judge-model",
    base_url="https://llm.castform.com/v1",
    auth=InjectedAuth("judge"),
)
rubrics = (
    Rubric("correctness", "the answer is correct"),
    Rubric("fabrication", "the answer invents facts", polarity="negative"),
)


class AnswerEnv(BaseEnv):
    reward_keys = ("rubric_correctness", "rubric_fabrication")

    async def compute_reward(self, rollout):
        return await score_rubrics(
            rollout.rollout_id,
            rollout.messages,
            ground_truth=rollout.example_args["answer"],
            rubrics=rubrics,
            question=rollout.example_args["question"],
            judge=judge,
        )
```

### authentication

use `InjectedAuth("judge")` when the judge runs through `llm.castform.com`. it lets the Castform runtime authenticate judge calls with the user's current Castform session, so the environment does not need a separate judge API key.

for a user-managed external endpoint, pass its API key explicitly with `StaticBearerAuth`:

```python
from benchmax.auth import StaticBearerAuth


judge = Judge(
    model="judge-model",
    base_url="https://models.example/v1",
    auth=StaticBearerAuth(judge_api_key),
)
```

the static key travels with `Judge` if the environment is bundled, so use a dedicated, scoped, revocable key. `InjectedAuth` is not a generic secret store for arbitrary providers; Castform binds it to its managed endpoint.

## ranking

ranking helpers compare outputs rather than assigning each output an isolated absolute score. anchors can provide stable reference points for those comparisons.

## adaptive rubrics

adaptive rubric helpers generate task-specific criteria and store them in a caller-owned `RubricCache`. the cache policy remains explicit rather than hidden in the environment runtime.

## diversity

diversity scoring can cluster sibling outputs with deterministic n-gram similarity or a configured model judge, then scale rewards based on how distinct the outputs are.

the exact examples and recommended compositions will be expanded alongside the benchmax example environments.
