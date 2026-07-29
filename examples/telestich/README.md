# telestich

a group-scored poetry environment built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) where the last letter of each line spells a hidden word.

## example task

each dataset row is a poem request naming a hidden word. the poem needs one line per letter, every line must end on a real word, and the hidden word itself may never appear in the poem.

```
prompt: write a telestich hiding nah
```

the model can draft, then call the `feedback` tool (up to three times) with the draft and the word to get deterministic notes on line count, wrong ending letters, filler endings, and leaked hidden words:

```
feedback(poem="A quiet thought drifts in with the rain\n...", word="nah")
→ "line 2 ends in 'r' but needs 'a'"
```

it then commits the final poem in answer tags:

```
<answer>
A quiet thought drifts in with the rain
I turn it over, looking for a small idea
By morning, I follow a calmer path
</answer>
```

reading the last letters top to bottom (rain → n, idea → a, path → h) spells the hidden word.

## launch training

```bash
cd examples/telestich
uv run python main.py launch

# if iterating on the env, validate first
uv run python main.py validate
```

launch splits the 719 committed examples into curriculum-ordered training data and an evaluation holdout, uploads the environment and dataset, validates them, then asks for confirmation before spending credits (pass `--yes` to skip).

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end.

## environment

```python
class TelestichEnv(BaseEnv):
    async def create_dataset(...):
        return JsonlDataset(...)

    async def list_tools(...):
        return [feedback]

    async def run_tool(...):
        return deterministic_poem_feedback(...)

    async def compute_group_rewards(...):
        return score_complete_sibling_group(...)
```

the deterministic gate checks the hidden word, line count, valid ending words, and cheating. correct poems are ranked against acceptable and great reference anchors by an llm judge; rhyme, sibling diversity, and concise tool use add smaller quality-scaled bonuses. near-complete poems receive a small partial reward.

the judge uses `InjectedAuth("judge")` with `llm.castform.com`, which lets castform supply the active credential. use `StaticBearerAuth` for a third-party endpoint that requires your own bearer token.
