# postgres-search

a retrieval environment library built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) for question answering against a provisioned corpora-service corpus.

## example task

a subclass supplies the question rows; the model searches the corpus, reasons over the retrieved chunks, then commits a cited answer:

```
question: Which region reported the highest rainfall in 2024?
tool: search("highest rainfall by region 2024") → chunks with source ids
assistant: <think>...</think>
<answer>
The Western Highlands recorded the highest rainfall in 2024 [Source: climate-summary-2024].
</answer>
```

rewards read only the final answer block: correctness is judged against the reference answer, and the inline `[Source: ...]` citations drive the retrieval and citation-precision components.

## use the environment

`postgres-search` intentionally ships without a corpus or question dataset, so it cannot validate or launch as a standalone example. run its contract tests with:

```bash
cd examples/postgres-search
uv run pytest tests
```

subclass `SearchEnv`, provide a configured search client and judge endpoint, then supply your own training and evaluation datasets.

```python
class MySearchEnv(SearchEnv):
    system_prompt = SearchEnv.render_system_prompt(...)

    async def create_dataset(...):
        return JsonlDataset(...)

    def __init__(self, ...):
        super().__init__(
            search=...,
            judge_base_url="https://llm.castform.com/v1",
            judge_model="gpt-5.4-mini",
        )
```

`SearchEnv` exposes a `search` tool and scores answer correctness, retrieval hits, citation precision, and answer length. `InjectedAuth("judge")` is the convenient default for `llm.castform.com`; use `StaticBearerAuth` when the judge endpoint requires your own bearer token.
