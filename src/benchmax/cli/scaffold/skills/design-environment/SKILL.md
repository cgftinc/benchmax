---
name: design-environment
description: Design a castform RL environment — a BaseEnv subclass with tools and reward functions. Use when creating or editing main.py / the env for a training run.
---

# Design the environment

This skill is stage 1 of `castform setup → data → validate → launch`. Tailor the
runnable seed `main.py` that `castform setup` wrote; it should contain one
`BaseEnv` subclass defining the system prompt, tools, and rewards.

## Fast path (to a green baseline)

For the first green baseline, keep the env single-turn unless the task truly needs
tools. Fill in three things:

1. **`system_prompt`** — what the model is told it's doing.
2. **`compute_reward`** — how a rollout is scored. Make it **discriminating** (it
   must give different scores to better/worse answers); a reward that never varies
   gives training no gradient. Keep the primary task (usually correctness) dominant
   and gate secondary bonuses on it.
3. **Datasets** — tailor the seed rows with the **generate-data** skill.

Next, hand off to the next skill — **load it, don't just run its command**:
invoke **generate-data**, then invoke **verify-environment** before every
`castform validate`.

## Going deeper

### The BaseEnv shape

```python
from benchmax.envs.base_env import BaseEnv

class MyEnv(BaseEnv):
    system_prompt = "…"

    async def list_tools(self):
        return []                      # [] = single-turn, no tools

    async def run_tool(self, rollout_id, tool_name, **tool_args):
        ...                            # only reached if list_tools is non-empty

    async def compute_reward(self, rollout_id, messages, task, **kwargs):
        # messages = full transcript; task = the dataset row (prompt, ground_truth…).
        # Return a DISCRIMINATING dict[str, float] — see Reward rules below. A
        # simple reward: `correct = ground_truth in the model's final answer`.
        ...

    # optional: relative/ranking reward across a rollout group
    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        return [{"rank": 0.0} for _ in rollout_ids]   # one dict per rollout
```

`dataset_preprocess` is inherited: it turns a row into a prompt from the row's
`prompt` (or `messages` / `prompt_messages`) field, and exposes the whole row as
`task`. Override it only if your columns differ.

Put per-rollout configuration in `__init__`, not at module level: the sandbox
**unpickles** your env instance rather than re-importing `main.py`, so top-level code
doesn't run in the sandbox (it runs only at bundle time, on your machine).

### Reading the rollout (`messages` and `task`)

Both reward hooks are `async`. `messages` is the full transcript as a list of
`{"role", "content"}` dicts (OpenAI chat shape):

```python
[
    {"role": "system", "content": "…"},     # your system_prompt
    {"role": "user", "content": "…"},        # the dataset prompt
    {"role": "assistant", "content": "…"},   # the model's answer
    # multi-turn (tools) appends more assistant / tool messages here
]
```

- `role` is one of `system` / `user` / `assistant` / `tool`; `content` is a
  string. The model's output is the **`assistant`** turn(s).
- `task` is the **dataset row as a dict** (e.g. `{"prompt": …, "ground_truth": …}`),
  or `None` if the env grades without per-row data — read it defensively with
  `(task or {}).get("ground_truth")`.

Copy-paste — get the model's final text answer (inline this in `main.py`; there is
**no importable `last_answer`** helper, so don't `import` one):

```python
def last_answer(messages) -> str:
    """The model's final text answer (last assistant turn)."""
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("content"):
            return m["content"]
    return ""
```

To join *every* assistant turn instead (multi-turn rollouts), use the shipped
helper: `from benchmax.envs.reward_helpers import extract_completion_text`.

For answer-tagged tasks (especially RAG), score only the answer the model commits
inside `<answer>...</answer>`. Do **not** fall back to scoring the whole completion
when tags are missing, or the model can earn correctness from its reasoning without
ever answering. Prefer the last *closed* answer block so a self-correction wins
while a stray literal `<answer>` later in the prose can't hijack scoring:

```python
import re

_BLOCK = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.I | re.S)
_OPEN = re.compile(r"<answer\s*>", re.I)

def extract_answer(text: str) -> str:
    text = text or ""
    blocks = list(_BLOCK.finditer(text))
    if blocks:                       # last committed (closed) answer wins
        return blocks[-1].group(1).strip()
    opens = list(_OPEN.finditer(text))
    if not opens:                    # never committed → no answer (score 0)
        return ""
    return text[opens[-1].end():].strip()   # forgive a missing close tag
```

### Reward rules (these decide whether training works)

- Return **positive** scores. Negatives destabilise training.
- **Every component is summed** into one scalar — scale components so the sum
  reflects the priorities you want.
- Gate secondary components on the primary one. If `correctness == 0`, citation,
  style, brevity, and tool-use bonuses should normally be `0`; if correctness is
  partial, multiply bonuses by that value so they cannot trade off against being
  right.
- Keep it **discriminating**: a reward that returns the same value for every
  rollout gives no gradient. If `validate`'s `⚠ … constant` check fires (a `std`
  of `0`), the reward or the data needs work (see generate-data's difficulty-filter).
- For qualitative scoring, be **comparative**: judge against `ground_truth`, or
  use `compute_group_reward` to **rank** completions within the group. Ranking is
  much more stable than an absolute LLM-judge score. A finer absolute rubric often
  still collapses to the same 0/1 decisions, so it does **not** break GRPO ties; use
  a true pairwise/listwise ranking prompt when you need resolution within a group.
- Calibrate LLM judges on real answers before launch. Re-score a sample of
  `0.5`/partial outputs by hand or with a stricter rubric; verbose hedges and
  question restatements can look "partially correct" to a lenient judge and dull the
  gradient.
- `compute_group_reward` must return one `dict[str, float]` per rollout, all
  finite. Override it only when reward needs cross-rollout context.

### Tools / turns

- No tool need? Return `[]` from `list_tools` (single-turn). Don't add tools the
  task doesn't require.
- If you DO use tools, the env is multi-turn. Two contract rules the runtime
  enforces — break either and the rollout errors, not just the tool call:
  - `list_tools` returns **`ToolDefinition` dataclasses**
    (`from benchmax.envs.types import ToolDefinition`), **not** OpenAI
    `{"type": "function", "function": {…}}` dicts. The runtime reads `tool.name` /
    `tool.input_schema`, so a dict throws `'dict' object has no attribute 'name'`.
  - `run_tool` gets the model's call `arguments` spread as `**tool_args` and must
    **return a string** (the tool's result text). On bad input, return a guidance
    string — **don't raise** (an exception aborts the whole rollout).

  ```python
  from benchmax.envs.types import ToolDefinition

  async def list_tools(self):
      return [
          ToolDefinition(
              name="lookup",
              description="Look up the definition of a term.",
              input_schema={                       # JSON-schema for the args
                  "type": "object",
                  "properties": {"term": {"type": "string"}},
                  "required": ["term"],
              },
          )
      ]

  async def run_tool(self, rollout_id, tool_name, **tool_args):
      term = tool_args.get("term")
      if not term:                                 # guide the model, don't raise
          return "Error: `lookup` needs a `term` argument."
      return self._glossary.get(term, f"No entry for {term!r}.")
  ```

- Tools make the env multi-turn, so `max_turns` defaults to **4**, `max_tool_calls`
  to **8** — and the trainer ignores any `recommended_max_*` on the env, so a rollout
  that needs more is silently truncated. Set the real budget at **both** steps (else
  validate caps at 4/8 too): `castform validate --max-turns N --max-tool-calls N`
  (both settable here) and `castform launch --set max_turns=N` (at launch only
  `max_turns` is a documented `--set` knob — `max_tool_calls` isn't, it defaults to 8;
  run `castform launch --list-args` for the live set).
  <!-- rag:start -->
  - For a rag `SearchEnv`, match `MAX_SEARCH_CALLS`: each search is one turn + one
    tool call, and the answer is inline (one extra turn, NOT a tool call), so budget
    `--max-turns S+1 --max-tool-calls S` for `S` searches. Keep `MAX_SEARCH_CALLS` ≤ 8
    if you need it honored in training; the scaffold defaults below that cap. Note the
    limit in `main.py` so it isn't forgotten.
  <!-- rag:end -->

### RAG and traces environments

RAG and traces use the same loop as any custom env: one `main.py`, then validate
and launch. The differences are the prompt, tools, reward details, and data source
(covered by **generate-data**). If the user's request names a source, use that
funnel; otherwise confirm before pulling data.

<!-- rag:start -->
**Use the user's real corpus.** For "search over my handbook/docs/wiki", ask for a
local folder, an existing Castform corpus (`castform corpus list`), or a vector-DB
corpus. Stop if they have none. Never generate fake/demo source docs; "synthetic
dataset" means QA pairs generated from their real corpus.

#### RAG — search a corpus and cite sources

`castform setup --template rag` writes a `SearchEnv` with a search tool and inline
reward arithmetic in `compute_reward` — the audited 4-component shape, which is
also the `SearchEnv` library default (`answer_correctness` gate, ungated
`retrieval_hit`, gated `citation_precision` + deterministic `answer_length`).
Treat it as an auditable starting point, not a law. Before a real launch, inspect
transcripts with `castform validate --reward-audit` and adjust:

- **Answer extraction:** only score a committed `<answer>` block (the default
  does). Missing tags score as no answer, and a final answer block beats an
  earlier draft.
- **Citations:** the default matcher is **loose** (`canonicalize_source_id_loose`
  — lowercase, strip directory prefix + extension, so id-hash and title-path
  variants match alike). Real corpora may still need corpus-specific rules
  (several valid pages for the same fact, exact-path corpora): pass a custom
  matcher — `score_citations(answer, chunks, canonicalize=lambda s: ...)` — or
  use the strict `canonicalize_source_id` for exact-path matching. Note the free
  helpers (`score_citations` etc.) default to the strict matcher; only the env
  defaults to loose.
- **Correctness gate:** multiply citation, brevity, and style bonuses by
  correctness so a wrong answer cannot bank source-format rewards (the default
  gates `citation_precision` and `answer_length` this way).
- **Retrieval signal (ungated):** unlike the format bonuses, the small
  `retrieval_hit` term stays **ungated** — citing a gold chunk is worth rewarding
  even when the final answer is wrong, so the model keeps learning to search.
  It's an answer-side proxy (gold sources cited in `<answer>`, not raw tool
  traffic); `castform validate`'s RAG probe reports true retrieval **gold-hit@k**,
  so you can read retrieval quality apart from answer correctness.
- **Conciseness:** the default brevity term is the deterministic `answer_length`
  (a prose-length cap, `ANSWER_LENGTH_CAP`) — dense and free. An LLM conciseness
  judge (`judge_answer_quality` + `CONCISENESS_RUBRIC`) is opt-in and may be
  sparse and expensive.
- **Search shaping:** the default reward has **no** search-efficiency bonus — a
  "fewer searches is better" term can fight multi-hop exploration. The hard
  `MAX_SEARCH_CALLS` cap covers safety; opt back in with
  `score_search_efficiency` and keep it small if it under-explores.
- **Tool-output budget:** large search responses across many turns count toward
  `max_rollout_len`. If you let each search return thousands of characters, lower
  `MAX_SEARCH_CALLS`, trim per-chunk bodies, or raise `max_rollout_len` after
  checking `castform launch --list-args`.

The template searches a **local CGFT corpus** (`PostgresSearch`, BM25). To search a
**provider** corpus, swap the `search=` client. Provider keys + resource ids come
from `DATA_*` env vars — and the sandbox does **not** see them, so read them at
**module load** (bundle time, on your machine) and bake them in; never read
`DATA_*` inside `__init__` (that runs in the sandbox → `KeyError`) and never write a
literal key in the file (it gets bundled and uploaded):

```python
import os
from benchmax.rag.corpus.turbopuffer.search import TpufSearch
from benchmax.rag.corpus.embed import platform_embed_fn

NAMESPACE = os.environ["DATA_namespace"]          # read at bundle time, baked in
REGION = os.environ.get("DATA_region", "aws-us-east-1")
_API_KEY = os.environ["DATA_api_key"]

class CustomSearchEnv(SearchEnv):
    def __init__(self, **kwargs):
        super().__init__(
            search=TpufSearch(
                namespace=NAMESPACE,
                region=REGION,
                token_provider=(lambda: _API_KEY),   # closes over the baked key
                embed_fn=platform_embed_fn(),         # vector/hybrid — see below
            ),
            judge_base_url=config.llm_url(),
            judge_model="gpt-5.4-mini",
            **kwargs,
        )
```

Pinecone → `PineconeSearch(index_name=…, index_host=…, token_provider=…)`;
Chroma → `ChromaSearch(collection_name=…, tenant=…, database=…, token_provider=…)`.
(`config.*` and `platform_embed_fn` are fine inside `__init__` — they resolve in the
sandbox against the env's own domain.)

**Ship the provider SDK to the sandbox — REQUIRED.** The rollout sandbox bundles only
`main.py` + benchmax; the provider SDK (`turbopuffer` / `pinecone` / `chromadb`) is NOT
there. If it's missing, the search client's import fails in the sandbox and gets
swallowed into an all-zero **hollow green** (the `pip install castform[<provider>]` you
did for `qa-gen` only fixes your *local* machine). Two ways to inject it — both compose,
so you don't memorize package names:

- **`--provider <name>` on validate/launch** (foolproof) — injects the right SDK,
  including chroma's un-guessable `snowballstemmer`:

  ```bash
  castform validate --provider turbopuffer --examples 2   # same flag on launch
  ```

- **The `PIP_DEPENDENCIES` slot in `main.py`** (declare once) — the rag scaffold's
  `CustomSearchEnv` ships an empty `PIP_DEPENDENCIES: list[str] = []`; fill it when you
  swap the `search=` client and it travels with the env, applied on every validate **and**
  launch with no flag:

  ```python
  class CustomSearchEnv(SearchEnv):
      PIP_DEPENDENCIES = ["turbopuffer>=1.16.2"]   # chroma: ["chromadb>=1.0.0", "snowballstemmer>=2.2.0"]
  ```

`--pip <dep>` (repeatable) is the manual override for anything neither covers.

**Does it actually retrieve?** A green `validate` can hide an empty search (the tool
swallows errors into a string — see verify-environment). The `embed_fn` decides it:
- **turbopuffer**: lexical/BM25 works with **no** `embed_fn`. Vector/hybrid
  **REQUIRES** `embed_fn=platform_embed_fn()` (tpuf has no server-side embed).
- **pinecone**: uses its hosted `multilingual-e5-large` automatically — works **only
  if the index was built with that model**. Built with another? pass
  `embed_fn=platform_embed_fn()` so query + index embeddings match.
- **chroma**: a Chroma Cloud collection (server-side EF) or a BM25 collection works
  as-is; a bare collection raises `LocalEmbeddingDownloadDisallowedError` — pass
  `embed_fn=platform_embed_fn()`. A **BM25** chroma collection also needs the stemmer
  in the sandbox — `--provider chroma` injects it (`chromadb` + `snowballstemmer`),
  the dep you can't guess.

`platform_embed_fn()` is `text-embedding-3-large` over the platform `/v1/embeddings`;
it inherits the env's domain in the sandbox. **Prefer the no-`embed_fn` lexical path when
your corpus is full-text-indexed** (turbopuffer/chroma BM25) — it needs nothing extra and
is the most robust. `platform_embed_fn` requires a recent benchmax **on the sandbox
image**; if `validate` fails to unpickle with `No module named 'benchmax.rag.corpus.embed'`,
the deployed image predates it — fall back to the lexical path (no `embed_fn`).
<!-- rag:end -->

#### Traces — learn the agent's task from its recorded traces

`castform data traces` (generate-data) pulls + shapes Braintrust traces into
`{prompt_messages, ground_truth, init_rollout_args}` rows and prints the **detected
system prompt + tools**. Set `system_prompt` to the agent's task (the detected one is
a good start) and author the env to fit the rows — three things to get right:

- **`ground_truth` is the recorded next assistant turn as a DICT**
  (`{"role","content","tool_calls"}`), not a string — a `ground_truth in answer`
  reward breaks. Read `gt.get("content")` and `gt.get("tool_calls")` and score
  comparatively (text overlap vs `content`; the tool the model picked vs the
  `tool_calls` it should have). Rank within a group for a stabler signal.
- **`prompt_messages` carry OpenAI tool fields** (`tool_calls`/`tool_call_id`/`name`)
  and `role:"tool"` turns. Override `dataset_preprocess` to **flatten** them into a
  clean chat prefix (e.g. render a tool call as a text line, a tool result as
  `[tool result …]`) rather than passing the raw turns through. The override is a
  **`@classmethod`** that returns an `Example` — not an instance method, and not a plain
  dict (that's the usual first miss). Build the return with `make_example`
  (`from benchmax.envs.example_id import make_example`); it returns the `Example` TypedDict
  (`benchmax.envs.types`) with the group id pre-computed:

  ```python
  @classmethod
  def dataset_preprocess(cls, row, **_) -> "Example":      # Example: benchmax.envs.types
      flat = _flatten(row["prompt_messages"])              # your tool-field flattener
      return make_example(
          prompt_messages=flat,
          task={"ground_truth": row["ground_truth"]},
          system_prompt=cls.system_prompt,
      )
  ```
- **Do NOT register the detected tools as live `list_tools`** unless you have a real
  backend for `run_tool` — a fake tool layer hollow-greens (every call errors). For
  trace imitation, keep it single-turn (`list_tools` → `[]`) and have the model
  *declare* its action (e.g. a `TOOL: <name>` line) that `compute_reward` scores
  against the recorded `tool_calls`.

### External-service envs (advanced)

If the env needs a separate service (a game/sim like Showdown), that service must
be reachable during rollout. Get a single-process env working first and add
external service deployment only after the reward loop is proven.

### Dependencies

Imports beyond benchmax must be bundled at launch. For PyPI deps (all merged,
de-duped): a `PIP_DEPENDENCIES = [...]` class attr on your env (declared once, travels
with it) and `--pip <pkg>` on validate/launch (or `pip_dependencies=[…]` via the SDK).
<!-- rag:start -->
For a known RAG provider's SDK, `--provider <name>` injects it as a third channel
(merged with the others).
<!-- rag:end -->
Local files are bundled from `main.py` automatically by `castform launch` (pass
`local_modules=[mod]` if calling the SDK directly).
