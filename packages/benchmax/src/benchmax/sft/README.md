# benchmax.sft

the validated `benchmax-sft-v1` supervised-finetuning dataset artifact. construction is
all-or-nothing: a dataset either satisfies the whole contract or raises with every ordered,
line-aware issue.

```python
from benchmax.sft import SftDataset, SftDatasetError

train = SftDataset.from_jsonl("train.jsonl")   # or SftDataset.from_rows(rows)
payload = train.to_jsonl_bytes()               # canonical, deterministic bytes
```

## row contract

one JSONL line per row. a row is a closed object: required `messages`, optional `tools`,
optional `metadata`.

```json
{
  "messages": [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4", "weight": 1}
  ],
  "tools": [],
  "metadata": {"id": "optional producer identity"}
}
```

- `system`/`user`: exactly `role` + non-empty string `content`.
- `assistant`: optional string-or-null `content`, optional non-empty `tool_calls`,
  optional integer `weight` `0 | 1` (omitted means `1`; `0` masks the turn from loss).
  each assistant turn needs non-empty content or a tool call; each row needs at least one
  assistant turn with effective weight `1`.
- `tool`: exactly `role` + string `content` + non-empty `tool_call_id`. every tool call
  gets exactly one result, in declaration order, before the next non-tool message.
- tool definitions/calls are OpenAI-style function shapes, validated deeply; called
  names must match a definition; `function.arguments` must decode as a JSON object.
- `metadata` is an open finite JSON object (canonical size ≤ 64 KiB; `_castform_*` keys
  are reserved for the runtime). rows are ≤ 1 MiB canonical, `messages` ≤ 1024 entries,
  `tools` ≤ 128 unique functions.

rejected outright: multimodal content parts, fractional weights, legacy
prompt/completion shapes, duplicate JSON keys, lone surrogates, non-finite numbers,
nesting deeper than 64 levels, and unknown keys anywhere outside `metadata` and
`parameters`.

## canonical bytes

`to_jsonl_bytes()` is deterministic for equivalent inputs: UTF-8 without a BOM,
`ensure_ascii=False`, sorted keys, compact separators, one trailing newline per row,
original row order.

## golden fixtures

`packages/benchmax/tests/fixtures/sft_v1/` is the cross-repository contract consumed by
the castform trainer: valid inputs with byte-exact canonical outputs, and invalid inputs
with exact ordered diagnostics in `expected_issues.json`. change them only with a format
version bump in mind.
