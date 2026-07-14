# Environments

```text
environment.py       Environment ABC and standard group executor
base/                Default chat/tool loop, JSONL loader, and authoring guide
dataset.py           Dataset protocol and frozen in-memory implementation
identity.py          Canonical hashing helper for JSON semantics
shared_types.py      Shared request, attempt, and outcome types
harbor/              Concrete optional adapter over native Harbor configs
postgres_search/     Search environment implementations
telestich/           Group-relative example environment
```

Most custom environments inherit [`BaseEnv`](base/env.py). It supplies the
conversation loop and optional tool dispatch. Subclasses own dataset semantics,
stable identity, and rewards.

Custom rollout loops can inherit `Environment` directly. `HarborEnv` follows
this path because Harbor owns the complete harness loop. Most Harbor users only
pass native configuration; see [harbor/README.md](harbor/README.md).

See the [BaseEnv authoring guide](base/README.md).
