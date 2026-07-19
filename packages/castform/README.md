# Castform

Castform is the SDK and CLI for the Castform training platform. It provides
authentication, API clients, uploads and launches, hosted corpus/RAG tooling,
trace preprocessing, scaffolding, and platform configuration. It depends on the
platform-independent `benchmax` environment runtime.

## Installation

```bash
uv add castform
```

Install an optional feature only when a project uses it, for example
`castform[rag]`, `castform[traces]`, or a provider extra such as
`castform[chroma]`.

Python 3.12 is required.

Environment validation is script-owned: call `castform.validate_environment`
from the project and run that script. The Castform CLI handles platform
operations such as login, data/corpus workflows, launches, and run monitoring.

## License

Apache 2.0 © 2026 CGFT Inc.
