# Envs

This directory contains:
```bash
├── postgres_search/  # Search/RAG env helpers
├── telestich/        # Example text-task env
├── types.py          # Shared types
└── base_env.py       # Base env class
```

Legacy bundled MCP-backed envs were removed from the base SDK. New projects should
define their env in a project-local `main.py` and use JSONL datasets; see
[how-to-extend-base-env.md](how-to-extend-base-env.md).
