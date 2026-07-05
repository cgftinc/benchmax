# Envs

This directory contains:
```bash
├── postgres_search/ # RAG search envs
├── telestich/      # Telestich poem env
├── types.py        # Shared types
└── base_env.py     # Base env class
```

To add a new environment, extend `BaseEnv`. [This guide](how-to-extend-base-env.md)
walks through the expected shape.
