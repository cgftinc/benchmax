# Envs

This directory contains:
```bash
├── crm/            # Salesforce env (extends MCP)
├── excel/          # Excel env (extends MCP)
├── math/           # Math env (extends MCP)
├── mcp/            # MCP env class (extends BaseEnv)
├── wikipedia/      # Wikipedia env (extends BaseEnv)
├── types.py        # Shared types
└── base_env.py     # Base env class
```

Pre-built envs like CRM, Excel, Math uses MCP which has built-in multi-node parallelization are ready to use out of the box. To learn how to create your own parallelized MCP env, check out [this guide here](mcp/README.md)

If you want to manually extend `BaseEnv` (no multi-node support), you can check out the Wikipedia env or [follow this guide](how-to-extend-base-env.md).

