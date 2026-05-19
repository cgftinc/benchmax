<picture>
  <img alt="Benchmax" src="./static/benchmax.png"  width="full">
</picture>

## benchmax: Framework-Agnostic RL Environments for LLM Fine-Tuning
*A lightweight, training-framework agnostic library for defining, running, and parallelizing environments, to fine-tune OSS LLMs with reinforcement learning.*
<div align="center">
</div>
<div id="badges" align="center">
  <a href="https://castform.com">
    <img src="https://img.shields.io/badge/castform.com-blue?style=for-the-badge" alt="Website"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
</div>

## 📌 News

- **[29 Oct 2025]** 🎉 Added support for easy multi-node parallelization across all major cloud providers using [SkyPilot](https://github.com/skypilot-org/skypilot)

## ℹ️ Overview

`benchmax` comes with:

- A collection of ready-to-use reinforcement learning (RL) environments for LLM fine-tuning ranging from multi-hop search to spreadsheet manipulation to CRM agents
- An easy way to define, compose, and parallelize your own environments, including leveraging the existing ecosystem of MCP servers
- Trainer-agnostic by design — `BaseEnv` exposes a small async interface (`list_tools`, `run_tool`, `compute_reward`, plus optional rollout lifecycle hooks) that any rollout loop can drive
- Optional batteries-included add-ons: synthetic RAG dataset generation (`benchmax[rag]`), agent trace import (`benchmax[traces]`), and clients for the Castform training platform (`benchmax.platform`)

Define your environment as:

1. A **toolset** (LLM calls, external APIs, calculators, MCPs, etc.).
2. **Output parsing** logic to extract structured observations.
3. **Reward functions** to score model outputs.

Rollout management, parallel execution, etc. comes out of the box.

⭐ Star our repository to show your support!

## 💡 Core Features

**Built-in examples & templates**

Get started with ready to use recipes, from Wikipedia search to spreadsheet manipulation. Easy to copy, customize, and extend. And yes, more are on the way.

**MCP support**

Tap into the growing MCP ecosystem and integrate them as tools within your environments.

**Multi-node parallel execution**

Multi-node parallelization enabled out of the box with state isolation across roll-outs (e.g. editing files on filesystem, etc.).


## 🌐 Creating Environments

### What is an environment?

An environment consists of:

- A list of tools that an LLM can call
- A list of reward functions that evaluate the quality & correctness of the model's final output.

We also support MCP servers natively, allowing you to easily leverage the many servers built by the community.

### Pre-built environments

Ready-to-use environments with pre-configured tools and reward functions.

- [CRM](/src/benchmax/envs/crm/README.md)
- [Excel](/src/benchmax/envs/excel/README.md)
- [Math](/src/benchmax/envs/math/README.md)
- [Wikipedia](/src/benchmax/envs/wikipedia/README.md)
- [PostgreSQL search](/src/benchmax/envs/postgres_search/) (`benchmax[rag]`)

### How do I create a custom environment?

1. [With existing MCP servers](/src/benchmax/envs/mcp/README.md) (Built-in support for multi-node parallelization)

2. [Extend BaseEnv](/src/benchmax/envs/README.md)

### How about more complex environments?

- Check out our excel spreadsheet RL environment: `benchmax.envs.excel.excel_env.ExcelEnv`

### I want a specific environment

Open an issue and tag us & we will look into building you one!

---

## 🎯 Motivation

- **Modularity and Simplicity**:

    We set out to build a lightweight, modular system for defining RL environments—breaking them down into simple, composable parts: tools, tool output parsing, and reward functions.

    The goal's to make it easy for software engineers to build and experiment with RL environments without needing deep RL expertise.

- **Task Recipes and Ideas**:

    We want `benchmax` to be a living library of reusable, RL-compatible task recipes, ready to inspire and extend beyond the usual suspects like math and coding. We aim to support more real-world workflows, including open-ended and long-horizon tasks.

- **Parallelization and Cloud Compatibility**:
    - Enable efficient parallelization with maintained statefulness between rollouts.
    - Facilitate easy deployment and scalability in cloud environments.

- **MCP as a first class citizen**:

    There has been an explosion of MCP servers/tools built out for use-cases ranging from browser use to excel to game creation. `benchmax` allows folks to leverage and compose these existing MCP servers to build environments integrated with real world systems e.g. excel


## 🤝 Contributing

We welcome new environment recipes and bug reports!

⭐ Star our repository to show your support!

## 📦 Add-ons

In addition to the core env library, `benchmax` ships several optional
modules behind extras:

| Extra | Module | Purpose |
|---|---|---|
| `benchmax[rag]` | `benchmax.rag.*` | Markdown chunking, corpus indexing (Postgres / Chroma / Pinecone / Turbopuffer), synthetic QA dataset generation, RAG-specific reward rubrics |
| `benchmax[traces]` | `benchmax.traces` | Agentic trace import (Braintrust today, Langfuse coming) and provider-agnostic processing pipeline |
| `benchmax[chroma]` / `[pinecone]` / `[turbopuffer]` | `benchmax.rag.corpus.*` | Corpus-backend pins (combine with `[rag]`) |
| _(core)_ | `benchmax.platform` | HTTP clients for the Castform platform — storage uploads, training-job launch, rollout server. Used both internally by `benchmax.rag` and by the high-level [`castform-sdk`](https://pypi.org/project/castform-sdk/). |

All platform URLs derive from `CASTFORM_BASE_DOMAIN` (default
`castform.com`) with per-component overrides; see
[`benchmax.config`](src/benchmax/config.py).

## 📜 License

Apache 2.0 © 2026 Castform
