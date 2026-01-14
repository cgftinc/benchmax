# Search Environment

This environment provides capabilities for searching through a pre-indexed corpus using BM25 full-text search with optional metadata and filename filtering.

## Prerequisites

You need a running search API server with:
- A corpus already created and indexed
- API key for authentication
- Corpus ID

The search API should be accessible at a base URL (default: `http://localhost:3000`).

## Installation

```bash
pip install "benchmax"
```

## Usage

Initialize `SearchEnv` with required parameters:

```python
from benchmax.envs.search.search_env import SearchEnv

env = SearchEnv(
    api_key="your-api-key-here",
    corpus_id="your-corpus-id-here",
    base_url="http://localhost:3000"
)
```

## Available Tools

The environment provides one MCP tool for corpus searching:

### search_corpus
Searches the corpus using BM25 with optional filtering:
- **query** (required): Search query string
- **metadata** (optional): Dictionary of metadata filters (e.g., `{'ticker': 'DDOG', 'year': 2024}`)
- **filename** (optional): Filename filter - can be a simple substring (e.g., 'config') or regex pattern (e.g., '.*\\.json$')
- **limit** (optional): Maximum number of results to return (default 10)

Returns formatted search results with:
- Filename and BM25 score (or "filtered" if metadata filtering was used)
- Chunk content
- Associated metadata
- Total number of matching results

### Examples

**Basic search:**
```python
await env.run_tool(rollout_id, "search_corpus", query="PostgreSQL database")
```

**Search with metadata filter:**
```python
await env.run_tool(
    rollout_id,
    "search_corpus",
    query="earnings report",
    metadata={'ticker': 'DDOG', 'year': 2024}
)
```

**Search with filename filter (substring):**
```python
await env.run_tool(
    rollout_id,
    "search_corpus",
    query="configuration",
    filename="config"
)
```

**Search with filename filter (regex):**
```python
await env.run_tool(
    rollout_id,
    "search_corpus",
    query="configuration",
    filename=".*\\.json$"
)
```

## Reward Function

The task scores 1.0 only if the ground-truth string, after XML-entity unescaping and whitespace normalization, exactly matches the text inside the first `<answer>...</answer>` block (case-insensitive); otherwise it returns 0.0. If that block is missing or empty, the reward defaults to 0.0. This binary scheme forces the model to place a single, exact final answer inside the first answer tag while allowing any additional explanation outside it.

## Features

- BM25 full-text search on pre-indexed corpus
- Metadata filtering for precise result narrowing
- Filename filtering with support for both substring matching and regex patterns
- Configurable result limits
- Error handling for API failures
- Stateless environment design
