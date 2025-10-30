# Math Environment

This environment provides capabilities for solving mathematical problems through a local calculator MCP server.

## Prerequisites

Before using this environment, ensure you have:
- Python 3.12 or later installed
- The `mcp-server-calculator` package installed

## Installation

```bash
pip install "benchmax[skypilot]"
```

Includes:
- fastmcp: For MCP server functionality that enables calculator operations

## Usage

Use `MathEnvLocal` to run the servers locally on the same machine as benchmax or use `MathEnvSkypilot` to parallelize the servers across multiple nodes. 

## Available Tools

The environment provides a calculator MCP tool through the server configuration:

### Calculator Tool
The calculator tool is provided through a local MCP server that:
- Handles mathematical computations
- Takes mathematical expressions as input
- Returns computed results
- Supports standard mathematical operations

## Reward Function 
Written in workdir/reward_fn.py so that the reward function can easily be calculated with the remote node.

The evaluator awards 1.0 when the ground-truth string, after case-insensitive comparison, whitespace normalization, and XML-entity unescaping—appears anywhere inside the first <answer>...</answer> block of the completion; otherwise the reward is 0.0. If that tag pair is missing or empty, the reward defaults to 0.0. This binary scheme incentivizes placing an exact, normalized final answer within the required XML tags.
