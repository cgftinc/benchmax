# Wikipedia Environment

This environment provides capabilities for interacting with Wikipedia through its API, enabling search and article retrieval functionality.

## Prerequisites

No additional software installation is required. However, we suggest instantiating the `WikipediaEnv` class with API keys for higher rate limits.

## Installation

```bash
pip install "benchmax[wikipedia]"
```
Includes:
- fastmcp: For MCP server functionality

## Available Tools

The environment provides two MCP tools for Wikipedia interaction:

### search_wikipedia
Searches Wikipedia articles by keyword:
- Takes a search query and optional result limit
- Returns a list of relevant articles with titles and snippets
- Handles proper escaping and HTML cleanup

### get_wikipedia_article
Fetches the full plaintext of a Wikipedia article:
- Takes an exact article title as input
- Returns the complete article text (up to specified character limit)
- Handles redirects automatically
- Returns plain text with HTML markup removed

## Reward Function
The task scores 1.0 only if the ground-truth string, after XML-entity unescaping and whitespace normalization, exactly matches the text inside the first <answer>...</answer> block (case-insensitive); otherwise it returns 0.0. If that block is missing or empty, the reward defaults to 0.0. This binary scheme forces the model to place a single, exact final answer inside the first answer tag while allowing any additional explanation outside it.

## Features

- API key rotation support to handle rate limits
- HTML cleanup and entity unescaping
- Configurable result limits
- Error handling for API failures
- Support for article redirects
- Plain text extraction