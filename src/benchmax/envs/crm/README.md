# CRM Environment

This environment provides capabilities for interacting with Salesforce instances through the Salesforce API. 

This is based off [CRMArena Pro](https://github.com/SalesforceAIResearch/CRMArena) & supports both B2B and B2C configurations.

## Prerequisites

**Important**: Before using this environment, ensure you have:
- Python 3.12 or higher

## Installation

```bash
pip install "benchmax[crm,skypilot]"
```

Includes:
- simple-salesforce: For Salesforce API interactions
- python-dateutil: For date/time handling
- fastmcp: For MCP server functionality

## Usage

Use `CRMEnvLocal` to run the servers locally on the same machine as benchmax or use `CRMEnvSkypilot` to parallelize the servers across multiple nodes. 

## Configuration

The environment has built-in Salesforce configurations:

### B2B Configuration
Default configuration using B2B Salesforce instance. Used automatically when initializing CRMEnv.

### B2C Configuration
Alternative configuration for B2C use cases, can be enabled by passing "b2c" to get_mcp_config().

## Available Tools

The environment provides a comprehensive set of MCP tools for Salesforce interactions:

### Case Management
- `get_cases`: Retrieve cases based on various filtering criteria (dates, agents, statuses)
- `get_non_transferred_case_ids`: Get cases not transferred between agents in a period
- `get_agent_handled_cases_by_period`: Get number of cases handled by each agent
- `get_agent_transferred_cases_by_period`: Get number of cases transferred between agents
- `get_livechat_transcript_by_case_id`: Retrieve live chat transcripts for a case

### Agent Analysis
- `get_qualified_agent_ids_by_case_count`: Filter agents based on case handling count
- `get_agents_with_max_cases`: Find agents with most cases in a subset
- `get_agents_with_min_cases`: Find agents with fewest cases in a subset
- `calculate_average_handle_time`: Calculate average case handling time per agent

### Regional Analysis
- `get_shipping_state`: Add shipping state information to cases
- `calculate_region_average_closure_times`: Calculate average case closure times by region

### Time Period Management
- `get_start_date`: Calculate start date based on period and interval
- `get_period`: Get date range for named periods (months, quarters, seasons)
- `get_month_to_case_count`: Count cases created in each month

### Product and Issue Management
- `search_products`: Search for products by name/description
- `get_purchase_history`: Get purchase history for account/products
- `get_issues`: Retrieve list of issue records
- `get_issue_counts`: Get issue counts for products in a time period
- `get_order_item_ids_by_product`: Get order items for a product

### Knowledge Base
- `search_knowledge_articles`: Search knowledge articles by term

### Account Management
- `get_account_id_by_contact_id`: Get Account ID for a Contact

### Utility Functions
- `find_id_with_max_value`: Find IDs with maximum value in a dataset
- `find_id_with_min_value`: Find IDs with minimum value in a dataset
- `issue_soql_query`: Execute custom SOQL queries
- `issue_sosl_query`: Execute custom SOSL queries

## Features

- Seamless interaction with Salesforce instances
- Support for both B2B and B2C configurations
- Robust answer parsing and evaluation through fuzzy matching
- Standardized example preprocessing for dataset handling

## Reward Functions

The environment provides two reward metrics for evaluating model completions:

### Exact Match
- Uses IoU (Intersection over Union) score
- Compares completion tokens with ground truth tokens
- Perfect score (1.0) for exact matches
- Partial score based on token overlap
- Returns 0.0 if one set is empty while other isn't

### Fuzzy Match
- Uses F1 score for more lenient evaluation
- Handles variations in text formatting and word order
- Normalizes text by removing punctuation and articles
- Suitable for cases where exact matching is too strict

## MCP Server

The environment runs a Salesforce MCP server that handles API interactions. The server:
- Manages authentication with Salesforce
- Provides secure access to Salesforce operations
- Handles API rate limiting and session management


# CRMArena-Pro data
crmarena_pro_queries = `load_dataset`("Salesforce/CRMArenaPro", "CRMArenaPro")
b2b_schema = load_dataset("Salesforce/CRMArenaPro", "b2b_schema")
b2c_schema = load_dataset("Salesforce/CRMArenaPro", "b2c_schema")

TODO: Link to HuggingFace dataset