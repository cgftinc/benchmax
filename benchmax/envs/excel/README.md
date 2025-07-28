# Excel Environment

This environment provides capabilities for interacting with Excel files through either LibreOffice (Linux) or Microsoft Excel (Windows/macOS).

This is based off the [SpreadsheetBench Benchmark](https://spreadsheetbench.github.io/)

## Prerequisites

**Important**: Before using this environment, ensure you have the appropriate spreadsheet application installed:
- **Linux**: LibreOffice must be installed
```bash
sudo apt install libreoffice
```
- **Windows/macOS**: Microsoft Excel must be installed

## Installation

### Linux
```bash
pip install "benchmax[excel-linux]"
```
Includes:
- openpyxl: For Excel file manipulation
- fastmcp: For MCP server functionality

### Windows/macOS
```bash
pip install "benchmax[excel]"
```
Includes:
- openpyxl: For Excel file manipulation
- xlwings: For direct Excel application interaction
- fastmcp: For MCP server functionality

## Available Tool

The environment provides a single MCP tool for Excel manipulation:

### run_excel_code
Executes Python code that uses openpyxl to manipulate Excel files. The tool:
- Takes Python code and an output Excel path as input
- Runs the code in a controlled environment
- Returns a string representation of the modified Excel file
    - Preserves spreadsheet formatting (colors, fonts, styles)
    - Handles both cell-level and sheet-level operations

## Reward Functions

Reward functions measure how well a generated spreadsheet matches the expected output.

### Default Reward Function

The built-in reward function compares the output spreadsheet against a ground truth using only the cells specified in the task.

- If all the values in the relevant cells are correct, the reward is **1.0**
- If there are any mismatches, the reward is **0.0**

### Comparison Strategy

- Compares evaluated values, not formulas (e.g., it checks the result `10`, not the formula `=5+5`)
- Only compares the cells within the defined answer range
- Supports multiple cell types like numbers, strings, times, and dates
- Can optionally check visual formatting, such as:
  - Fill color (background)
  - Font color

### Error Handling

If the comparison fails due to a missing file, invalid cell reference, or formatting error, the reward function will return **0.0** and log a helpful message for debugging.

### Outcome

- **1.0** score if the generated spreadsheet is fully correct
- **0.0** if any discrepancy is found