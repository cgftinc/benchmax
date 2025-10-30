from fastmcp import FastMCP
import subprocess
import sys

try:
    from excel_utils import excel_to_str_repr
except Exception:
    # Added except local import for unit testing purposes
    from .excel_utils import excel_to_str_repr

mcp = FastMCP(
    name="ExcelCodeRunner",
    instructions="This server provides a tool for running Python code to manipulate Excel files.",
)

WHITE_LIKE_COLORS = [
    "00000000",
    "FFFFFFFF",
    "FFFFFF00",
]


def run_excel_code_impl(python_code: str, output_excel_path: str) -> str:
    """
    Run Python code which should use openpyxl to manipulate an Excel file.
    Call load_workbook with the input excel path as specified by the user.
    Remember to save the workbook to the output path that you specified and then call close() so you do not overwrite the input file.

    If code executes with no errors, return the string representation of the Excel file with styles.
    If there are errors, return an error message.
    """
    code_path = "script.py"
    # Write the user code to a file
    with open(code_path, "w") as f:
        f.write(python_code)
    try:
        subprocess.run(
            [sys.executable, code_path], check=True, capture_output=True, timeout=60
        )
    except subprocess.CalledProcessError as e:
        return f"ERROR: User code failed: {e.stderr.decode()}"
    except Exception as e:
        return f"ERROR: Error running user code: {str(e)}"
    # Convert the manipulated Excel file to JSON with styles
    excel_str = excel_to_str_repr(output_excel_path)
    return excel_str


@mcp.tool
def run_excel_code(python_code: str, output_excel_path: str) -> str:
    """
    Run Python code which should use openpyxl to manipulate an Excel file.
    If code executes with no errors, returns the string representation of the Excel file with styles.
    If there are errors, return an error message.
    """
    return run_excel_code_impl(python_code, output_excel_path)


if __name__ == "__main__":
    mcp.run(show_banner=False)
