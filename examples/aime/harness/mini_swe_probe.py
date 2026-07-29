"""Self-contained mini-swe-agent 2.4.5 loop for Castform sandboxes.

Replicates the tool-call agent loop of mini-swe-agent 2.4.5 (mini.yaml
defaults) against one OpenAI-compatible endpoint using only the standard
library. Uploaded into the sandbox as a single file and run with the image's
python3, so agent setup needs no apt, PyPI, or wheel transfer.

Kept in sync with: minisweagent/agents/default.py, config/mini.yaml, and
models/utils/actions_toolcall.py at v2.4.5.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request

SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer.\n"

INSTANCE_TEMPLATE = """Please solve this issue: {task}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

[Makes bash tool call with {{"command": "ls -la"}} as arguments]
</example_response>

<system_information>
{system} {release} {version} {machine}
</system_information>

## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:
```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```
"""

FORMAT_ERROR_TRUNCATED = (
    "Your previous response reached the output token limit (finish_reason={finish_reason}) "
    "before you produced a tool call, so it was cut off. Respond more concisely and finish "
    "with exactly one bash tool call. If you need to think more, do so briefly."
)

FORMAT_ERROR_GENERAL = """Tool call error:

<error>
{error}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {{"command": "your_command_here"}}

If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command."""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_CONSECUTIVE_FORMAT_ERRORS = 3


class FormatError(Exception):
    def __init__(self, message: dict) -> None:
        super().__init__(message["content"])
        self.message = message


class Submitted(Exception):  # noqa: N818 — control-flow signal, not an error
    def __init__(self, submission: str) -> None:
        super().__init__("submitted")
        self.submission = submission


def chat_completion(args: argparse.Namespace, messages: list[dict]) -> dict:
    payload = {
        "model": args.model,
        "messages": [{k: v for k, v in m.items() if k != "extra"} for m in messages],
        "tools": [BASH_TOOL],
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
            # Cloudflare WAF on *.castform.dev rejects default Python-urllib UAs.
            "User-Agent": "castform-mini-swe-probe/2.4.5",
        },
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        return json.load(response)


def parse_actions(message: dict, finish_reason: str | None) -> list[dict]:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        if finish_reason == "length":
            content = FORMAT_ERROR_TRUNCATED.format(finish_reason=finish_reason)
        else:
            content = FORMAT_ERROR_GENERAL.format(
                error=(
                    "No tool calls found in the response. Every response MUST "
                    "include at least one tool call."
                )
            )
        raise FormatError({"role": "user", "content": content})
    actions = []
    for call in tool_calls:
        error = ""
        arguments: object = {}
        try:
            arguments = json.loads(call["function"]["arguments"])
        except Exception as exc:
            error = f"Error parsing tool call arguments: {exc}."
        if call["function"]["name"] != "bash":
            error += f"Unknown tool '{call['function']['name']}'."
        if not isinstance(arguments, dict) or "command" not in arguments:
            error += "Missing 'command' argument in bash tool call."
        if error:
            raise FormatError(
                {
                    "role": "user",
                    "content": FORMAT_ERROR_GENERAL.format(error=error.strip()),
                }
            )
        actions.append({"command": arguments["command"], "tool_call_id": call["id"]})
    return actions


def execute(command: str, timeout: int) -> dict:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="backslashreplace",
        )
        output = {
            "output": completed.stdout + completed.stderr,
            "returncode": completed.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        collected = (exc.stdout or b"") + (exc.stderr or b"")
        if isinstance(collected, bytes):
            collected = collected.decode(errors="backslashreplace")
        output = {
            "output": collected,
            "returncode": -1,
            "exception_info": f"command timed out after {timeout} seconds",
        }
    lines = output["output"].lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == SUBMIT_MARKER and output["returncode"] == 0:
        raise Submitted("".join(lines[1:]))
    return output


def observation_content(output: dict) -> str:
    text = output["output"]
    if len(text) < 10_000:
        body = {"returncode": output["returncode"], "output": text}
    else:
        body = {
            "returncode": output["returncode"],
            "output_head": text[:5_000],
            "output_tail": text[-5_000:],
            "elided_chars": len(text) - 10_000,
            "warning": "Output too long.",
        }
    if output.get("exception_info"):
        body["exception_info"] = output["exception_info"]
    return json.dumps(body, indent=2)


def save_trajectory(
    path: str, messages: list[dict], usage: dict, exit_status: str, submission: str
) -> None:
    data = {
        "info": {
            "model_stats": {"instance_cost": 0.0, "api_calls": usage["calls"]},
            "mini_version": "2.4.5-castform-probe",
            "exit_status": exit_status,
            "submission": submission,
        },
        "messages": messages,
        "trajectory_format": "mini-swe-agent-1.1",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-limit", type=int, default=30)
    parser.add_argument("--shell-timeout", type=int, default=60)
    parser.add_argument("--request-timeout", type=int, default=600)
    args = parser.parse_args()

    uname = platform.uname()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": INSTANCE_TEMPLATE.format(
                task=args.task,
                system=uname.system,
                release=uname.release,
                version=uname.version,
                machine=uname.machine,
            ),
        },
    ]
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    exit_status, submission = "LimitsExceeded", ""
    consecutive_format_errors = 0

    for _ in range(args.step_limit):
        usage["calls"] += 1
        response = chat_completion(args, messages)
        choice = response["choices"][0]
        message = choice["message"]
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, int) and key in usage:
                usage[key] += value
        assistant: dict = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            assistant["tool_calls"] = message["tool_calls"]
        # Shape matches harbor's _message_usage reader (extra.response.usage).
        assistant["extra"] = {"response": {"usage": response.get("usage") or {}}}
        messages.append(assistant)

        try:
            actions = parse_actions(message, choice.get("finish_reason"))
            consecutive_format_errors = 0
        except FormatError as error:
            consecutive_format_errors += 1
            if consecutive_format_errors >= MAX_CONSECUTIVE_FORMAT_ERRORS:
                exit_status = "RepeatedFormatError"
                break
            messages.append(error.message)
            continue

        try:
            outputs = [execute(action["command"], args.shell_timeout) for action in actions]
        except Submitted as done:
            exit_status, submission = "Submitted", done.submission
            break
        for action, output in zip(actions, outputs):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": action["tool_call_id"],
                    "content": observation_content(output),
                }
            )

    save_trajectory(args.output, messages, usage, exit_status, submission)
    print(f"exit_status={exit_status} calls={usage['calls']}")
    return 0 if exit_status in ("Submitted", "LimitsExceeded") else 1


if __name__ == "__main__":
    sys.exit(main())
