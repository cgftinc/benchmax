#!/usr/bin/env python3
"""Run Harvey's native agent loop inside a Harbor-owned sandbox.

HarveyHarnessAgent uploads this module alongside the Harvey LAB source and
executes it inside the task environment. Harbor owns the sandbox lifecycle;
this module only adapts Harvey's model and tool interfaces to that environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import openai

HARVEY_ROOT = Path(__file__).resolve().parent / "harvey-labs"


def _load_harvey() -> None:
    if not (HARVEY_ROOT / "harness" / "run.py").is_file():
        raise RuntimeError(
            f"Harvey LAB source is missing at {HARVEY_ROOT}; "
            "HarveyHarnessAgent must upload it before starting the runtime"
        )
    sys.path.insert(0, str(HARVEY_ROOT))


_load_harvey()

from harness.adapters.base import ModelAdapter, ModelResponse, ToolCall  # noqa: E402
from harness.agent_loop import run_agent  # noqa: E402
from harness.tools import ToolExecutor, get_all_tool_definitions  # noqa: E402
from sandbox.sandbox import (  # noqa: E402
    DEFAULT_IMAGE,
    DOCUMENTS_PATH,
    OUTPUT_PATH,
    WORKSPACE_PATH,
    ExecResult,
    Sandbox,
)

SYSTEM_PROMPT_PREAMBLE = (HARVEY_ROOT / "harness" / "system_prompt.md").read_text(encoding="utf-8")
SKILLS_DIR = HARVEY_ROOT / "harness" / "skills"
DEFAULT_SKILLS = sorted(path.parent.name for path in SKILLS_DIR.glob("*/SKILL.md"))
DEFAULT_MAX_TOOL_RESULT_CHARS = 12000
TOOL_RESULT_GUIDANCE = """\
Use the `read` tool for document files and use offsets/limits for large documents.
Tool results may be truncated; request the next relevant range instead of rereading
the whole file. Start drafting by the midpoint of the turn budget."""


def _load_skills(skill_names: list[str]) -> str:
    sections: list[str] = []
    for name in skill_names:
        skill_path = SKILLS_DIR / name / "SKILL.md"
        if not skill_path.is_file():
            raise RuntimeError(f"Harvey skill is missing: {skill_path}")
        sections.append(f"\n\n## Skill: {name}\n\n{skill_path.read_text(encoding='utf-8')}")
    return "\n".join(sections)


def _setup_skill_scripts(skill_names: list[str], workspace_dir: Path) -> None:
    for name in skill_names:
        scripts_dir = SKILLS_DIR / name / "scripts"
        if scripts_dir.is_dir():
            shutil.copytree(
                scripts_dir,
                workspace_dir / "skills" / name / "scripts",
                dirs_exist_ok=True,
            )


def _openai_v1_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return f"{stripped}/v1"
    return stripped


class OpenAIChatCompletionsAdapter(ModelAdapter):
    """Expose an OpenAI-compatible endpoint through Harvey's model interface."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        reasoning_effort: str | None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, reasoning_effort=reasoning_effort)
        self.termination_reason: str | None = None
        self.gateway_controls_sampling = _gateway_controls_sampling(base_url)
        client_options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url.rstrip("/"),
            "timeout": httpx.Timeout(timeout=None),
        }
        if self.gateway_controls_sampling:
            client_options["max_retries"] = 0
        self.client = openai.OpenAI(**client_options)

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [self._translate_tool(tool) for tool in tools],
        }
        if not self.gateway_controls_sampling:
            request["temperature"] = self.temperature
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        response = self._create_with_retries(request)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            self.termination_reason = "output_exceeded"
        message = choice.message
        usage = response.usage
        return ModelResponse(
            message=message.model_dump(exclude_none=True),
            tool_calls=[
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments or "{}",
                )
                for call in (message.tool_calls or [])
            ],
            text=message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def _create_with_retries(self, request: dict[str, Any]):
        if self.gateway_controls_sampling:
            try:
                return self.client.chat.completions.create(**request)
            except openai.BadRequestError as error:
                if _is_context_length_exceeded(error):
                    self.termination_reason = "context_exceeded"
                    # Harvey recognizes this spelling as a graceful context stop
                    # and preserves artifacts for verification.
                    raise RuntimeError(f"context_length_exceeded: {error}") from error
                raise

        for attempt in range(5):
            try:
                return self.client.chat.completions.create(**request)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ):
                if attempt == 4:
                    raise
                time.sleep(min(30, 2**attempt) + random.uniform(0, 1))
        raise RuntimeError("unreachable")

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        return [
            {"role": "tool", "tool_call_id": call_id, "content": result}
            for call_id, result in results
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    @staticmethod
    def _translate_tool(tool: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }


def _gateway_controls_sampling(base_url: str) -> bool:
    path = urlparse(base_url).path
    return any(
        segment in path for segment in ("/v1/sessions/", "/tito/sessions/", "/forward/sessions/")
    )


def _is_context_length_exceeded(error: openai.BadRequestError) -> bool:
    context_error_codes = {
        "context_budget_exceeded",
        "context_length_exceeded",
        "context_window_exceeded",
    }
    if getattr(error, "code", None) in context_error_codes:
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        if body.get("code") in context_error_codes:
            return True
        nested = body.get("error")
        if isinstance(nested, dict) and nested.get("code") in context_error_codes:
            return True
    error_text = str(error).lower()
    return any(code in error_text for code in context_error_codes)


class TruncatingToolExecutor(ToolExecutor):
    """Cap tool observations so one document read cannot consume the context."""

    def __init__(self, *args: Any, max_result_chars: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if max_result_chars < 0:
            raise ValueError("max_result_chars must be non-negative")
        self.max_result_chars = max_result_chars

    def execute(self, tool_name: str, arguments: str | dict) -> str:
        result = self._sanitize(super().execute(tool_name, arguments))
        if self.max_result_chars == 0 or len(result) <= self.max_result_chars:
            return result
        omitted = len(result) - self.max_result_chars
        return (
            result[: self.max_result_chars] + f"\n\n[truncated {omitted} characters; "
            "rerun read with offset/limit for more]"
        )

    @staticmethod
    def _sanitize(value: str) -> str:
        return "".join(char if char in "\n\r\t" or ord(char) >= 32 else "\ufffd" for char in value)


class HarborOwnedSandbox(Sandbox):
    """Map Harvey's sandbox interface onto the current Harbor environment."""

    def start(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_workspace_links()
        self.container_name = "harbor-owned"
        self._started = True

    def stop(self) -> None:
        self.container_name = None
        self._started = False

    def _ensure_workspace_links(self) -> None:
        for name, target in {
            "documents": self.documents_dir,
            "output": self.output_dir,
        }.items():
            link = self.workspace_dir / name
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                if link.is_dir() and not any(link.iterdir()):
                    link.rmdir()
                elif link.is_dir() and link.resolve() == target.resolve():
                    continue
                else:
                    raise RuntimeError(
                        f"Cannot expose {target} at {link}: the destination is "
                        "neither empty nor the requested directory"
                    )
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError as error:
                raise RuntimeError(f"Cannot expose Harbor directory {target} at {link}") from error

    def exec(
        self,
        command: str,
        *,
        cwd: str = WORKSPACE_PATH,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if not self.container_name:
            raise RuntimeError("sandbox is not running - call start() first")

        self.assert_sandbox_path(cwd)
        host_cwd = self._to_host(cwd)
        timeout = timeout if timeout is not None else self.default_timeout
        process_env = {
            **os.environ,
            "DOCUMENTS_DIR": str(self.documents_dir),
            "OUTPUT_DIR": str(self.output_dir),
            "WORKSPACE_DIR": str(self.workspace_dir),
            **self.extra_env,
            **(env or {}),
        }
        try:
            rewritten = self._rewrite_virtual_paths(command)
            process = subprocess.Popen(
                ["bash", "-lc", rewritten],
                cwd=host_cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(process)
                stdout, stderr = process.communicate()
                return ExecResult(
                    stdout,
                    stderr,
                    returncode=None,
                    timed_out=True,
                )
            return ExecResult(stdout, stderr, process.returncode, timed_out=False)
        except (OSError, BrokenPipeError) as error:
            return ExecResult(
                stdout="",
                stderr=f"local exec failed: {type(error).__name__}: {error}",
                returncode=1,
                timed_out=False,
            )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _rewrite_virtual_paths(self, command: str) -> str:
        rewritten = command
        replacements = {
            DOCUMENTS_PATH: ("__HARVEY_DOCUMENTS_PATH__", str(self.documents_dir)),
            OUTPUT_PATH: ("__HARVEY_OUTPUT_PATH__", str(self.output_dir)),
            WORKSPACE_PATH: ("__HARVEY_WORKSPACE_PATH__", str(self.workspace_dir)),
        }
        for virtual, (placeholder, _) in replacements.items():
            rewritten = rewritten.replace(virtual, placeholder)
        for _, (placeholder, actual) in replacements.items():
            rewritten = rewritten.replace(placeholder, actual)
        return rewritten


def _redact_artifacts(path: Path, secret: str) -> None:
    if not path.exists():
        return
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            original = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        redacted = original.replace(secret, "${REDACTED_API_KEY}")
        if redacted != original:
            candidate.write_text(redacted, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be injected into the Harbor agent environment")
    base_url = _openai_v1_base_url(args.base_url)
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")

    results_dir = HARVEY_ROOT / "results" / args.run_id
    output_dir = Path(args.output_dir)
    workspace_dir = results_dir / "workspace"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    skill_names = DEFAULT_SKILLS if args.skills is None else args.skills
    config = {
        "model": args.model,
        "task": args.task_name,
        "run_id": args.run_id,
        "documents_dir": str(Path(args.documents_dir)),
        "output_dir": str(output_dir),
        "max_turns": args.max_turns,
        "max_tool_result_chars": args.max_tool_result_chars,
        "temperature": args.temperature,
        "shell_timeout": args.shell_timeout,
        "reasoning_effort": args.reasoning_effort,
        "skills": skill_names,
        "base_url": base_url,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_start", **config}, indent=2))

    sandbox = HarborOwnedSandbox(
        documents_dir=Path(args.documents_dir),
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        image=DEFAULT_IMAGE,
        default_timeout=args.shell_timeout,
    )
    sandbox.start()
    try:
        system_prompt = SYSTEM_PROMPT_PREAMBLE + "\n\n" + TOOL_RESULT_GUIDANCE
        if skill_names:
            system_prompt += _load_skills(skill_names)
            _setup_skill_scripts(skill_names, workspace_dir)

        adapter = OpenAIChatCompletionsAdapter(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        result = run_agent(
            adapter=adapter,
            system_prompt=system_prompt,
            user_prompt=instruction,
            tool_executor=TruncatingToolExecutor(
                sandbox=sandbox,
                shell_timeout=args.shell_timeout,
                max_result_chars=args.max_tool_result_chars,
            ),
            tools=get_all_tool_definitions(),
            max_turns=args.max_turns,
            transcript_path=str(results_dir / "transcript.jsonl"),
        )
    finally:
        sandbox.stop()

    metrics = {
        "model": args.model,
        "task": args.task_name,
        "run_id": args.run_id,
        "turn_count": result["turn_count"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["input_tokens"] + result["output_tokens"],
        "wall_clock_seconds": result["wall_clock_seconds"],
        "finished_cleanly": result["finished_cleanly"],
        "termination_reason": adapter.termination_reason or "finished",
        "completed_at": datetime.now(UTC).isoformat(),
        **result["tool_metrics"],
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _redact_artifacts(results_dir, api_key)
    print(
        json.dumps(
            {"event": "run_complete", "results_dir": str(results_dir), **metrics},
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Harvey's native agent loop inside a Harbor environment"
    )
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--documents-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=DEFAULT_MAX_TOOL_RESULT_CHARS,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--shell-timeout", type=int, default=60)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--skills", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
