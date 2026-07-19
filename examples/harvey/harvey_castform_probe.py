#!/usr/bin/env python3
"""Harvey LAB harness probe for Castform OpenAI-compatible endpoints.

HarveyHarnessAgent uploads this script into the Harbor sandbox and executes it
with Harvey's dependency set. It can also be run locally for debugging:

    uv run --project archive/harvey-labs archive/harvey_castform_probe.py run \
      --allow-local-tools

    uv run --project archive/harvey-labs archive/harvey_castform_probe.py judge \
      --run-id castform-probe/real-estate/extract-psa-key-terms/scenario-01/qwen3-5-4b/20260703-120000

It imports Harvey LAB from archive/harvey-labs and keeps the provider plumbing
Castform-specific without modifying Harvey's source files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import openai

REPO_ROOT = Path(__file__).resolve().parents[1]
HARVEY_ROOT = REPO_ROOT / "archive" / "harvey-labs"
DEFAULT_TASK = "real-estate/extract-psa-key-terms/scenario-01"
DEFAULT_AGENT_MODEL = "qwen3.5-4b"
DEFAULT_JUDGE_MODEL = "gpt-5.4-nano"
DEFAULT_BASE_URL = "https://llm.castform.dev/v1"
DEFAULT_PROBE_HINT = """\
Probe execution guidance:
- Use the `read` tool for .docx/.xlsx/.pptx/.pdf files; do not inspect binary documents with `bash cat`.
- Use offsets/limits for large documents and gather only enough evidence to write the deliverable.
- Start drafting by the midpoint of the turn budget.
- If the task asks for a .docx deliverable, create it under /workspace/output with python-docx or the docx skill scripts.
- Before finishing, verify the expected deliverable file exists in /workspace/output.
"""


def _load_harvey_modules() -> None:
    if not (HARVEY_ROOT / "harness" / "run.py").exists():
        raise SystemExit(
            f"Harvey LAB checkout not found at {HARVEY_ROOT}. "
            "Clone it first, e.g. git clone --filter=blob:none --sparse --depth 1 "
            "https://github.com/harveyai/harvey-labs.git archive/harvey-labs"
        )
    sys.path.insert(0, str(HARVEY_ROOT))


_load_harvey_modules()

from evaluation import scoring as harvey_scoring  # noqa: E402
from evaluation.report import generate_report  # noqa: E402
from evaluation.run_eval import validate_task_config  # noqa: E402
from harness.adapters.base import ModelAdapter, ModelResponse, ToolCall  # noqa: E402
from harness.agent_loop import run_agent  # noqa: E402
from harness.run import (  # noqa: E402
    DEFAULT_SKILLS,
    SYSTEM_PROMPT_PREAMBLE,  # noqa: E402
    load_skills,
    load_task,
    setup_skill_scripts,
)
from harness.tools import ToolExecutor, get_all_tool_definitions  # noqa: E402
from sandbox.sandbox import (  # noqa: E402
    DEFAULT_IMAGE,
    DOCUMENTS_PATH,
    OUTPUT_PATH,
    WORKSPACE_PATH,
    ExecResult,
    Sandbox,
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _openai_v1_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return f"{stripped}/v1"
    return stripped


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").replace(".", "-")


def _api_key(base_url: str) -> str:
    if "castform" in base_url:
        api_key = os.environ.get("CASTFORM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CASTFORM_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or CASTFORM_API_KEY.")
    return api_key


def _base_url(cli_base_url: str | None = None) -> str:
    raw = (
        cli_base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("CASTFORM_LLM_URL")
        or os.environ.get("LLM_PROXY_URL")
        or DEFAULT_BASE_URL
    )
    return _openai_v1_base_url(raw)


def _bootstrap_env(cli_base_url: str | None = None) -> tuple[str, str]:
    load_env_file(REPO_ROOT / "neon-gateway.env")
    load_env_file(REPO_ROOT / "core" / "trainer" / ".secret.env")
    base_url = _base_url(cli_base_url)
    api_key = _api_key(base_url)
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url
    return api_key, base_url


class CastformChatCompletionsAdapter(ModelAdapter):
    """Harvey adapter backed by OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            model=model, temperature=temperature, reasoning_effort=reasoning_effort
        )
        self.max_tokens = max_tokens
        self.gateway_controlled_sampling = _gateway_controls_sampling(base_url)
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url.rstrip("/"),
            "timeout": httpx.Timeout(timeout=None),
        }
        if self.gateway_controlled_sampling:
            client_kwargs["max_retries"] = 0
        self.client = openai.OpenAI(**client_kwargs)

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [self._translate_tool(tool) for tool in tools],
        }
        if not self.gateway_controlled_sampling:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        response = self._create_with_retries(kwargs)
        message_obj = response.choices[0].message
        message = message_obj.model_dump(exclude_none=True)
        tool_calls = [
            ToolCall(
                id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}"
            )
            for tc in (message_obj.tool_calls or [])
        ]
        usage = response.usage
        return ModelResponse(
            message=message,
            tool_calls=tool_calls,
            text=message_obj.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def _create_with_retries(self, kwargs: dict[str, Any]):
        if self.gateway_controlled_sampling:
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as error:
                # Harvey's agent loop treats context_length_exceeded as a
                # graceful terminal condition and lets the verifier grade any
                # artifacts produced so far. Castform's session gateway uses
                # the equivalent context_budget_exceeded code, so normalize it
                # here instead of turning an expected context stop into a
                # failed Harbor trial with no trustworthy reward.
                if _is_context_budget_exceeded(error):
                    raise RuntimeError(
                        f"context_length_exceeded: {error}"
                    ) from error
                raise

        for attempt in range(5):
            try:
                return self.client.chat.completions.create(**kwargs)
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
            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
            for tool_call_id, result in results
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    def _translate_tool(self, tool: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }


def _gateway_controls_sampling(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return any(
        segment in parsed.path
        for segment in ("/v1/sessions/", "/tito/sessions/", "/forward/sessions/")
    )


def _is_context_budget_exceeded(error: openai.BadRequestError) -> bool:
    """Recognize Castform's gateway-owned context exhaustion response."""

    if getattr(error, "code", None) == "context_budget_exceeded":
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        if body.get("code") == "context_budget_exceeded":
            return True
        nested = body.get("error")
        if isinstance(nested, dict) and nested.get("code") == "context_budget_exceeded":
            return True
    return "context_budget_exceeded" in str(error)


class CastformChatJudge:
    """Criterion judge backed by OpenAI-compatible chat completions."""

    def __init__(
        self, *, model: str, api_key: str, base_url: str, max_tokens: int = 8192
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))

    def evaluate(
        self, prompt_template: str, variables: dict, temperature: float = 0.0
    ) -> dict:
        prompt = prompt_template.format(**variables)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only JSON with keys verdict and reasoning. "
                        "verdict must be pass or fail."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        return _parse_json(text)

    def evaluate_from_file(self, prompt_name: str, variables: dict) -> dict:
        template = (
            HARVEY_ROOT / "evaluation" / "prompts" / f"{prompt_name}.txt"
        ).read_text(encoding="utf-8")
        return self.evaluate(template, variables)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in judge response: {text[:300]}")


class LocalSandbox(Sandbox):
    """Harvey Sandbox-compatible backend for an already-sandboxed process.

    This is the Harbor-style path: Harbor owns the outer sandbox/container, so
    Harvey's tool calls run directly in the current process environment.
    """

    def start(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_workspace_links()
        self.container_name = "local"
        self._started = True

    def stop(self) -> None:
        self.container_name = None
        self._started = False

    def _ensure_workspace_links(self) -> None:
        links = {
            "documents": self.documents_dir,
            "output": self.output_dir,
        }
        for name, target in links.items():
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
                        f"Cannot mount {target} at {link}: path already exists "
                        "and is not an empty directory or matching symlink."
                    )
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                # Symlinks may be unavailable in some environments; bash tools
                # still need the mount-like path to expose the right files.
                if target.is_dir():
                    shutil.copytree(target, link, dirs_exist_ok=True)
                else:
                    raise

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
        timeout = timeout if timeout is not None else self.default_timeout
        host_cwd = self._to_host(cwd)
        rewritten = self._rewrite_virtual_paths(command)
        process_env = {
            **os.environ,
            "DOCUMENTS_DIR": str(self.documents_dir),
            "OUTPUT_DIR": str(self.output_dir),
            "WORKSPACE_DIR": str(self.workspace_dir),
            **self.extra_env,
            **(env or {}),
        }
        try:
            result = subprocess.run(
                ["bash", "-lc", rewritten],
                cwd=host_cwd,
                env=process_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            return ExecResult(
                result.stdout, result.stderr, result.returncode, timed_out=False
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                stdout=exc.stdout.decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=exc.stderr.decode()
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                returncode=None,
                timed_out=True,
            )
        except (OSError, BrokenPipeError) as exc:
            return ExecResult(
                stdout="",
                stderr=f"local exec failed: {type(exc).__name__}: {exc}",
                returncode=1,
                timed_out=False,
            )

    def _rewrite_virtual_paths(self, command: str) -> str:
        rewritten = command
        placeholders = {
            "__CASTFORM_DOCUMENTS_PATH__": str(self.documents_dir),
            "__CASTFORM_OUTPUT_PATH__": str(self.output_dir),
            "__CASTFORM_WORKSPACE_PATH__": str(self.workspace_dir),
        }
        rewritten = rewritten.replace(DOCUMENTS_PATH, "__CASTFORM_DOCUMENTS_PATH__")
        rewritten = rewritten.replace(OUTPUT_PATH, "__CASTFORM_OUTPUT_PATH__")
        rewritten = rewritten.replace(WORKSPACE_PATH, "__CASTFORM_WORKSPACE_PATH__")
        for placeholder, host in placeholders.items():
            rewritten = rewritten.replace(placeholder, host)
        return rewritten


class TruncatingToolExecutor(ToolExecutor):
    """ToolExecutor wrapper that caps oversized tool observations."""

    def __init__(self, *args: Any, max_result_chars: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_result_chars = max_result_chars

    def execute(self, tool_name: str, arguments: str | dict) -> str:
        result = super().execute(tool_name, arguments)
        result = self._sanitize(result)
        if self.max_result_chars <= 0 or len(result) <= self.max_result_chars:
            return result
        omitted = len(result) - self.max_result_chars
        return (
            result[: self.max_result_chars]
            + f"\n\n[truncated {omitted} characters; rerun read with offset/limit for more]"
        )

    @staticmethod
    def _sanitize(value: str) -> str:
        return "".join(
            char if char in "\n\r\t" or ord(char) >= 32 else "\ufffd" for char in value
        )


def _make_sandbox(
    args: argparse.Namespace, task: dict, output_dir: Path, workspace_dir: Path
) -> Sandbox:
    if args.sandbox_backend == "local":
        if not args.allow_local_tools:
            raise SystemExit("--sandbox-backend local requires --allow-local-tools")
        sandbox_cls = LocalSandbox
    else:
        sandbox_cls = Sandbox
    return sandbox_cls(
        documents_dir=Path(task["docs_dir"]),
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        image=args.sandbox_image,
        default_timeout=args.shell_timeout,
    )


def _results_dir_for_run(run_id: str) -> Path:
    return HARVEY_ROOT / "results" / run_id


def _redact_artifacts(path: Path, secrets: list[str]) -> None:
    live_secrets = [secret for secret in secrets if secret]
    if not live_secrets or not path.exists():
        return
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        redacted = text
        for secret in live_secrets:
            redacted = redacted.replace(secret, "${REDACTED_API_KEY}")
        if redacted != text:
            candidate.write_text(redacted, encoding="utf-8")


def run_harness(args: argparse.Namespace) -> None:
    api_key, base_url = _bootstrap_env(args.base_url)
    task = load_task(args.task)

    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"castform-probe/{args.task}/{_slug(args.model)}/{timestamp}"

    results_dir = _results_dir_for_run(run_id)
    output_dir = results_dir / "output"
    workspace_dir = results_dir / "workspace"
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    skill_names = DEFAULT_SKILLS if args.skills is None else args.skills
    sandbox = _make_sandbox(args, task, output_dir, workspace_dir)
    config = {
        "model": args.model,
        "task": args.task,
        "run_id": run_id,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "shell_timeout": args.shell_timeout,
        "reasoning_effort": args.reasoning_effort,
        "skills": skill_names,
        "sandbox_image": args.sandbox_image,
        "sandbox_backend": args.sandbox_backend,
        "base_url": base_url,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(json.dumps({"event": "run_start", **config}, indent=2))
    sandbox.start()
    try:
        adapter = CastformChatCompletionsAdapter(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        tool_executor = TruncatingToolExecutor(
            sandbox=sandbox,
            shell_timeout=args.shell_timeout,
            max_result_chars=args.max_tool_result_chars,
        )
        system_prompt = SYSTEM_PROMPT_PREAMBLE
        if args.probe_hint:
            system_prompt += "\n\n" + args.probe_hint.strip() + "\n"
        if skill_names:
            system_prompt += load_skills(skill_names)
            setup_skill_scripts(skill_names, workspace_dir)

        result = run_agent(
            adapter=adapter,
            system_prompt=system_prompt,
            user_prompt=task["instructions"],
            tool_executor=tool_executor,
            tools=get_all_tool_definitions(),
            max_turns=args.max_turns,
            transcript_path=str(results_dir / "transcript.jsonl"),
        )
    finally:
        sandbox.stop()

    metrics = {
        "model": args.model,
        "task": args.task,
        "run_id": run_id,
        "turn_count": result["turn_count"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["input_tokens"] + result["output_tokens"],
        "wall_clock_seconds": result["wall_clock_seconds"],
        "finished_cleanly": result["finished_cleanly"],
        "completed_at": datetime.now(UTC).isoformat(),
        **result["tool_metrics"],
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _redact_artifacts(results_dir, [api_key, os.environ.get("CASTFORM_API_KEY", "")])
    print(
        json.dumps(
            {"event": "run_complete", "results_dir": str(results_dir), **metrics},
            indent=2,
        )
    )


def run_harbor_task(args: argparse.Namespace) -> None:
    api_key, base_url = _bootstrap_env(args.base_url)
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")

    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"castform-probe/{args.task_name}/{_slug(args.model)}/{timestamp}"

    results_dir = _results_dir_for_run(run_id)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "output"
    workspace_dir = (
        Path(args.workspace_dir) if args.workspace_dir else results_dir / "workspace"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    skill_names = DEFAULT_SKILLS if args.skills is None else args.skills
    sandbox = _make_sandbox(
        args,
        {"docs_dir": str(Path(args.documents_dir)), "instructions": instruction},
        output_dir,
        workspace_dir,
    )
    config = {
        "model": args.model,
        "task": args.task_name,
        "run_id": run_id,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "shell_timeout": args.shell_timeout,
        "reasoning_effort": args.reasoning_effort,
        "skills": skill_names,
        "sandbox_image": args.sandbox_image,
        "sandbox_backend": args.sandbox_backend,
        "documents_dir": str(Path(args.documents_dir)),
        "output_dir": str(output_dir),
        "base_url": base_url,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(json.dumps({"event": "run_start", **config}, indent=2))
    sandbox.start()
    try:
        adapter = CastformChatCompletionsAdapter(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        tool_executor = TruncatingToolExecutor(
            sandbox=sandbox,
            shell_timeout=args.shell_timeout,
            max_result_chars=args.max_tool_result_chars,
        )
        system_prompt = SYSTEM_PROMPT_PREAMBLE
        if args.probe_hint:
            system_prompt += "\n\n" + args.probe_hint.strip() + "\n"
        if skill_names:
            system_prompt += load_skills(skill_names)
            setup_skill_scripts(skill_names, workspace_dir)

        result = run_agent(
            adapter=adapter,
            system_prompt=system_prompt,
            user_prompt=instruction,
            tool_executor=tool_executor,
            tools=get_all_tool_definitions(),
            max_turns=args.max_turns,
            transcript_path=str(results_dir / "transcript.jsonl"),
        )
    finally:
        sandbox.stop()

    metrics = {
        "model": args.model,
        "task": args.task_name,
        "run_id": run_id,
        "turn_count": result["turn_count"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["input_tokens"] + result["output_tokens"],
        "wall_clock_seconds": result["wall_clock_seconds"],
        "finished_cleanly": result["finished_cleanly"],
        "completed_at": datetime.now(UTC).isoformat(),
        **result["tool_metrics"],
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _redact_artifacts(results_dir, [api_key, os.environ.get("CASTFORM_API_KEY", "")])
    print(
        json.dumps(
            {"event": "run_complete", "results_dir": str(results_dir), **metrics},
            indent=2,
        )
    )


def judge_run(args: argparse.Namespace) -> None:
    api_key, base_url = _bootstrap_env(args.base_url)
    run_dir = _results_dir_for_run(args.run_id)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    task_path = HARVEY_ROOT / "tasks" / args.task / "task.json"
    config = json.loads(task_path.read_text(encoding="utf-8"))
    validate_task_config(config=config, task_path=task_path)

    # Avoid Harvey's fallback Claude call for fuzzy deliverable matching while
    # we are explicitly testing cheap OpenAI-compatible judges.
    harvey_scoring._llm_match_deliverables = lambda *_, **__: {}

    for judge_model in args.judges:
        judge = CastformChatJudge(
            model=judge_model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.judge_max_tokens,
        )
        result = harvey_scoring.score_rubric(
            criteria=config["criteria"],
            run_dir=run_dir,
            judge=judge,
            task_desc=config["title"],
            parallel=args.parallel,
        )
        n_total = len(result.criteria_results)
        n_passed = sum(
            1 for item in result.criteria_results if item["verdict"] == "pass"
        )
        scores = {
            "score": result.score,
            "max_score": result.max_score,
            "all_pass": n_total > 0 and n_passed == n_total,
            "n_criteria": n_total,
            "n_passed": n_passed,
            "criteria_results": result.criteria_results,
            "run_id": args.run_id,
            "task": args.task,
            "judge_model": judge_model,
            "scored_at": datetime.now(UTC).isoformat(),
        }
        safe_model = _slug(judge_model)
        scores_path = run_dir / f"scores.{safe_model}.json"
        scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        if len(args.judges) == 1:
            (run_dir / "scores.json").write_text(
                json.dumps(scores, indent=2), encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "event": "judge_complete",
                    "judge_model": judge_model,
                    "n_passed": n_passed,
                    "n_criteria": n_total,
                    "scores_path": str(scores_path),
                },
                indent=2,
            )
        )

    try:
        report_path = generate_report(run_id=args.run_id)
        print(
            json.dumps(
                {"event": "report_complete", "report_path": str(report_path)}, indent=2
            )
        )
    except Exception as exc:
        print(json.dumps({"event": "report_skipped", "error": str(exc)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", default=DEFAULT_TASK)
    run_parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--max-turns", type=int, default=40)
    run_parser.add_argument("--max-tokens", type=int, default=16384)
    run_parser.add_argument("--max-tool-result-chars", type=int, default=12000)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--shell-timeout", type=int, default=60)
    run_parser.add_argument("--reasoning-effort")
    run_parser.add_argument("--skills", nargs="*", default=None)
    run_parser.add_argument("--probe-hint", default=DEFAULT_PROBE_HINT)
    run_parser.add_argument(
        "--no-probe-hint", dest="probe_hint", action="store_const", const=""
    )
    run_parser.add_argument(
        "--sandbox-backend", choices=["local", "podman"], default="local"
    )
    run_parser.add_argument(
        "--allow-local-tools",
        action="store_true",
        help="Required for --sandbox-backend local because bash tools run in this process environment.",
    )
    run_parser.add_argument("--sandbox-image", default=DEFAULT_IMAGE)
    run_parser.set_defaults(func=run_harness)

    harbor_parser = subparsers.add_parser("run-harbor-task")
    harbor_parser.add_argument("--instruction-file", required=True)
    harbor_parser.add_argument("--task-name", default="harbor-task")
    harbor_parser.add_argument("--documents-dir", default="/workspace/documents")
    harbor_parser.add_argument("--output-dir")
    harbor_parser.add_argument("--workspace-dir")
    harbor_parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    harbor_parser.add_argument("--base-url")
    harbor_parser.add_argument("--run-id")
    harbor_parser.add_argument("--max-turns", type=int, default=40)
    harbor_parser.add_argument("--max-tokens", type=int, default=16384)
    harbor_parser.add_argument("--max-tool-result-chars", type=int, default=12000)
    harbor_parser.add_argument("--temperature", type=float, default=0.0)
    harbor_parser.add_argument("--shell-timeout", type=int, default=60)
    harbor_parser.add_argument("--reasoning-effort")
    harbor_parser.add_argument("--skills", nargs="*", default=None)
    harbor_parser.add_argument("--probe-hint", default=DEFAULT_PROBE_HINT)
    harbor_parser.add_argument(
        "--no-probe-hint", dest="probe_hint", action="store_const", const=""
    )
    harbor_parser.add_argument(
        "--sandbox-backend", choices=["local", "podman"], default="local"
    )
    harbor_parser.add_argument(
        "--allow-local-tools",
        action="store_true",
        help="Required for --sandbox-backend local because bash tools run in this process environment.",
    )
    harbor_parser.add_argument("--sandbox-image", default=DEFAULT_IMAGE)
    harbor_parser.set_defaults(func=run_harbor_task)

    judge_parser = subparsers.add_parser("judge")
    judge_parser.add_argument("--task", default=DEFAULT_TASK)
    judge_parser.add_argument("--run-id", required=True)
    judge_parser.add_argument("--base-url")
    judge_parser.add_argument("--judges", nargs="+", default=[DEFAULT_JUDGE_MODEL])
    judge_parser.add_argument("--parallel", type=int, default=2)
    judge_parser.add_argument("--judge-max-tokens", type=int, default=8192)
    judge_parser.set_defaults(func=judge_run)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
