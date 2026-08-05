"""Judge-guided AutoCompact collection for Harvey's native agent loop.

The module is intentionally independent of Harbor and Harvey imports so its
state machine, record construction, and filtering can be unit tested without a
sandbox.  Runtime adapters only need the small ModelAdapter/ToolExecutor
interfaces used by :func:`run_autocompact_agent`.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

SCHEMA_VERSION = 1
SUMMARY_HEADER = "# Auto Context Summary"
COMPACT_TOOL = {
    "name": "compact",
    "description": (
        "Compact completed, stale work into a precise working-state summary before "
        "moving to the next phase. Do not call this during an unfinished tool sequence."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}
AUTOCOMPACT_INSTRUCTIONS = f"""

## AutoCompact

You may call `compact()` when a task phase is complete and older exploration is
now distracting. Never compact during an unfinished evidence-gathering or tool
sequence. After calling it, the runtime will ask you to produce a
`{SUMMARY_HEADER}` and will then continue the task from the original task,
recent completed work, and that summary.
"""
SUMMARY_REQUEST = f"""The previous action selected `compact()`. Using the full
history above, produce exactly one `{SUMMARY_HEADER}`. Preserve the requested
deliverable, jurisdiction, parties, dates, numbers, material facts, authorities
with source locations, inspected and remaining documents, conclusions,
uncertainties, artifact paths, verification status, and the next action. Do not
invent facts and do not continue the task yet."""

ANNOTATION_PROTOCOL = """Review only the context supplied. Compaction is valid
at a completed phase boundary: evidence discovery to analysis, resolved analysis
to drafting, or drafting to verification. It is invalid during an unfinished
tool sequence, unresolved fact extraction, or before exact facts and authorities
have been preserved. A summary must preserve the deliverable, jurisdiction,
parties, dates, numbers, material facts, authorities and source locations,
inspected and remaining documents, conclusions and uncertainties, artifact
paths, verification status, and next action. Never rely on a hidden rubric."""


class Adapter(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]: ...

    def make_system_message(self, content: str) -> dict[str, Any]: ...

    def make_user_message(self, content: str) -> dict[str, Any]: ...


class ToolExecutorLike(Protocol):
    def execute(self, tool_name: str, arguments: str | dict[str, Any]) -> str: ...

    def get_metrics(self) -> dict[str, Any]: ...


class Judge(Protocol):
    input_tokens: int
    output_tokens: int

    def review_action(
        self,
        visible_context: list[dict[str, Any]],
        candidate: dict[str, Any],
        *,
        compactions_used: int,
        max_compactions: int,
    ) -> dict[str, Any]: ...

    def review_summary(
        self,
        visible_context: list[dict[str, Any]],
        candidate: str,
    ) -> dict[str, Any]: ...

    def review_continuation(
        self,
        visible_context: list[dict[str, Any]],
        candidate: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class NoopJudge:
    """Keep model output unchanged for judge-free inference."""

    input_tokens: int = 0
    output_tokens: int = 0

    def review_action(self, visible_context, candidate, **kwargs):
        del visible_context, candidate, kwargs
        return {"decision": "keep"}

    def review_summary(self, visible_context, candidate):
        del visible_context, candidate
        return {"decision": "keep"}

    def review_continuation(self, visible_context, candidate):
        del visible_context, candidate
        return {"decision": "keep"}


class JudgeProtocolError(RuntimeError):
    pass


class HTTPJudge:
    """Small provider client for the online intervention judge."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 180,
    ) -> None:
        if provider not in {"anthropic", "openai"}:
            raise ValueError(f"unsupported judge provider: {provider}")
        if not api_key:
            raise ValueError("judge API key must be non-empty")
        if provider == "anthropic" and base_url:
            raise ValueError("judge base_url is supported only for openai")
        self.provider = provider
        self.model = model.removeprefix(f"{provider}/")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.input_tokens = 0
        self.output_tokens = 0

    def review_action(self, visible_context, candidate, *, compactions_used, max_compactions):
        payload = {
            "kind": "action",
            "protocol": ANNOTATION_PROTOCOL,
            "compactions_used": compactions_used,
            "max_compactions": max_compactions,
            "visible_context": visible_context,
            "candidate": candidate,
            "response_schema": {
                "decision": "keep | replace_with_compact",
                "reason_code": "short machine-readable code",
            },
        }
        decision = self._request(payload)
        choice = decision.get("decision")
        if choice not in {"keep", "replace_with_compact"}:
            raise JudgeProtocolError(f"invalid action judge decision: {choice!r}")
        return decision

    def review_summary(self, visible_context, candidate):
        decision = self._request(
            {
                "kind": "summary",
                "protocol": ANNOTATION_PROTOCOL,
                "visible_context": visible_context,
                "candidate": candidate,
                "response_schema": {
                    "decision": "keep | repair",
                    "corrected_summary": "required for repair",
                    "reason_code": "short machine-readable code",
                },
            }
        )
        choice = decision.get("decision")
        if choice not in {"keep", "repair"}:
            raise JudgeProtocolError(f"invalid summary judge decision: {choice!r}")
        if choice == "repair" and not isinstance(decision.get("corrected_summary"), str):
            raise JudgeProtocolError("summary repair omitted corrected_summary")
        return decision

    def review_continuation(self, visible_context, candidate):
        decision = self._request(
            {
                "kind": "continuation",
                "protocol": ANNOTATION_PROTOCOL,
                "visible_context": visible_context,
                "candidate": candidate,
                "response_schema": {
                    "decision": "keep | repair",
                    "corrected_message": "required assistant message for repair",
                    "reason_code": "short machine-readable code",
                },
            }
        )
        choice = decision.get("decision")
        if choice not in {"keep", "repair"}:
            raise JudgeProtocolError(f"invalid continuation judge decision: {choice!r}")
        if choice == "repair":
            _validate_assistant_message(decision.get("corrected_message"))
        return decision

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                text = self._provider_request(payload)
                parsed = json.loads(_extract_json(text))
                if not isinstance(parsed, dict):
                    raise JudgeProtocolError("judge response must be a JSON object")
                self._validate_decision(str(payload.get("kind")), parsed)
                return parsed
            except (httpx.HTTPError, json.JSONDecodeError, JudgeProtocolError, ValueError) as error:
                last_error = error
        raise JudgeProtocolError(
            f"judge returned invalid output twice: {last_error}"
        ) from last_error

    @staticmethod
    def _validate_decision(kind: str, decision: dict[str, Any]) -> None:
        choice = decision.get("decision")
        if kind == "action":
            if choice not in {"keep", "replace_with_compact"}:
                raise JudgeProtocolError(f"invalid action judge decision: {choice!r}")
            return
        if kind == "summary":
            if choice not in {"keep", "repair"}:
                raise JudgeProtocolError(f"invalid summary judge decision: {choice!r}")
            if choice == "repair" and not isinstance(
                decision.get("corrected_summary"), str
            ):
                raise JudgeProtocolError("summary repair omitted corrected_summary")
            return
        if kind == "continuation":
            if choice not in {"keep", "repair"}:
                raise JudgeProtocolError(f"invalid continuation judge decision: {choice!r}")
            if choice == "repair":
                _validate_assistant_message(decision.get("corrected_message"))
            return
        raise JudgeProtocolError(f"unknown judge request kind: {kind!r}")

    def _provider_request(self, payload: dict[str, Any]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False)
        if self.provider == "anthropic":
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 4096,
                    "system": "Return only the requested JSON object.",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage", {})
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))
            return "".join(
                item.get("text", "")
                for item in body.get("content", [])
                if item.get("type") == "text"
            )

        base_url = self.base_url or "https://api.openai.com/v1"
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Return only the requested JSON object."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        self.input_tokens += int(usage.get("prompt_tokens", 0))
        self.output_tokens += int(usage.get("completion_tokens", 0))
        return body["choices"][0]["message"].get("content", "")


def run_autocompact_agent(
    *,
    adapter: Adapter,
    judge: Judge | None,
    system_prompt: str,
    user_prompt: str,
    tool_executor: ToolExecutorLike,
    tools: list[dict[str, Any]],
    max_turns: int,
    max_compactions: int,
    trajectory_path: str | Path,
) -> dict[str, Any]:
    """Run Harvey with judge intervention and emit lossless call-level records."""

    if max_compactions < 1:
        raise ValueError("max_compactions must be positive")
    judge = judge or NoopJudge()
    all_tools = [*tools, COMPACT_TOOL]
    openai_tools = _openai_tools(all_tools)
    system_message = adapter.make_system_message(system_prompt + AUTOCOMPACT_INSTRUCTIONS)
    user_message = adapter.make_user_message(user_prompt)
    messages = [system_message, user_message]
    recent_exchanges: list[list[dict[str, Any]]] = []
    audit_steps: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    turn_count = 0
    compactions_used = 0
    finished_cleanly = False
    context_overflow = False
    termination_reason = "max_turns_exceeded"
    start_time = time.time()

    try:
        while turn_count < max_turns:
            pre_context = copy.deepcopy(messages)
            try:
                proposal = adapter.chat(messages, all_tools)
            except Exception as error:
                if _is_context_error(error):
                    context_overflow = True
                    termination_reason = "context_exceeded"
                    break
                raise
            turn_count += 1
            total_input_tokens += int(proposal.input_tokens)
            total_output_tokens += int(proposal.output_tokens)
            decision = judge.review_action(
                pre_context,
                proposal.message,
                compactions_used=compactions_used,
                max_compactions=max_compactions,
            )
            executed_message = _approved_action(decision, proposal.message, compactions_used)
            audit_steps.append(
                {
                    "kind": "action",
                    "visible_context": pre_context,
                    "candidate": proposal.message,
                    "judge_decision": copy.deepcopy(decision),
                    "executed": executed_message,
                }
            )

            compact_call = _compact_call(executed_message)
            if compact_call is None:
                messages.append(copy.deepcopy(executed_message))
                tool_calls = _tool_calls(executed_message)
                if not tool_calls:
                    finished_cleanly = True
                    termination_reason = "finished"
                    break
                tool_messages = _execute_tools(adapter, tool_executor, tool_calls)
                messages.extend(tool_messages)
                recent_exchanges.append(
                    [copy.deepcopy(executed_message), *copy.deepcopy(tool_messages)]
                )
                continue

            if compactions_used >= max_compactions:
                termination_reason = "compaction_limit_exceeded"
                break
            if turn_count >= max_turns:
                termination_reason = "max_turns_exceeded"
                break

            event_id = compactions_used
            summary_prompt = [
                *copy.deepcopy(pre_context),
                adapter.make_user_message(SUMMARY_REQUEST),
            ]
            summary_response = adapter.chat(summary_prompt, [])
            total_input_tokens += int(summary_response.input_tokens)
            total_output_tokens += int(summary_response.output_tokens)
            if getattr(summary_response, "tool_calls", None):
                raise JudgeProtocolError("summary generation must not call tools")
            summary_decision = judge.review_summary(summary_prompt, summary_response.text)
            corrected_summary = (
                summary_response.text
                if summary_decision["decision"] == "keep"
                else summary_decision["corrected_summary"]
            ).strip()
            if not corrected_summary.startswith(SUMMARY_HEADER):
                raise JudgeProtocolError(f"corrected summary must start with {SUMMARY_HEADER!r}")

            compacted_context = [copy.deepcopy(system_message), copy.deepcopy(user_message)]
            if recent_exchanges:
                compacted_context.extend(copy.deepcopy(recent_exchanges[-1]))
            compacted_context.append(copy.deepcopy(executed_message))
            compacted_context.extend(
                adapter.make_tool_result_messages([(compact_call["id"], corrected_summary)])
            )

            continuation_response = adapter.chat(compacted_context, all_tools)
            turn_count += 1
            total_input_tokens += int(continuation_response.input_tokens)
            total_output_tokens += int(continuation_response.output_tokens)
            continuation_decision = judge.review_continuation(
                compacted_context,
                continuation_response.message,
            )
            corrected_continuation = (
                continuation_response.message
                if continuation_decision["decision"] == "keep"
                else continuation_decision["corrected_message"]
            )
            _validate_assistant_message(corrected_continuation)
            continuation_calls = _tool_calls(corrected_continuation)
            tool_messages = (
                _execute_tools(adapter, tool_executor, continuation_calls)
                if continuation_calls
                else []
            )

            event_prefix = f"compact-{event_id}"
            records.extend(
                [
                    _record(
                        record_id=f"{event_prefix}-trigger",
                        category="autocompact_trigger",
                        event_id=event_id,
                        prompt=pre_context,
                        completion=executed_message,
                        tools=openai_tools,
                        decisions={"action": decision["decision"]},
                    ),
                    _record(
                        record_id=f"{event_prefix}-summary",
                        category="autocompact_summary",
                        event_id=event_id,
                        prompt=summary_prompt,
                        completion={"role": "assistant", "content": corrected_summary},
                        tools=[],
                        decisions={"summary": summary_decision["decision"]},
                    ),
                    _record(
                        record_id=f"{event_prefix}-continuation",
                        category="autocompact_continuation",
                        event_id=event_id,
                        prompt=compacted_context,
                        completion=corrected_continuation,
                        tools=openai_tools,
                        decisions={"continuation": continuation_decision["decision"]},
                    ),
                ]
            )
            audit_steps.append(
                {
                    "kind": "compaction",
                    "event_id": event_id,
                    "source_context": pre_context,
                    "summary_prompt": summary_prompt,
                    "candidate_summary": summary_response.text,
                    "summary_decision": copy.deepcopy(summary_decision),
                    "corrected_summary": corrected_summary,
                    "compacted_context": compacted_context,
                    "candidate_continuation": continuation_response.message,
                    "continuation_decision": copy.deepcopy(continuation_decision),
                    "corrected_continuation": corrected_continuation,
                }
            )
            compactions_used += 1
            messages = [*copy.deepcopy(compacted_context), copy.deepcopy(corrected_continuation)]
            if not continuation_calls:
                finished_cleanly = True
                termination_reason = "finished"
                break
            messages.extend(tool_messages)
            recent_exchanges = [
                [copy.deepcopy(corrected_continuation), *copy.deepcopy(tool_messages)]
            ]
    except JudgeProtocolError as error:
        termination_reason = "judge_error"
        audit_steps.append({"kind": "error", "error": str(error)})
    finally:
        trajectory = {
            "schema_version": SCHEMA_VERSION,
            "messages": messages,
            "steps": audit_steps,
            "sft_records": records,
            "metrics": {
                "turn_count": turn_count,
                "compactions": compactions_used,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "judge_input_tokens": int(judge.input_tokens),
                "judge_output_tokens": int(judge.output_tokens),
                "termination_reason": termination_reason,
            },
        }
        destination = Path(trajectory_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(trajectory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "messages": messages,
        "turn_count": turn_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "wall_clock_seconds": round(time.time() - start_time, 2),
        "finished_cleanly": finished_cleanly,
        "context_overflow": context_overflow,
        "tool_metrics": tool_executor.get_metrics(),
        "finish_summary": None,
        "termination_reason": termination_reason,
        "compactions": compactions_used,
        "sft_record_count": len(records),
        "judge_input_tokens": int(judge.input_tokens),
        "judge_output_tokens": int(judge.output_tokens),
    }


def _approved_action(
    decision: dict[str, Any],
    candidate: dict[str, Any],
    event_id: int,
) -> dict[str, Any]:
    choice = decision["decision"]
    if choice == "keep":
        _validate_assistant_message(candidate)
        return copy.deepcopy(candidate)
    if choice != "replace_with_compact":
        raise JudgeProtocolError(f"invalid approved action decision: {choice!r}")
    call_id = f"compact-{event_id}-{uuid.uuid4().hex[:8]}"
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "compact", "arguments": "{}"},
            }
        ],
    }


def _compact_call(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = _tool_calls(message)
    compact = [call for call in calls if call["name"] == "compact"]
    if not compact:
        return None
    if len(calls) != 1:
        raise JudgeProtocolError("compact must be the only tool call in an action")
    return compact[0]


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        normalized.append(
            {
                "id": str(call.get("id") or ""),
                "name": str(function.get("name") or call.get("name") or ""),
                "arguments": function.get("arguments", call.get("arguments", "{}")),
            }
        )
    return normalized


def _execute_tools(adapter, tool_executor, calls):
    results: list[tuple[str, str]] = []
    for call in calls:
        if call["name"] == "compact":
            raise JudgeProtocolError("nested compact call must be handled by the context manager")
        results.append(
            (call["id"], tool_executor.execute(call["name"], call["arguments"]))
        )
    return adapter.make_tool_result_messages(results)


def _record(
    *,
    record_id: str,
    category: str,
    event_id: int,
    prompt: list[dict[str, Any]],
    completion: dict[str, Any],
    tools: list[dict[str, Any]],
    decisions: dict[str, str],
) -> dict[str, Any]:
    masked_prompt = []
    for message in copy.deepcopy(prompt):
        if message.get("role") == "assistant":
            message["step_loss_mask"] = 0
        masked_prompt.append(message)
    target = copy.deepcopy(completion)
    target["step_loss_mask"] = 1
    messages = [_sft_message(message) for message in [*masked_prompt, target]]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "category": category,
        "source_kind": "judge_guided_on_policy",
        "prompt_messages": masked_prompt,
        "completion_messages": [target],
        "messages": messages,
        "tools": tools,
        "task": {
            "compaction_event_id": event_id,
            "judge_decisions": decisions,
        },
    }


def _sft_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one OpenAI message for Slime's standard JSONL Dataset."""

    normalized = {
        key: copy.deepcopy(value)
        for key, value in message.items()
        if key != "tool_calls" and value is not None
    }
    calls = _tool_calls(message)
    if calls:
        normalized["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": _tool_arguments(call["arguments"]),
                },
            }
            for call in calls
        ]
    return normalized


def _tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return copy.deepcopy(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"value": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if arguments is None:
        return {}
    return {"value": copy.deepcopy(arguments)}


def _openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def _validate_assistant_message(message: Any) -> None:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise JudgeProtocolError("corrected action must be an assistant message")
    if not message.get("content") and not message.get("tool_calls"):
        raise JudgeProtocolError("assistant message must contain content or tool_calls")


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1])
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise JudgeProtocolError("judge response did not contain a JSON object")
    return stripped[start : end + 1]


def _is_context_error(error: Exception) -> bool:
    value = str(error).lower()
    return "prompt is too long" in value or "context_length_exceeded" in value


__all__ = [
    "AUTOCOMPACT_INSTRUCTIONS",
    "COMPACT_TOOL",
    "HTTPJudge",
    "JudgeProtocolError",
    "NoopJudge",
    "SCHEMA_VERSION",
    "SUMMARY_HEADER",
    "run_autocompact_agent",
]
