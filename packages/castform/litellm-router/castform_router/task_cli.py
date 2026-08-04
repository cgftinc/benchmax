"""Submit repository tasks to the local Castform router and print its trace."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, TextIO


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="castform",
        description="Submit repository tasks to Castform.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser(
        "task",
        help="route one repository task and print the main execution flow",
    )
    task.add_argument("prompt", nargs="+", help="repository task to route")
    task.add_argument(
        "--base-url",
        default=os.getenv("CASTFORM_BASE_URL", "http://localhost:3000"),
        help="Castform server URL (default: http://localhost:3000)",
    )
    task.add_argument(
        "--session",
        help="session ID used for route pinning (default: a new terminal session)",
    )
    task.add_argument(
        "--route",
        help="override automatic routing with an exact route ID",
    )
    task.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="request timeout in seconds (default: 120)",
    )
    task.add_argument(
        "--verbose",
        action="store_true",
        help="also print every low-level observability event",
    )
    task.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable response instead",
    )
    return parser


def _repository_type(workspace: Path) -> str:
    markers = (
        ("pyproject.toml", "python"),
        ("package.json", "typescript_or_javascript"),
        ("Cargo.toml", "rust"),
        ("go.mod", "go"),
        ("pom.xml", "java"),
    )
    for marker, repository_type in markers:
        if (workspace / marker).exists():
            return repository_type
    return "unknown"


def _request_payload(
    prompt: str,
    *,
    session_id: str,
    workspace: Path,
    route_override: str | None,
) -> dict[str, Any]:
    return {
        "question": prompt,
        "session_id": session_id,
        "route_override": route_override,
        "user_context": {
            "declared_role": "developer",
            "client": "terminal",
        },
        "workspace_context": {
            "repository_name": workspace.name,
            "repository_path": str(workspace),
            "repository_type": _repository_type(workspace),
            "tools": ["repository", "terminal", "tests"],
        },
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Castform returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"could not reach Castform at {url}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Castform returned invalid JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError("Castform returned a non-object response")
    return result


def _print_value(label: str, value: Any, output: TextIO) -> None:
    print(f"  {label}:", file=output)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    for line in rendered.splitlines():
        print(f"    {line}", file=output)


def _render_trace(events: list[Any], output: TextIO) -> None:
    print(f"\nObservability trace ({len(events)} events)", file=output)
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            _print_value(f"event {index}", event, output)
            continue
        stage = event.get("stage", "unknown")
        actor = event.get("actor", "unknown")
        print(f"\n[{index:02d}] {stage}", file=output)
        print(f"  Actor: {actor}", file=output)
        print(f"  Time: {event.get('timestamp', 'unknown')}", file=output)
        print(f"  {event.get('summary', '')}", file=output)
        for field in ("input", "output", "details"):
            if field in event:
                _print_value(field.capitalize(), event[field], output)


def _render_result(
    result: dict[str, Any],
    *,
    base_url: str,
    prompt: str,
    output: TextIO,
    verbose: bool = False,
) -> None:
    selected_route = result.get("selected_route")
    if not isinstance(selected_route, dict):
        selected_route = {}
    decision = result.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    response = result.get("response")
    if not isinstance(response, dict):
        response = {}
    choices = response.get("choices")
    content = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")

    trace_id = result.get("trace_id", "unknown")
    router_output = result.get("router_output")
    if not isinstance(router_output, dict):
        router_output = {}
    predictions = router_output.get("predictions")
    if not isinstance(predictions, list):
        predictions = []

    print("Castform task completed\n", file=output)
    print("Main flow", file=output)
    print(f"[1/5] TASK\n      {prompt}", file=output)
    print("[2/5] QWEN ROUTE SCORING", file=output)
    print(
        "      LiteLLM → "
        f"{result.get('router_model_label', 'Qwen 0.8B')} → "
        f"{len(predictions)} route scores",
        file=output,
    )
    print(
        f"      Router latency: {result.get('router_duration_ms', 'unknown')} ms",
        file=output,
    )
    print("[3/5] CASTFORM ROUTE SELECTION", file=output)
    print(
        f"      {selected_route.get('route_id', 'unknown')}",
        file=output,
    )
    print(
        "      Policy: "
        f"{decision.get('reason', 'unknown')} "
        f"(cache_hit={decision.get('cache_hit', False)})",
        file=output,
    )
    print("[4/5] APPROVAL", file=output)
    print(
        "      Not implemented yet — this local lab proceeds automatically.",
        file=output,
    )
    print("[5/5] CODING HARNESS (SIMULATED)", file=output)
    print(
        f"      {selected_route.get('harness', 'unknown')} → "
        f"LiteLLM ({selected_route.get('gateway_model', 'unknown')}) → "
        f"{selected_route.get('provider', 'unknown')}",
        file=output,
    )
    print("      No repository files were changed.", file=output)

    print("\nResult", file=output)
    print(f"Session: {result.get('session_id', 'unknown')}", file=output)
    print(f"Trace: {trace_id}", file=output)
    print(
        f"Trace API: {base_url.rstrip('/')}/api/traces/{trace_id}",
        file=output,
    )
    print(f"Response: {content if content is not None else 'none'}", file=output)

    if verbose:
        events = result.get("events")
        _render_trace(events if isinstance(events, list) else [], output)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "task":
        return 2

    prompt = " ".join(args.prompt).strip()
    session_id = args.session or f"terminal-{uuid.uuid4().hex[:8]}"
    base_url = args.base_url.rstrip("/")
    payload = _request_payload(
        prompt,
        session_id=session_id,
        workspace=Path.cwd().resolve(),
        route_override=args.route,
    )
    try:
        result = _post_json(f"{base_url}/api/ask", payload, args.timeout)
    except RuntimeError as error:
        print(f"castform: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _render_result(
            result,
            base_url=base_url,
            prompt=prompt,
            output=sys.stdout,
            verbose=args.verbose,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
