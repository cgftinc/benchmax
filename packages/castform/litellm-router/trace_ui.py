"""Local trace viewer server and LiteLLM client.

The server uses only Python's standard library so the demo stays self-contained.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from castform_router.benchmax_workflow import (
    build_benchmax_plan,
    write_benchmax_plan,
)
from castform_router.job_router import ROUTES, HeuristicJobRouter, JobRouter
from castform_router.router_protocol import (
    SYSTEM_PROMPT,
    OpenAICompatibleRouteScorer,
    model_request_payload,
    model_response_payload,
)
from castform_router.trace import append_trace, read_trace, valid_trace_id
from castform_router.training_environment import (
    TRAINING_ROUTE_CATALOG,
    build_training_workspace,
    parse_github_repo,
    route_catalog_json,
)
from castform_router.types import HarnessRouteRequest
from castform_router.workflow_demo import (
    advance_demo,
    load_demo_state,
    reset_demo,
)

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://litellm:4000")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-local-dev")
WEB_ROOT = Path(os.getenv("CASTFORM_WEB_ROOT", "/workspace/web"))
TRAINING_RUNS_ROOT = Path(
    os.getenv("CASTFORM_TRAINING_RUNS_DIR", "/training-runs")
)
ROUTER_MODEL_BASE_URL = os.getenv("CASTFORM_ROUTER_MODEL_BASE_URL")
ROUTER_MODEL_NAME = os.getenv(
    "CASTFORM_ROUTER_MODEL_NAME",
    "castform-router-0.8b",
)
ROUTER_MODEL_API_KEY = os.getenv(
    "CASTFORM_ROUTER_MODEL_API_KEY",
    LITELLM_MASTER_KEY,
)
ROUTER_MODEL_LABEL = os.getenv(
    "CASTFORM_ROUTER_MODEL_LABEL",
    ROUTER_MODEL_NAME,
)
ROUTER_MODEL_STATUS = os.getenv(
    "CASTFORM_ROUTER_MODEL_STATUS",
    "plumbing_only",
)
JOB_ROUTER = (
    JobRouter(
        scorer=OpenAICompatibleRouteScorer(
            base_url=ROUTER_MODEL_BASE_URL,
            model=ROUTER_MODEL_NAME,
            api_key=ROUTER_MODEL_API_KEY,
        )
    )
    if ROUTER_MODEL_BASE_URL
    else HeuristicJobRouter()
)
WORKSPACE_ID = re.compile(r"^router-[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$")

STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "CastformTraceUI/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[trace-ui] {format % args}", flush=True)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/training/catalog":
            self._json(
                HTTPStatus.OK,
                {
                    "routes": route_catalog_json(),
                    "defaults": {
                        "tasks_per_repo": 10,
                        "repetitions": 1,
                        "average_run_cost_usd": 1.0,
                        "privacy_mode": "castform_hosted",
                        "router_rung": "knn",
                        "router_model": "claude-sonnet-4-6",
                    },
                },
            )
            return
        if path == "/api/router/status":
            self._json(
                HTTPStatus.OK,
                {
                    "enabled": bool(ROUTER_MODEL_BASE_URL),
                    "model": ROUTER_MODEL_NAME,
                    "label": ROUTER_MODEL_LABEL,
                    "transport": "LiteLLM" if ROUTER_MODEL_BASE_URL else "local",
                    "status": ROUTER_MODEL_STATUS,
                    "system_prompt": SYSTEM_PROMPT,
                    "policy": JOB_ROUTER.policy_version,
                    "quality_threshold": JOB_ROUTER.quality_threshold,
                },
            )
            return
        demo_workspace = self._demo_workspace(path)
        if demo_workspace is not None:
            try:
                self._json(HTTPStatus.OK, load_demo_state(demo_workspace))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if path.startswith("/api/traces/"):
            trace_id = valid_trace_id(path.removeprefix("/api/traces/"))
            if trace_id is None:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid trace id"})
                return
            self._json(
                HTTPStatus.OK,
                {"trace_id": trace_id, "events": read_trace(trace_id)},
            )
            return
        filename = STATIC_FILES.get(path)
        if filename is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._static(WEB_ROOT / filename)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/github/inspect":
            self._inspect_github_repository()
            return
        if path == "/api/training/environments":
            self._create_training_environment()
            return
        if path.endswith("/demo/next"):
            workspace = self._demo_workspace(path.removesuffix("/next"))
            if workspace is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                self._json(HTTPStatus.OK, advance_demo(workspace))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if path.endswith("/demo/reset"):
            workspace = self._demo_workspace(path.removesuffix("/reset"))
            if workspace is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                self._json(HTTPStatus.OK, reset_demo(workspace))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if path != "/api/ask":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            body = self._request_json()
            question = self._required_string(body, "question")
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            session_id = f"session-{uuid.uuid4().hex[:8]}"
        override = body.get("route_override")
        route_ids = {route.route_id for route in ROUTES}
        if override not in (None, "") and override not in route_ids:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid route_override"})
            return

        trace_id = uuid.uuid4().hex
        user_context = body.get("user_context")
        if not isinstance(user_context, dict):
            user_context = {"declared_role": "developer"}
        workspace_context = body.get("workspace_context")
        if not isinstance(workspace_context, dict):
            workspace_context = {
                "repository_type": "typescript",
                "tools": ["repository", "terminal", "tests"],
            }
        route_request = HarnessRouteRequest(
            request_id=trace_id,
            session_id=session_id,
            task_text=question,
            task_domain="software_engineering",
            user_context=user_context,
            workspace_context=workspace_context,
            candidate_routes=ROUTES,
            route_override=override or None,
        )
        append_trace(
            trace_id,
            actor="Trace UI",
            stage="client.task_submitted",
            summary="Submitted one repository task to the Castform job router.",
            input={
                "task_text": question,
                "session_id": session_id,
                "route_override": override or None,
            },
        )
        append_trace(
            trace_id,
            actor="Castform orchestrator",
            stage="job.request_normalized",
            summary="Separated the task, user context, workspace context, and constraints.",
            input={
                "task_text": question,
                "user_context": user_context,
                "workspace_context": workspace_context,
            },
            output={
                "request_id": trace_id,
                "session_id": session_id,
                "task_domain": route_request.task_domain,
            },
        )
        append_trace(
            trace_id,
            actor="Route registry",
            stage="route.candidates_built",
            summary="Built eligible harness, model, and provider combinations.",
            output={"candidate_routes": [asdict(route) for route in ROUTES]},
        )
        routing_started = time.perf_counter()
        try:
            decision = JOB_ROUTER.route(route_request)
        except ValueError as error:
            append_trace(
                trace_id,
                actor="Castform orchestrator",
                stage="job.routing_failed",
                summary="The job router rejected the request.",
                output={"error": str(error)},
            )
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"trace_id": trace_id, "error": str(error)},
            )
            return
        router_duration_ms = round(
            (time.perf_counter() - routing_started) * 1000,
            2,
        )

        prediction_payload = [
            asdict(prediction) for prediction in decision.predictions
        ]
        route_payload = asdict(decision.selected_route)
        if decision.cache_hit:
            append_trace(
                trace_id,
                actor="Session router",
                stage="session.pin_reused",
                summary="Reused the complete execution route pinned for this session.",
                input={"session_id": session_id},
                output={"selected_route": route_payload},
            )
        else:
            learned_request = model_request_payload(route_request)
            append_trace(
                trace_id,
                actor="Small LLM router",
                stage="router.candidates_scored",
                summary="Predicted success, token use, and uncertainty for every route.",
                input=learned_request,
                output=model_response_payload(
                    router_model_version=decision.router_model_version,
                    predictions=decision.predictions,
                ),
            )
            append_trace(
                trace_id,
                actor="Decision policy",
                stage="policy.route_selected",
                summary="Selected a complete route using quality threshold and live cost.",
                input={
                    "quality_threshold": decision.quality_threshold,
                    "predictions": prediction_payload,
                    "live_route_costs": {
                        route.route_id: route.estimated_cost_usd for route in ROUTES
                    },
                },
                output={
                    "selected_route": route_payload,
                    "reason": decision.reason,
                    "policy_version": decision.policy_version,
                    "router_duration_ms": router_duration_ms,
                },
            )

        append_trace(
            trace_id,
            actor="Harness dispatcher",
            stage="harness.started",
            summary=f"Started the {decision.selected_route.harness} harness.",
            input={"selected_route": route_payload},
            output={
                "harness": decision.selected_route.harness,
                "status": "running",
            },
        )
        metadata: dict[str, Any] = {
            "trace_id": trace_id,
            "request_id": trace_id,
            "session_id": session_id,
            "castform_route": route_payload,
        }

        litellm_body = {
            "model": decision.selected_route.gateway_model,
            "messages": [{"role": "user", "content": question}],
            "metadata": metadata,
        }
        append_trace(
            trace_id,
            actor=decision.selected_route.harness,
            stage="harness.model_request",
            summary="The selected harness created its first model request.",
            input=litellm_body,
        )
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{LITELLM_BASE_URL}/v1/chat/completions",
            data=json.dumps(litellm_body).encode(),
            headers={
                "Authorization": f"Bearer {LITELLM_MASTER_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                response_body = json.loads(response.read())
                response_headers = dict(response.headers.items())
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read().decode(errors="replace")
            try:
                response_body = json.loads(raw)
            except json.JSONDecodeError:
                response_body = {"error": raw}
            response_headers = dict(error.headers.items())
            status = error.code
        except (urllib.error.URLError, TimeoutError) as error:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            append_trace(
                trace_id,
                actor=decision.selected_route.harness,
                stage="harness.failed",
                summary="The selected harness could not reach LiteLLM.",
                output={"error": str(error), "duration_ms": duration_ms},
            )
            self._json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "selected_route": route_payload,
                    "error": str(error),
                    "events": read_trace(trace_id),
                },
            )
            return

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        visible_headers = {
            key: value
            for key, value in response_headers.items()
            if key.lower().startswith(("x-litellm", "content-type"))
        }
        append_trace(
            trace_id,
            actor=decision.selected_route.harness,
            stage="harness.completed",
            summary="The harness received the model response and completed the demo task.",
            output={
                "status": "completed" if status < 400 else "failed",
                "model_response": response_body,
            },
        )
        append_trace(
            trace_id,
            actor="Trace UI",
            stage="client.response_received",
            summary="Received the completed job and its trace.",
            output={
                "http_status": status,
                "duration_ms": duration_ms,
                "headers": visible_headers,
                "body": response_body,
            },
        )
        events = read_trace(trace_id)
        self._json(
            HTTPStatus.OK if status < 400 else HTTPStatus.BAD_GATEWAY,
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "duration_ms": duration_ms,
                "router_duration_ms": router_duration_ms,
                "router_model_label": ROUTER_MODEL_LABEL,
                "selected_route": route_payload,
                "router_output": model_response_payload(
                    router_model_version=decision.router_model_version,
                    predictions=decision.predictions,
                ),
                "decision": {
                    "reason": decision.reason,
                    "policy_version": decision.policy_version,
                    "quality_threshold": decision.quality_threshold,
                    "cache_hit": decision.cache_hit,
                },
                "response": response_body,
                "events": events,
            },
        )

    def _inspect_github_repository(self) -> None:
        try:
            body = self._request_json()
            repository = parse_github_repo(body.get("repository"))
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository['full_name']}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "CastformRouterLab/0.1",
            },
        )
        warning = None
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                github = json.load(response)
        except urllib.error.HTTPError as error:
            github = {}
            warning = (
                "GitHub could not verify this repository. Private repositories "
                "require a GitHub App installation."
                if error.code in {403, 404}
                else f"GitHub inspection returned HTTP {error.code}."
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            github = {}
            warning = "GitHub inspection was unavailable; the URL was configured locally."

        result = {
            **repository,
            "default_branch": github.get("default_branch") or "main",
            "visibility": github.get("visibility") or "unknown",
            "language": github.get("language"),
            "archived": bool(github.get("archived", False)),
            "verification": "verified_public" if github else "configured_unverified",
        }
        self._json(
            HTTPStatus.OK,
            {"repository": result, "warning": warning},
        )

    def _create_training_environment(self) -> None:
        try:
            body = self._request_json()
            repositories = body.get("repositories")
            selected_route_ids = body.get("selected_route_ids")
            if not isinstance(repositories, list):
                raise ValueError("repositories must be an array")
            if not all(isinstance(repository, dict) for repository in repositories):
                raise ValueError("every repository must be an object")
            if not isinstance(selected_route_ids, list) or not all(
                isinstance(route_id, str) for route_id in selected_route_ids
            ):
                raise ValueError("selected_route_ids must be an array of strings")
            valid_route_ids = {
                route.route_id for route in TRAINING_ROUTE_CATALOG
            }
            if any(
                route_id not in valid_route_ids for route_id in selected_route_ids
            ):
                raise ValueError("selected_route_ids contains an unknown route")
            if any(
                repository.get("verification") != "verified_public"
                for repository in repositories
            ):
                raise ValueError(
                    "private repositories need a GitHub App auth profile; "
                    "configure them with the code-first CLI"
                )
            tasks_per_repo = int(body.get("tasks_per_repo", 10))
            repetitions = int(body.get("repetitions", 1))
            average_run_cost_usd = float(
                body.get("average_run_cost_usd", 1.0)
            )
            privacy_mode = str(
                body.get("privacy_mode", "castform_hosted")
            )
            router_rung = body.get("router_rung", "knn")
            if router_rung not in {"knn", "profile", "baseline"}:
                raise ValueError(
                    "router_rung must be knn, profile, or baseline"
                )
            router_model = body.get(
                "router_model",
                "claude-sonnet-4-6",
            )
            if (
                not isinstance(router_model, str)
                or not router_model.strip()
                or len(router_model) > 200
            ):
                raise ValueError(
                    "router_model must be a non-empty model name"
                )
            router_model = router_model.strip()
            result = build_training_workspace(
                TRAINING_RUNS_ROOT,
                repositories=repositories,
                selected_route_ids=selected_route_ids,
                tasks_per_repo=tasks_per_repo,
                repetitions=repetitions,
                average_run_cost_usd=average_run_cost_usd,
                privacy_mode=privacy_mode,
            )
            workspace = Path(result["workspace_path"])
            project_spec = {
                "schema_version": "1",
                "name": f"ui-{result['workspace_id']}",
                "repositories": [
                    {
                        "repo": repository["full_name"],
                        "revision": repository.get("default_branch") or "main",
                        "visibility": repository.get("visibility") or "public",
                        "auth": {"strategy": "public"},
                    }
                    for repository in repositories
                ],
                "pull_requests": {
                    "limit_per_repo": tasks_per_repo,
                    "eval_ratio": 0.2,
                },
                "allowed_routes": selected_route_ids,
                "benchmark": {
                    "repetitions": repetitions,
                    "average_run_cost_usd": average_run_cost_usd,
                    "execution": privacy_mode,
                },
            }
            (workspace / "project.spec.json").write_text(
                json.dumps(
                    project_spec,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            plan = build_benchmax_plan(
                workspace,
                router_rung=router_rung,
                router_model=router_model,
            )
            plan_path = write_benchmax_plan(workspace, plan)
            result["status"] = "ready_for_benchmax_mining"
            result["files"] = sorted(
                [
                    *result["files"],
                    "project.spec.json",
                    str(plan_path.relative_to(workspace)),
                ]
            )
            display_path = f"training_runs/{result['workspace_id']}"
            result["workspace_path"] = display_path
            result["next_command"] = (
                f"castform-router benchmax {display_path} "
                "--through gate --execute"
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._json(HTTPStatus.CREATED, result)

    @staticmethod
    def _demo_workspace(path: str) -> Path | None:
        prefix = "/api/training/environments/"
        suffix = "/demo"
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        workspace_id = path[len(prefix) : -len(suffix)]
        if WORKSPACE_ID.fullmatch(workspace_id) is None:
            return None
        workspace = TRAINING_RUNS_ROOT / workspace_id
        if not (workspace / "manifest.json").is_file():
            return None
        return workspace

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    @staticmethod
    def _required_string(body: dict[str, Any], key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return value.strip()

    def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _static(self, path: Path) -> None:
        if not path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
            return
        encoded = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    print("[trace-ui] listening on 0.0.0.0:3000", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 3000), Handler).serve_forever()
