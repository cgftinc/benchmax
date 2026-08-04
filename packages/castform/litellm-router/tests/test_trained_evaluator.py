from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from castform_router.trained_evaluator import evaluate_trained_router


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        request = json.loads(body["messages"][1]["content"])
        predictions = []
        for index, route in enumerate(request["candidate_routes"]):
            predictions.append(
                {
                    "route_id": route["route_id"],
                    "success_probability": 0.9 - (index * 0.1),
                    "expected_input_tokens": 100,
                    "expected_cache_read_tokens": 20,
                    "expected_output_tokens": 30,
                }
            )
        content = json.dumps(
            {
                "schema_version": "1",
                "router_model_version": "test-router-v1",
                "predictions": predictions,
            }
        )
        response = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def test_evaluator_emits_benchmax_picks_and_metrics(tmp_path: Path) -> None:
    routes = [
        {
            "route_id": "claude-code/sonnet@anthropic",
            "harness": "claude-code",
            "model": "sonnet",
            "provider": "anthropic",
            "harbor_model": "claude-sonnet-4-6",
        },
        {
            "route_id": "codex/balanced@openai",
            "harness": "codex",
            "model": "balanced",
            "provider": "openai",
            "harbor_model": "gpt-balanced",
        },
    ]
    (tmp_path / "router" / "data").mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"candidate_routes": routes}),
        encoding="utf-8",
    )
    (tmp_path / "router" / "data" / "route_costs.json").write_text(
        json.dumps(
            {
                "routes": {
                    "claude-code/sonnet@anthropic": 0.3,
                    "codex/balanced@openai": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    example = {
        "example_id": "task-1",
        "request": {
            "request_id": "task-1",
            "task": {"text": "Fix it.", "domain": "software_engineering"},
            "user_context": {},
            "workspace_context": {"repository": "acme/api"},
        },
        "target": {
            "predictions": [
                {
                    "route_id": route["route_id"],
                    "success_probability": 0.8,
                    "expected_input_tokens": 100,
                    "expected_cache_read_tokens": 20,
                    "expected_output_tokens": 30,
                }
                for route in routes
            ]
        },
    }
    (tmp_path / "router" / "data" / "eval.jsonl").write_text(
        json.dumps(example) + "\n",
        encoding="utf-8",
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = evaluate_trained_router(
            workspace=tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="test-router",
        )
    finally:
        server.shutdown()
        server.server_close()

    picks = [
        json.loads(line)
        for line in Path(result["picks"]).read_text(encoding="utf-8").splitlines()
    ]
    assert result["router_model_version"] == "test-router-v1"
    assert result["brier_score"] == 0.005
    assert picks[0]["model"] == "claude-sonnet-4-6"
    assert picks[0]["route_id"] == "claude-code/sonnet@anthropic"
