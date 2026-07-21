from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

type ResponseFactory = Callable[[str, int, dict[str, Any]], tuple[int, dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    session_id: str
    call_index: int
    path: str
    authorization: str | None
    body: dict[str, Any]


def completion_response(
    *,
    content: str | None,
    finish_reason: str = "stop",
    model: str = "test-model",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    return {
        "id": "completion-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


class LocalModelServer:
    """Small real HTTP boundary for BaseEnv tests."""

    def __init__(
        self,
        response_factory: ResponseFactory,
        *,
        concurrent_calls: int | None = None,
    ) -> None:
        self._response_factory = response_factory
        self._barrier = (
            threading.Barrier(concurrent_calls)
            if concurrent_calls is not None
            else None
        )
        self._lock = threading.Lock()
        self._call_counts: Counter[str] = Counter()
        self.requests: list[ModelRequest] = []
        self._server = _ModelHttpServer(("127.0.0.1", 0), _ModelHandler, self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="benchmax-test-model-server",
            daemon=True,
        )

    def __enter__(self) -> LocalModelServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def base_url(self, session_id: str) -> str:
        address = self._server.server_address
        return f"http://{address[0]}:{address[1]}/sessions/{session_id}/v1"

    def respond(
        self,
        path: str,
        body: dict[str, Any],
        authorization: str | None,
    ) -> tuple[int, dict[str, Any]]:
        session_id = self._session_id(path)
        with self._lock:
            call_index = self._call_counts[session_id]
            self._call_counts[session_id] += 1
            self.requests.append(
                ModelRequest(
                    session_id=session_id,
                    call_index=call_index,
                    path=path,
                    authorization=authorization,
                    body=body,
                )
            )
        if self._barrier is not None:
            self._barrier.wait(timeout=2)
        return self._response_factory(session_id, call_index, body)

    @staticmethod
    def _session_id(path: str) -> str:
        parts = path.split("/")
        try:
            index = parts.index("sessions")
            return parts[index + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"request did not use a session URL: {path}") from exc


class _ModelHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        owner: LocalModelServer,
    ) -> None:
        super().__init__(address, handler)
        self.owner = owner


class _ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def model_server(self) -> LocalModelServer:
        server = self.server
        assert isinstance(server, _ModelHttpServer)
        return server.owner

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            status, response = self.model_server.respond(
                self.path,
                body,
                self.headers.get("Authorization"),
            )
        except Exception as exc:
            status = 500
            response = {
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "fake_server_error",
                }
            }

        encoded = json.dumps(response).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return
