"""Tiny OpenAI-compatible upstream used by the local Compose stack."""

from __future__ import annotations

import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from castform_router.trace import append_trace, valid_trace_id


class Handler(BaseHTTPRequestHandler):
    server_version = "CastformMockUpstream/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[mock-upstream] {format % args}", flush=True)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in {
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/responses/input_tokens",
            "/v1/messages",
        }:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            body = self._request_json()
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        if path == "/v1/responses/input_tokens":
            prompt_tokens = max(1, len(json.dumps(body.get("input") or [])) // 4)
            self._json(HTTPStatus.OK, {"input_tokens": prompt_tokens})
            return

        trace_id = valid_trace_id(self.headers.get("x-castform-trace-id"))
        model = str(body.get("model") or "unknown")
        append_trace(
            trace_id,
            actor="Mock provider",
            stage="provider.request_received",
            summary="The selected provider deployment received the rewritten request.",
            input={
                "path": self.path,
                "model": model,
                "messages": body.get("messages"),
                "stream": body.get("stream", False),
            },
        )
        content = self._content(body, model)
        if body.get("stream") is True:
            if path == "/v1/chat/completions":
                self._stream(model, content)
            elif path == "/v1/responses":
                self._stream_responses(model, content)
            elif path == "/v1/messages":
                self._stream_messages(model, content)
            else:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "unsupported mock streaming protocol"},
                )
            return

        prompt_value = body.get("messages", body.get("input", []))
        prompt_tokens = max(1, len(json.dumps(prompt_value)) // 4)
        if path == "/v1/responses":
            response_body = self._responses_body(
                model=model,
                content=content,
                prompt_tokens=prompt_tokens,
            )
        elif path == "/v1/messages":
            response_body = self._messages_body(
                model=model,
                content=content,
                prompt_tokens=prompt_tokens,
            )
        else:
            response_body = self._chat_body(
                model=model,
                content=content,
                prompt_tokens=prompt_tokens,
            )
        append_trace(
            trace_id,
            actor="Mock provider",
            stage="provider.response_created",
            summary="The mock provider generated a protocol-compatible completion.",
            output=response_body,
        )
        self._json(HTTPStatus.OK, response_body)

    @staticmethod
    def _chat_body(
        *,
        model: str,
        content: str,
        prompt_tokens: int,
    ) -> dict[str, Any]:
        response_body = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 6,
                "total_tokens": prompt_tokens + 6,
            },
        }
        return response_body

    @staticmethod
    def _responses_body(
        *,
        model: str,
        content: str,
        prompt_tokens: int,
    ) -> dict[str, Any]:
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": model,
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": False,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt_tokens + 6,
            },
            "user": None,
            "metadata": {},
        }

    @staticmethod
    def _messages_body(
        *,
        model: str,
        content: str,
        prompt_tokens: int,
    ) -> dict[str, Any]:
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": 6,
            },
        }

    @staticmethod
    def _content(body: dict[str, Any], model: str) -> str:
        """Return a contract-valid stand-in for the 0.8B router alias."""

        if model != "qwen35-08b-router":
            return f"Mock response from {model}"
        try:
            messages = body["messages"]
            request = json.loads(messages[-1]["content"])
            routes = request["candidate_routes"]
            response_format = body["response_format"]
            json_schema = response_format["json_schema"]
            schema = json_schema["schema"]
            route_enum = schema["properties"]["predictions"]["items"][
                "properties"
            ]["route_id"]["enum"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid router model request") from error
        route_ids = [route["route_id"] for route in routes]
        if (
            response_format.get("type") != "json_schema"
            or json_schema.get("strict") is not True
            or schema.get("additionalProperties") is not False
            or route_enum != route_ids
        ):
            raise ValueError("router request must include the strict route schema")
        predictions = [
            {
                "route_id": route["route_id"],
                "success_probability": max(
                    0.05,
                    round(0.9 - (index * 0.05), 2),
                ),
                "expected_input_tokens": 1000,
                "expected_cache_read_tokens": 0,
                "expected_output_tokens": 100,
                "uncertainty": 0.25,
                "reason_codes": ["mock_plumbing_only"],
            }
            for index, route in enumerate(routes)
        ]
        return json.dumps(
            {
                "schema_version": "1",
                "router_model_version": "qwen35-08b-untrained-mock-v1",
                "predictions": predictions,
            },
            separators=(",", ":"),
        )

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _stream(self, model: str, content: str) -> None:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        chunks = [
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_responses(self, model: str, content: str) -> None:
        response = self._responses_body(
            model=model,
            content=content,
            prompt_tokens=1,
        )
        item = response["output"][0]
        part = item["content"][0]
        events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 3,
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": content,
                "logprobs": [],
            },
            {
                "type": "response.output_text.done",
                "sequence_number": 4,
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "text": content,
                "logprobs": [],
            },
            {
                "type": "response.content_part.done",
                "sequence_number": 5,
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": part,
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 6,
                "output_index": 0,
                "item": item,
            },
            {
                "type": "response.completed",
                "sequence_number": 7,
                "response": response,
            },
        ]
        self._event_stream(events)

    def _stream_messages(self, model: str, content: str) -> None:
        message_id = f"msg_{uuid.uuid4().hex}"
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
            {"type": "message_stop"},
        ]
        self._event_stream(events)

    def _event_stream(self, events: list[dict[str, Any]]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for event in events:
            self.wfile.write(f"event: {event['type']}\n".encode())
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()


if __name__ == "__main__":
    print("[mock-upstream] listening on 0.0.0.0:8080", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
