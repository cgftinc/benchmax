"""Judge auth-retry: the credential is re-resolved *inside* the retry loop.

Marked `integration` because it drives the real ``openai`` SDK HTTP path end to
end — a 401 has to surface as a genuine ``openai.AuthenticationError`` for the
retry-with-reconstruction to trigger. Unlike the other integration tests it does
NOT hit a live API: it stands up a local OpenAI-compatible stub server, the only
way to script a deterministic ``401 -> fresh-token -> 200`` transition. No
credentials required (the test supplies a rotating runtime auth provider).

Run: uv run pytest tests/integration/test_rubric_auth_retry.py -v -m integration
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from benchmax.auth import ModelRequestContext
from benchmax.rewards import Judge, JudgeError, Rubric, evaluate_single_rubric

_VALID_TOKEN = "fresh-token"


def _make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):  # keep test output clean
            pass

        def do_POST(self):
            auth = self.headers.get("Authorization", "")
            state["requests"].append(auth)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))

            if auth != f"Bearer {_VALID_TOKEN}":
                self._respond(
                    401,
                    {
                        "error": {
                            "message": "invalid token",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    },
                )
                return

            content = json.dumps({"score": 1, "reasoning": "ok"})
            self._respond(
                200,
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "stub",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        def _respond(self, status: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


@pytest.fixture
def stub_server():
    state: dict = {"requests": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _rubric() -> Rubric:
    return Rubric(title="T", description="D")


class _RotatingAuth:
    def __init__(self, tokens):
        self._tokens = iter(tokens)

    async def headers_for_request(self, context: ModelRequestContext):
        del context
        return {"Authorization": f"Bearer {next(self._tokens)}"}


@pytest.mark.integration
async def test_auth_retry_refreshes_token_in_loop(stub_server):
    base_url, state = stub_server
    # Attempt 1 hands the server a stale token (401); the rebuilt client picks up
    # a fresh token from the provider and attempt 2 succeeds.
    judge = Judge(
        model="stub",
        base_url=base_url,
        auth=_RotatingAuth(["stale-token", _VALID_TOKEN]),
    )
    result = await evaluate_single_rubric(
        rubric=_rubric(),
        question="q",
        response="r",
        judge=judge,
    )
    assert result.score == 1
    assert state["requests"] == ["Bearer stale-token", f"Bearer {_VALID_TOKEN}"]


@pytest.mark.integration
async def test_persistent_401_raises_judge_error(stub_server):
    base_url, state = stub_server
    with pytest.raises(JudgeError, match="401"):
        await evaluate_single_rubric(
            rubric=_rubric(),
            question="q",
            response="r",
            judge=Judge(
                model="stub",
                base_url=base_url,
                auth=_RotatingAuth(["always-stale"] * 3),
            ),
        )
    # 1 initial + 2 rebuild-and-retry attempts, all 401.
    assert len(state["requests"]) == 3
