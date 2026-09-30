# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic loopback OpenAI Chat server for embodied E2E tests."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OFFLINE_MODEL_NAME = "rpent-offline-e2e"


@dataclass(frozen=True)
class ScriptedToolCall:
    """One tool call returned by the offline planner server."""

    name: str
    arguments: dict[str, Any]


class OfflinePlannerServer:
    """Serve a fixed sequence of OpenAI-compatible tool-call responses."""

    def __init__(self, script: tuple[ScriptedToolCall, ...]) -> None:
        self._script = script
        self._lock = threading.Lock()
        self._request_count = 0

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                owner._handle_request(self)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        host, port = self._httpd.server_address
        self.base_url = f"http://{host}:{port}/v1"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="rpent-offline-planner",
            daemon=True,
        )

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def __enter__(self) -> OfflinePlannerServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("offline planner server did not stop")

    def assert_complete(self) -> None:
        actual = self.request_count
        expected = len(self._script)
        if actual != expected:
            raise RuntimeError(
                f"offline planner received {actual} requests; expected {expected}"
            )

    def _handle_request(self, handler: BaseHTTPRequestHandler) -> None:
        if not handler.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(handler, 404, {"error": {"message": "unknown endpoint"}})
            return

        try:
            length = int(handler.headers.get("Content-Length", "0"))
            request = json.loads(handler.rfile.read(length))
            tool_names = {
                tool.get("function", {}).get("name")
                for tool in request.get("tools", [])
                if tool.get("type") == "function"
            }
            with self._lock:
                index = self._request_count
                if index >= len(self._script):
                    raise RuntimeError("scripted planner call budget exhausted")
                call = self._script[index]
                if call.name not in tool_names:
                    raise RuntimeError(
                        f"tool {call.name!r} was not advertised by RPent"
                    )
                self._request_count += 1
        except Exception as exc:  # noqa: BLE001 - returned as a local API error
            self._send_json(handler, 400, {"error": {"message": str(exc)}})
            return

        response = {
            "id": f"chatcmpl-rpent-e2e-{index + 1}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", OFFLINE_MODEL_NAME),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_rpent_e2e_{index + 1}",
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        self._send_json(handler, 200, response)

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
