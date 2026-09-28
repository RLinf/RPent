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

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from rpent.robots.components.locator_server import (
    JSON_POINT_INSTRUCTIONS,
    ApiBackend,
    CodexBackend,
    manipulation_prompt,
)


@pytest.mark.parametrize(
    "answer, expected",
    [('{"point": [250, 750]}', (250, 750)), ('{"point": null}', None)],
)
def test_api_uses_one_image_and_no_history(monkeypatch, answer, expected):
    calls = []

    def respond(messages, info):
        calls.append(messages)
        assert len(messages) == 1
        content = messages[0].parts[0].content
        assert content[0] == manipulation_prompt("cup") + JSON_POINT_INSTRUCTIONS
        assert isinstance(content[1], BinaryContent)
        assert Image.open(io.BytesIO(content[1].data)).size == (80, 40)
        assert not info.function_tools
        assert not info.model_settings or "timeout" not in info.model_settings
        return ModelResponse(parts=[TextPart(answer)])

    build = Mock(side_effect=lambda *_: FunctionModel(respond))
    monkeypatch.setattr("rpent.planner.base.build_api_model", build)
    backend = ApiBackend("openai:test", base_url="https://example.invalid")
    for _ in range(2):
        result = backend.predict(Image.new("RGB", (80, 40)), manipulation_prompt("cup"))
        assert result.point == expected
        assert result.answer == answer
    assert len(calls) == 2
    build.assert_called_with("openai:test", "https://example.invalid")


@pytest.mark.parametrize(
    "answer",
    [
        "{}",
        '{"point": [1]}',
        '{"point": [true, 2]}',
        "not JSON",
        '{"point": [1, 2], "extra": 3}',
    ],
)
def test_api_rejects_malformed_model_output(monkeypatch, answer):
    model = FunctionModel(lambda *_: ModelResponse(parts=[TextPart(answer)]))
    monkeypatch.setattr("rpent.planner.base.build_api_model", lambda *_: model)
    with pytest.raises(ValueError):
        ApiBackend("openai:test").predict(Image.new("RGB", (8, 8)), "prompt")


@pytest.mark.parametrize(
    "answer, expected",
    [('{"point": [250, 750]}', (250, 750)), ('{"point": null}', None)],
)
@pytest.mark.parametrize("override_base_url", [False, True])
def test_claude_api_image_request(monkeypatch, answer, expected, override_base_url):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["x-api-key"], payload))
            response = json.dumps(
                {
                    "id": "msg_locator_test",
                    "type": "message",
                    "role": "assistant",
                    "model": payload["model"],
                    "content": [{"type": "text", "text": answer}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    monkeypatch.setenv("ANTHROPIC_API_KEY", "locator-test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL",
            "https://example.invalid" if override_base_url else endpoint,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = ApiBackend(
                "anthropic:claude-sonnet-4-6",
                base_url=endpoint if override_base_url else None,
            )
            result = backend.predict(
                Image.new("RGB", (80, 40)), manipulation_prompt("cup")
            )
        finally:
            server.shutdown()
            thread.join(timeout=2)

    assert result.point == expected
    assert result.answer == answer
    assert len(requests) == 1
    path, api_key, payload = requests[0]
    assert path.split("?")[0] == "/v1/messages"
    assert api_key == "locator-test-key"
    assert payload["model"] == "claude-sonnet-4-6"
    assert not payload.get("tools")
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    text, image = message["content"]
    assert text == {
        "type": "text",
        "text": manipulation_prompt("cup") + JSON_POINT_INSTRUCTIONS,
    }
    assert image["type"] == "image"
    source = image["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert Image.open(io.BytesIO(base64.b64decode(source["data"]))).size == (80, 40)


@pytest.mark.parametrize(
    "outcome",
    ["success", "miss", "failed", "empty", "malformed"],
)
def test_codex_image_schema_and_cleanup(monkeypatch, outcome):
    import openai_codex
    from openai_codex.generated.v2_all import TurnStatus

    response = SimpleNamespace(
        status=TurnStatus.failed if outcome == "failed" else TurnStatus.completed,
        error="backend failed" if outcome == "failed" else None,
        final_response={
            "success": '{"point": [125, 875]}',
            "miss": '{"point": null}',
            "empty": None,
            "malformed": "not JSON",
        }.get(outcome, "unused"),
    )

    turn = SimpleNamespace(run=AsyncMock(return_value=response))
    thread = SimpleNamespace(turn=AsyncMock(return_value=turn))
    codex = SimpleNamespace(
        thread_start=AsyncMock(return_value=thread), close=AsyncMock()
    )
    factory = Mock(return_value=codex)
    monkeypatch.setattr(openai_codex, "AsyncCodex", factory)
    from rpent.planner import codex as codex_module

    runner = Mock(wraps=codex_module.run_codex_turn)
    monkeypatch.setattr(codex_module, "run_codex_turn", runner)
    backend = CodexBackend("test-model", reasoning_effort="low")
    if outcome in {"success", "miss"}:
        result = backend.predict(Image.new("RGB", (80, 40)), manipulation_prompt("cup"))
        assert result.point == ((125, 875) if outcome == "success" else None)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            backend.predict(Image.new("RGB", (80, 40)), manipulation_prompt("cup"))
    assert "timeout_s" not in runner.call_args.kwargs
    codex.close.assert_awaited_once()
    options = codex.thread_start.call_args.kwargs
    assert options["ephemeral"] is True
    assert options["model"] == "test-model"
    assert not Path(options["cwd"]).exists()
    inputs = thread.turn.call_args.args[0]
    assert inputs[0].text == manipulation_prompt("cup") + JSON_POINT_INSTRUCTIONS
    encoded = inputs[1].url.split(",", 1)[1]
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (80, 40)
    assert thread.turn.call_args.kwargs["output_schema"]["required"] == ["point"]
