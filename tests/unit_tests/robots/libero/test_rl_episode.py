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

import asyncio
import io
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robots.libero import rl_episode
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool


class RobotToolkit(Toolkit):
    """A deterministic robot boundary, using the real tool dispatcher and state."""

    def __init__(self, path):
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(path),
            memory=MemoryManager(path / "memory"),
        )
        self.success = False
        self.actions = 0
        self.close_count = 0
        self.add_tool(self.move)
        self.get_env_state()

    @tool
    def move(self, succeed: bool) -> ToolResult:
        """Move the test robot."""
        self.actions += 1
        self.success = succeed
        return ToolResult(data={"moved": True})

    @tool
    def look(self) -> ToolResult:
        """Return a camera image."""
        image = io.BytesIO()
        Image.new("RGB", (4, 4)).save(image, format="PNG")
        return ToolResult(data={"looked": True}, images=[image.getvalue()])

    def solved(self) -> bool:
        return self.success

    def get_env_state(self, **kwargs):
        with self.state.record_step(
            state={"actions": self.actions}, terminated=self.success
        ):
            self.state.save("camera.png", np.zeros((4, 4, 3), dtype=np.uint8))
        return ToolResult(data={"actions": self.actions})

    def close(self):
        self.close_count += 1
        super().close()


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    toolkit = RobotToolkit(tmp_path)
    toolkit.add_tool(toolkit.look)
    yield toolkit
    toolkit.close()


def tool_call(name: str, arguments: str = "{}") -> tuple[int, dict]:
    """An OpenAI completion that calls one tool."""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }
    return 200, {
        "id": "completion",
        "object": "chat.completion",
        "created": 0,
        "model": "policy",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def stream_events(completion: dict) -> str:
    """The completion as server-sent events, the way the trainer streams it."""
    choice = completion["choices"][0]
    delta = dict(choice["message"])
    delta["tool_calls"] = [
        {"index": idx, **call} for idx, call in enumerate(delta.get("tool_calls", []))
    ]
    common = {key: completion[key] for key in ("id", "created", "model")}
    common["object"] = "chat.completion.chunk"
    chunks = [
        {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **common,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
            ],
            "usage": completion["usage"],
        },
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


CONTEXT_LIMIT = (
    400,
    {
        "error": {
            "message": "the request leaves no room for output",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    },
)


@pytest.fixture
def policy_endpoint():
    """Start a chat-completions endpoint that answers with ``replies`` in order."""
    servers = []

    def start(replies, delay_s=0.0):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                requests.append(request)
                time.sleep(delay_s)
                status, body = replies[min(len(requests), len(replies)) - 1]
                stream = status == 200 and request.get("stream")
                payload = (stream_events(body) if stream else json.dumps(body)).encode()
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    "text/event-stream" if stream else "application/json",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/v1", requests

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def run_planner(toolkit, base_url, max_turns=3, **options):
    """Run the episode planner against ``base_url`` and return its outcome."""
    settings = {
        "max_tokens": 64,
        "request_timeout_s": 600,
        "episode_timeout_s": 1200,
        "no_images": False,
        "history_compaction": None,
        **options,
    }
    planner, client = rl_episode.build_planner(
        base_url=base_url, model_name="policy", count_tokens=None, **settings
    )
    result = asyncio.run(
        rl_episode.solve(
            planner,
            client,
            toolkit,
            system_prompt="Move the robot.",
            user_message="Move the robot.",
            max_turns=max_turns,
        )
    )
    return result, rl_episode.episode_outcome(result, toolkit, max_turns)


def unused_endpoint() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}/v1"


def test_planner_requests_have_their_own_timeout(toolkit, policy_endpoint):
    # Three slow replies together outlast one request's timeout.
    base_url, requests = policy_endpoint([tool_call("move", '{"succeed": false}')], 0.4)
    _, status = run_planner(toolkit, base_url, request_timeout_s=1)
    assert len(requests) == 3 and toolkit.actions == 3
    assert status["terminal"] and status["termination_reason"] == "request_limit"
    assert requests[0]["tools"] and requests[0]["parallel_tool_calls"] is False


def test_rejected_context_ends_the_episode_without_a_retry(toolkit, policy_endpoint):
    base_url, requests = policy_endpoint([CONTEXT_LIMIT])
    _, status = run_planner(toolkit, base_url)
    assert len(requests) == 1 and toolkit.actions == 0
    assert status["terminal"] and not status["success"]
    assert status["termination_reason"] == status["error_kind"] == "context_limit"


def test_episode_deadline_drains_tools_and_returns_native_failure(
    toolkit, tmp_path, policy_endpoint
):
    stopped = threading.Event()

    @tool
    def slow_move() -> ToolResult:
        """Move until the episode budget interrupts the robot."""
        try:
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.add_tool(slow_move)
    base_url, requests = policy_endpoint([tool_call("slow_move")])
    result, status = run_planner(toolkit, base_url, episode_timeout_s=1)
    assert len(requests) == 1 and stopped.is_set()
    assert status["terminal"] and not status["success"]
    assert status["termination_reason"] == status["error_kind"] == "episode_timeout"
    rl_episode.write_transcript(tmp_path, result)
    recorded = json.loads((tmp_path / "planner.json").read_text())
    assert recorded["stats"]["turns_used"] == 1
    assert recorded["error_kind"] == "episode_timeout"


def test_unreachable_endpoint_ends_without_a_task_outcome(toolkit):
    _, status = run_planner(toolkit, unused_endpoint())
    assert not status["terminal"] and status["error_kind"] == "model_transport"


@pytest.mark.parametrize("no_images", [False, True])
def test_episode_can_withhold_tool_images(toolkit, policy_endpoint, no_images):
    base_url, requests = policy_endpoint([tool_call("look")])
    run_planner(toolkit, base_url, max_turns=2, no_images=no_images)
    parts = [
        part
        for message in requests[1]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    images = [part for part in parts if part.get("type") == "image_url"]
    assert bool(images) is not no_images
    assert ("Image omitted" in json.dumps(requests[1])) is no_images


@pytest.mark.parametrize(
    "success,truncated,turns,error,reason,terminal,error_kind",
    [
        (True, False, 2, None, "success", True, None),
        (True, False, 2, "backend failed after success", "success", True, None),
        (
            False,
            False,
            2,
            "output truncated",
            "response_truncated",
            True,
            "response_truncated",
        ),
        (False, False, 2, "unknown tool", "tool_name", True, "tool_name"),
        (False, False, 2, "no room", "context_limit", True, "context_limit"),
        (
            False,
            False,
            2,
            "planner timed out after 1800s",
            "episode_timeout",
            True,
            "episode_timeout",
        ),
        (False, True, 2, None, "environment_limit", True, None),
        (False, False, 4, None, "request_limit", True, None),
        (False, False, 2, None, "planner_completed", True, None),
        (False, False, 2, "backend failed", "error", False, None),
        (
            False,
            False,
            2,
            "invalid arguments",
            "tool_arguments",
            True,
            "tool_arguments",
        ),
    ],
)
def test_planner_episode_terminal_contract(
    success, truncated, turns, error, reason, terminal, error_kind
):
    toolkit = SimpleNamespace(
        solved=lambda: success,
        state=SimpleNamespace(get=lambda: SimpleNamespace(truncated=truncated)),
    )
    result = SimpleNamespace(
        stats={"turns_used": turns}, error=error, error_kind=error_kind
    )
    outcome = rl_episode.episode_outcome(result, toolkit, max_turns=4)
    assert outcome["terminal"] is terminal
    assert outcome["termination_reason"] == reason
    assert outcome["success"] is success
