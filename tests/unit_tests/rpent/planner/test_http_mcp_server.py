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
import json
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rpent.dashboard.events import DashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from rpent.utils.logging import init_output_dir

CONCURRENT_CALLS = [
    ("list_dir", {"path": "resources/libero/memory"}),
    ("read_text_file", {"path": "robots/libero/guides/strict_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/pro_hybrid_guide.md"}),
    ("read_text_file", {"path": "robots/libero/guides/env_calibration.md"}),
    ("view_env_state", {"step": 0}),
]


class RecordingSink(DashboardEventSink):
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class FakeToolkit(Toolkit):
    """Minimal toolkit whose tools sleep to widen the overlap window."""

    def __init__(self, state_dir: Path) -> None:
        super().__init__(
            dashboard_events=RecordingSink(),
            state=EnvState(state_dir),
            memory=MemoryManager(state_dir / "memory"),
        )
        self.overlap_errors: list[tuple[str, dict[str, Any]]] = []
        self._register_fake_tools()

    def _register_fake_tools(self) -> None:
        @readonly
        def read_text_file(path: str, max_chars: int = 40000) -> dict:
            time.sleep(0.05)
            p = Path(path)
            return {"path": str(p), "size": 0, "content": "fake content"}

        @readonly
        def list_dir(path: str = "") -> dict:
            time.sleep(0.05)
            return {"path": path, "count": 0, "files": []}

        def view_env_state(step: int = -1) -> dict:
            time.sleep(0.3)
            return {"step": step, "mode": "evaluation"}

        self.add_tool(
            "read_text_file",
            {
                "name": "read_text_file",
                "description": "Read a UTF-8 text file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            read_text_file,
        )
        self.add_tool(
            "list_dir",
            {
                "name": "list_dir",
                "description": "List files in a directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
            list_dir,
        )
        self.add_tool(
            "view_env_state",
            {
                "name": "view_env_state",
                "description": "View the current environment state.",
                "input_schema": {
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                },
            },
            view_env_state,
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        result = super().execute_tool(name, input_dict)
        if result.result.get("error") == "another tool operation is still active":
            self.overlap_errors.append((name, dict(input_dict)))
        return result

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return {"observed": True}

    def solved(self) -> bool:
        return False


def test_http_mcp_server_serializes_concurrent_tool_calls(tmp_path: Path) -> None:
    init_output_dir(tmp_path / "log")
    toolkit = FakeToolkit(tmp_path)
    server = HttpMcpServer(toolkit)
    try:
        url = server.start()
        rejected = asyncio.run(_fire_concurrent(url))
    finally:
        server.stop()

    assert rejected == 0
    assert toolkit.overlap_errors == []


async def _fire_concurrent(url: str) -> int:
    rejected = 0
    async with streamable_http_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            results = await asyncio.gather(
                *(session.call_tool(name, args) for name, args in CONCURRENT_CALLS)
            )
            for result in results:
                if "another tool operation is still active" in json.dumps(
                    result.content, default=str
                ):
                    rejected += 1
    return rejected
