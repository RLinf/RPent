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
import threading
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from rpent.memory import MemoryManager
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, tool

from ._native_helpers import read


@tool
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


def test_http_shared_calls_overlap_and_keep_native_validation(tmp_path: Path) -> None:
    toolkit = Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=threading.Barrier(2),
        output_dir=tmp_path,
        tools=(finish, read),
    )
    server = HttpMcpServer(toolkit)

    async def scenario(url):
        async with (
            httpx.AsyncClient(trust_env=False) as client,
            streamable_http_client(url, http_client=client) as (reader, writer, _),
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                specs = await session.list_tools()
                assert "read_image" not in {tool.name for tool in specs.tools}
                for name in ("read_image", "mcp__rpent__read_image"):
                    rejected = await session.call_tool(name, {"name": "frame.png"})
                    assert rejected.isError
                    assert json.loads(rejected.content[0].text) == {
                        "error": "Unknown tool: read_image"
                    }
                assert (
                    next(t for t in specs.tools if t.name == "read").inputSchema
                    == read.input_schema
                )
                results = await asyncio.gather(
                    session.call_tool("read", {"number": "1", "ctx": "ignored"}),
                    session.call_tool("read", {"number": 2}),
                )
                assert all(not result.isError for result in results)
                assert [json.loads(r.content[0].text)["number"] for r in results] == [
                    1,
                    2,
                ]
                invalid = await session.call_tool("read", {"number": "bad"})
                assert invalid.isError
                failure = json.loads(invalid.content[0].text)
                assert set(failure) == {"error"}
                assert failure["error"].startswith("Invalid arguments for read.")
                assert "number" in failure["error"]
                assert "valid integer" in failure["error"]
                results = await asyncio.gather(
                    *[session.call_tool("list_dir", {}) for _ in range(4)]
                )
                assert all(not r.isError for r in results)

    try:
        asyncio.run(scenario(server.start()))
    finally:
        server.stop()
        toolkit.close()
