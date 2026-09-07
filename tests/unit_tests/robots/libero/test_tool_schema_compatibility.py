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

"""Compare complete exported declarations with the pre-migration snapshot.

The fixture was extracted from commit 949f61ecea1c571168ca6a29a59f6d3207f6f947:
common/LIBERO TOOLS_SPEC, memory descriptions, and the API image-reader wrapper.
Tests do not depend on Git history or regenerate expected schemas from Args.
"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from rpent.planner.api_loop import _build_tools
from rpent.planner.claude_code import _build_rpent_server
from rpent.planner.utils.http_mcp_server import HttpMcpServer

_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/pre_native_tools.json").read_text()
)


@pytest.fixture(params=["evaluation", "exploration"])
def toolkit_and_specs(request, make_toolkit):
    toolkit, _, _ = make_toolkit(mode=request.param)
    specs = _BASELINE["common"] + [
        spec
        for spec in _BASELINE["libero"]
        if request.param == "exploration" or spec["name"] != "reset"
    ]
    specs = deepcopy(specs)
    # list_dir now resolves its default directory from each invocation context.
    directory = next(spec for spec in specs if spec["name"] == "list_dir")
    directory["description"] = directory["description"].replace(
        " Default = {{output_dir}}.", ""
    )
    directory["input_schema"]["properties"]["path"]["description"] = (
        "Directory path. Defaults to the current task's output directory."
    )
    return toolkit, specs


def _mcp_specs(tools):
    return [
        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
        for t in tools
    ]


def test_native_declarations_match_before_migration(toolkit_and_specs):
    toolkit, expected = toolkit_and_specs
    exported = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in toolkit.list_tools()
    ]
    assert exported == [
        *expected[:3],
        _BASELINE["api_read_image"]["vision"],
        *expected[3:],
    ]


@pytest.mark.parametrize("no_images", [False, True])
def test_api_declarations_match_before_migration(toolkit_and_specs, no_images):
    toolkit, expected = toolkit_and_specs
    exported = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.function_schema.json_schema,
        }
        for t in _build_tools(toolkit, no_images=no_images)
    ]
    reader = _BASELINE["api_read_image"]["text_only" if no_images else "vision"]
    assert exported == [reader, *expected]


def test_claude_mcp_declarations_match_before_migration(toolkit_and_specs):
    toolkit, expected = toolkit_and_specs
    server = _build_rpent_server(toolkit=toolkit)["instance"]
    response = asyncio.run(
        server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest(method="tools/list")
        )
    )
    assert _mcp_specs(response.root.tools) == expected


def test_codex_http_declarations_match_before_migration(toolkit_and_specs):
    toolkit, expected = toolkit_and_specs
    server = HttpMcpServer(toolkit)

    async def fetch(url):
        async with (
            httpx.AsyncClient(trust_env=False) as client,
            streamable_http_client(url, http_client=client) as (reader, writer, _),
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                return _mcp_specs((await session.list_tools()).tools)

    try:
        assert asyncio.run(fetch(server.start())) == expected
    finally:
        server.stop()
