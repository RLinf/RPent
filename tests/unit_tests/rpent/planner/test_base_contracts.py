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
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import (
    add_mcp_prefix,
    build_api_model,
    build_planner,
    check_planner_mcp_servers,
    check_planner_mcp_support,
    display_tool_name,
)


@pytest.mark.parametrize("source", ["argument", "environment", "override"])
@pytest.mark.parametrize("complete_url", [False, True])
def test_google_endpoint_configuration_preserves_request_url(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    complete_url: bool,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    endpoint = "https://www.packyapi.com/v1beta/models/gemini-4-argon:generateContent"
    configured_url = endpoint if complete_url else "https://www.packyapi.com"
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    if source == "environment":
        monkeypatch.setenv("GEMINI_BASE_URL", configured_url)
    elif source == "override":
        monkeypatch.setenv("GEMINI_BASE_URL", "https://unused.invalid")

    model = build_api_model(
        "google:gemini-4-argon",
        None if source == "environment" else configured_url,
    )

    assert type(model).__name__ == "GoogleModel"
    assert model.model_name == "gemini-4-argon"
    assert model.base_url == "https://www.packyapi.com"

    seen_urls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )

    async def request() -> None:
        api_client = model.client._api_client
        original_client = api_client._async_httpx_client
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        api_client._async_httpx_client = mock_client
        api_client._http_options.httpx_async_client = mock_client
        try:
            response = await model.request(
                [ModelRequest(parts=[UserPromptPart(content="hello")])],
                None,
                ModelRequestParameters(),
            )
            assert response.parts[0].content == "ok"
        finally:
            await mock_client.aclose()
            await original_client.aclose()

    asyncio.run(request())

    assert seen_urls == [endpoint]


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_google_endpoint_rejects_conflicting_explicit_model(monkeypatch, source):
    from rpent.planner.check import LlmCheckRequest, check_llm

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    endpoint = "https://gateway.invalid/v1beta/models/url-model:generateContent"
    monkeypatch.setenv("GEMINI_BASE_URL", endpoint)
    base_url = endpoint if source == "argument" else None
    with pytest.raises(ValueError, match="does not match"):
        build_api_model("google:explicit-model", base_url)
    result = check_llm(
        LlmCheckRequest(planner="api", model="google:explicit-model", base_url=base_url)
    )
    assert result.status == "missing_config"
    assert result.model == "google:explicit-model"
    assert "does not match" in result.detail


def test_add_mcp_prefix_namespaces_bare_names_once() -> None:
    assert add_mcp_prefix("finish") == "mcp__rpent__finish"
    assert add_mcp_prefix("mcp__rpent__finish") == "mcp__rpent__finish"


def test_display_tool_name_shortens_mcp_names() -> None:
    assert display_tool_name("mcp__rpent__finish") == "finish"
    assert display_tool_name("mcp__web__search") == "web:search"
    assert display_tool_name("Read") == "Read"
    # Non-MCP names, including pydantic-ai's ``<server>_<tool>``, pass through.
    assert display_tool_name("web_search") == "web_search"


def test_check_planner_mcp_servers_skips_the_toolkit_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.tools import mcp_tools
    from rpent.utils.mcp_servers import McpServerSpec

    def unreachable(spec):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(mcp_tools, "list_remote_tools", unreachable)

    # A server reserved for the robot's Toolkit is not attached to the
    # planner, so a planner-side connection failure is not this run's.
    assert (
        check_planner_mcp_servers(
            (McpServerSpec(name="dock", url="https://dock.example/mcp"),)
        )
        is None
    )


def test_check_planner_mcp_servers_names_the_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.tools import mcp_tools
    from rpent.utils.mcp_servers import McpServerSpec

    checked: list[str] = []

    def unreachable(spec):
        checked.append(spec.name)
        raise ConnectionError("connection refused")

    monkeypatch.setattr(mcp_tools, "list_remote_tools", unreachable)

    error = check_planner_mcp_servers(
        (
            McpServerSpec(name="web", url="https://web.example/mcp", planner_tools="*"),
            McpServerSpec(
                name="docs", url="https://docs.example/mcp", planner_tools="*"
            ),
        )
    )

    assert error == ("mcp_servers.web: ConnectionError: connection refused")
    assert checked == ["web"]


def test_check_planner_mcp_servers_resolves_env_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.utils.mcp_servers import McpServerSpec

    monkeypatch.delenv("FACILITY_TOKEN", raising=False)

    error = check_planner_mcp_servers(
        (
            McpServerSpec(
                name="facility",
                url="https://facility.example/mcp",
                env_http_headers={"Authorization": "FACILITY_TOKEN"},
                planner_tools="*",
            ),
        )
    )

    assert error is not None and "FACILITY_TOKEN" in error


def test_check_planner_mcp_servers_requires_the_named_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.tools import mcp_tools
    from rpent.utils.mcp_servers import McpServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "list_remote_tools",
        lambda spec: [SimpleNamespace(name="read_power_meter")],
    )

    error = check_planner_mcp_servers(
        (
            McpServerSpec(
                name="facility",
                url="https://facility.example/mcp",
                planner_tools=("read_power_meter", "set_outlet"),
            ),
        )
    )

    # Same contract as Toolkit registration: a name the server does not expose is
    # a configuration error, not a tool the model silently does without.
    assert error == (
        "mcp_servers.facility did not expose tool(s) ['set_outlet']; "
        "available: ['read_power_meter']"
    )


def test_check_planner_mcp_servers_accepts_exposed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.tools import mcp_tools
    from rpent.utils.mcp_servers import McpServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "list_remote_tools",
        lambda spec: [SimpleNamespace(name="read_power_meter")],
    )

    assert (
        check_planner_mcp_servers(
            (
                McpServerSpec(
                    name="facility",
                    url="https://facility.example/mcp",
                    planner_tools=("read_power_meter",),
                ),
            )
        )
        is None
    )


def test_check_planner_mcp_servers_rejects_joined_name_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.tools import mcp_tools
    from rpent.utils.mcp_servers import McpServerSpec

    def exposed(spec) -> list[Any]:
        # ``a`` + ``b_c`` and ``a_b`` + ``c`` both join to ``a_b_c``.
        return [SimpleNamespace(name="b_c" if spec.name == "a" else "c")]

    monkeypatch.setattr(mcp_tools, "list_remote_tools", exposed)
    servers = (
        McpServerSpec(name="a", url="https://a.example/mcp", planner_tools="*"),
        McpServerSpec(name="a_b", url="https://ab.example/mcp", planner_tools="*"),
    )

    error = check_planner_mcp_servers(servers, local_names=set())

    assert error is not None and "'a_b_c'" in error
    assert "mcp_servers.a.b_c" in error and "mcp_servers.a_b.c" in error


def test_check_planner_mcp_servers_rejects_a_toolkit_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.tools import mcp_tools
    from rpent.utils.mcp_servers import McpServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "list_remote_tools",
        lambda spec: [SimpleNamespace(name="search")],
    )
    servers = (
        McpServerSpec(name="web", url="https://web.example/mcp", planner_tools="*"),
    )

    error = check_planner_mcp_servers(servers, local_names={"web_search"})

    assert error is not None and "already an RPent tool" in error
    assert check_planner_mcp_servers(servers, local_names={"finish"}) is None
    # A backend that namespaces tools itself skips the join check.
    assert check_planner_mcp_servers(servers) is None


def test_planner_mcp_support_names_the_backend_and_its_servers() -> None:
    from rpent.utils.mcp_servers import McpServerSpec

    attached = McpServerSpec(
        name="web", url="https://web.example/mcp", planner_tools="*"
    )
    toolkit_only = McpServerSpec(name="power", url="https://power.example/mcp")

    error = check_planner_mcp_support("flash", (attached, toolkit_only))

    assert error is not None and "flash" in error and "web" in error
    assert "power" not in error
    assert check_planner_mcp_support("flash", (toolkit_only,)) is None
    assert check_planner_mcp_support("api", (attached,)) is None


def test_flash_ends_the_run_for_planner_attached_servers(tmp_path: Path) -> None:
    from rpent.utils.mcp_servers import McpServerSpec

    planner = build_planner(
        "flash",
        output_dir=tmp_path,
        recipe_tag="libero_goal_task",
        robot_name="libero",
        dashboard_events=NullDashboardEventSink(),
        mcp_servers=(
            McpServerSpec(name="web", url="https://web.example/mcp", planner_tools="*"),
        ),
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=Mock(),
        max_turns=1,
    )

    assert result.stop_reason == "planner_error"
    assert result.error is not None and "web" in result.error
