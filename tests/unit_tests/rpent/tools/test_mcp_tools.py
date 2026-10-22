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

from types import SimpleNamespace
from typing import Any

import pytest

from rpent.tools import mcp_tools
from rpent.utils.mcp_servers import McpServerSpec


class FakeToolkit:
    """Minimal Toolkit surface used by ``register_mcp_tools``."""

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self.tools: dict[str, Any] = {
            name: SimpleNamespace(name=name) for name in existing
        }

    def list_tools(self) -> list[Any]:
        return list(self.tools.values())

    def add_tool(self, tool: Any) -> None:
        self.tools[tool.name] = tool


def _remote_tool(
    name: str,
    *,
    description: str = "",
    read_only: bool = False,
    schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
        annotations=SimpleNamespace(readOnlyHint=read_only),
    )


@pytest.fixture
def spec() -> McpServerSpec:
    return McpServerSpec(
        name="facility",
        url="https://facility.example/mcp",
    )


def test_registers_named_tools_with_remote_schemas(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    remote = [
        _remote_tool("set_scene", description="Change the lights."),
        _remote_tool("unused"),
    ]
    monkeypatch.setattr(mcp_tools, "_list_remote_tools", lambda server: remote)
    toolkit = FakeToolkit()

    registered = mcp_tools.register_mcp_tools(
        toolkit, (spec,), tools={"facility": ["set_scene"]}
    )

    assert registered == ("set_scene",)
    adopted = toolkit.tools["set_scene"]
    assert adopted.name == "set_scene"
    assert adopted.description == "Change the lights."
    assert adopted.input_schema == {"type": "object", "properties": {}}
    assert "unused" not in toolkit.tools


def test_read_only_hint_maps_to_the_readonly_marker(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    monkeypatch.setattr(
        mcp_tools,
        "_list_remote_tools",
        lambda server: [
            _remote_tool("get_battery", read_only=True),
            _remote_tool("dock", read_only=False),
        ],
    )
    toolkit = FakeToolkit()

    mcp_tools.register_mcp_tools(
        toolkit, (spec,), tools={"facility": ["get_battery", "dock"]}
    )

    assert toolkit.tools["get_battery"].readonly is True
    assert toolkit.tools["dock"].readonly is False


def test_handler_calls_the_remote_tool_and_renders_its_result(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    monkeypatch.setattr(
        mcp_tools, "_list_remote_tools", lambda server: [_remote_tool("status")]
    )
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def call(server: McpServerSpec, name: str, arguments: dict[str, Any]) -> Any:
        calls.append((server.name, name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            structuredContent={"level": 2},
            isError=False,
        )

    monkeypatch.setattr(mcp_tools, "_call_remote_tool", call)
    toolkit = FakeToolkit()
    mcp_tools.register_mcp_tools(toolkit, (spec,), tools={"facility": ["status"]})

    result = toolkit.tools["status"].handler(detail="high")

    assert result.data == {"text": "ok", "structured_content": {"level": 2}}
    assert calls == [("facility", "status", {"detail": "high"})]


def test_optional_arguments_are_not_forwarded_as_null(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "detail": {"type": "string"},
            "level": {"type": "integer"},
            "modes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["detail"],
    }
    monkeypatch.setattr(
        mcp_tools,
        "_list_remote_tools",
        lambda server: [_remote_tool("status", schema=schema)],
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mcp_tools,
        "_call_remote_tool",
        lambda server, name, arguments: (
            calls.append(arguments)
            or SimpleNamespace(content=[], structuredContent=None, isError=False)
        ),
    )
    toolkit = FakeToolkit()
    mcp_tools.register_mcp_tools(toolkit, (spec,), tools={"facility": ["status"]})
    adopted = toolkit.tools["status"]

    parameters = adopted.args_schema.model_validate({"detail": "high"}, strict=True)
    adopted.handler(
        **{name: getattr(parameters, name) for name in type(parameters).model_fields}
    )

    assert adopted.input_schema == schema
    assert calls == [{"detail": "high"}]


def test_required_arguments_are_validated_locally(spec: McpServerSpec) -> None:
    args = mcp_tools._args_model(
        "status",
        {
            "type": "object",
            "properties": {"detail": {"type": "string"}},
            "required": ["detail"],
        },
    )

    with pytest.raises(ValueError):
        args.model_validate({}, strict=True)


def test_error_results_surface_as_errors(spec: McpServerSpec) -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="door is locked")],
        structuredContent=None,
        isError=True,
    )

    assert mcp_tools._result_payload(result) == {
        "text": "door is locked",
        "error": "door is locked",
    }


def test_rejects_undeclared_server(spec: McpServerSpec) -> None:
    with pytest.raises(ValueError, match="undeclared server"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(), (spec,), tools={"missing": ["status"]}
        )


def test_rejects_unknown_remote_tool(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    monkeypatch.setattr(
        mcp_tools, "_list_remote_tools", lambda server: [_remote_tool("status")]
    )

    with pytest.raises(ValueError, match=r"did not expose tool\(s\) \['dock'\]"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(), (spec,), tools={"facility": ["dock"]}
        )


def test_rejects_name_already_registered_on_the_toolkit(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    monkeypatch.setattr(
        mcp_tools, "_list_remote_tools", lambda server: [_remote_tool("finish")]
    )
    toolkit = FakeToolkit(existing=("finish",))

    with pytest.raises(ValueError, match="already registered"):
        mcp_tools.register_mcp_tools(toolkit, (spec,), tools={"facility": ["finish"]})


def test_rejects_the_same_name_from_two_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mcp_tools, "_list_remote_tools", lambda server: [_remote_tool("status")]
    )

    with pytest.raises(ValueError, match="already registered"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(),
            (
                McpServerSpec(name="one", url="https://one.example/mcp"),
                McpServerSpec(name="two", url="https://two.example/mcp"),
            ),
            tools={"one": ["status"], "two": ["status"]},
        )
