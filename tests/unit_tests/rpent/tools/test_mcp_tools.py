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
import base64
import json
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import types

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.session import EnvState, Payload
from rpent.tools import Toolkit, ToolResult, mcp_tools
from rpent.tools import tool as declare_tool
from rpent.tools.base import MAX_TOOL_TEXT_BYTES
from rpent.utils.logging import init_output_dir
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


def _registered(
    monkeypatch: pytest.MonkeyPatch,
    spec: McpServerSpec,
    result: Any = None,
    *,
    name: str = "status",
    call: Callable[..., Any] | None = None,
    toolkit: Any = None,
) -> Any:
    """Register one remote tool that answers with ``result`` and return it.

    ``call`` replaces the fake remote call when the test inspects the forwarded
    arguments; ``toolkit`` registers on a real Toolkit instead of the fake one.
    """
    monkeypatch.setattr(
        mcp_tools, "list_remote_tools", lambda server: [_remote_tool(name)]
    )
    monkeypatch.setattr(
        mcp_tools, "_call_remote_tool", call or (lambda srv, tool, arguments: result)
    )
    target = toolkit if toolkit is not None else FakeToolkit()
    mcp_tools.register_mcp_tools(target, (spec,), tools={"facility": [name]})
    return next(tool for tool in target.list_tools() if tool.name == name)


def test_registers_named_tools_with_remote_schemas(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    remote = [
        _remote_tool("set_scene", description="Change the lights."),
        _remote_tool("unused"),
    ]
    monkeypatch.setattr(mcp_tools, "list_remote_tools", lambda server: remote)
    toolkit = FakeToolkit()

    registered = mcp_tools.register_mcp_tools(
        toolkit, (spec,), tools={"facility": ["set_scene"]}
    )

    assert registered == ("set_scene",)
    tool = toolkit.tools["set_scene"]
    assert tool.name == "set_scene"
    assert tool.description == "Change the lights."
    assert tool.input_schema == {"type": "object", "properties": {}}
    assert "unused" not in toolkit.tools


def test_coordinated_tools_ignore_the_server_read_only_hint(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    # A third-party server that annotates a side-effecting tool as read-only
    # must not let it bypass the exclusive admission and observation capture
    # that coordinated registration exists to guarantee.
    monkeypatch.setattr(
        mcp_tools,
        "list_remote_tools",
        lambda server: [
            _remote_tool("get_battery", read_only=True),
            _remote_tool("dock", read_only=False),
        ],
    )
    toolkit = FakeToolkit()

    mcp_tools.register_mcp_tools(
        toolkit, (spec,), tools={"facility": ["get_battery", "dock"]}
    )

    assert toolkit.tools["get_battery"].readonly is False
    assert toolkit.tools["dock"].readonly is False


def test_handler_calls_the_remote_tool_and_renders_its_result(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def call(server: McpServerSpec, name: str, arguments: dict[str, Any]) -> Any:
        calls.append((server.name, name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            structuredContent={"level": 2},
            isError=False,
        )

    tool = _registered(monkeypatch, spec, call=call)

    result = tool.handler(detail="high")

    assert result.data == {"text": "ok", "structured_content": {"level": 2}}
    assert calls == [("facility", "status", {"detail": "high"})]


def test_a_registered_tool_bounds_what_the_toolkit_records(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    # The handler's payload is what the Toolkit persists, so a remote server
    # must not be able to grow the trajectory or a Dashboard step past the
    # same bound a local tool result has.
    oversized = "x" * (MAX_TOOL_TEXT_BYTES + 1000)
    tool = _registered(
        monkeypatch,
        spec,
        SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text=oversized),
                # Server-controlled metadata is persisted too.
                types.ResourceLink(
                    type="resource_link",
                    name="log",
                    uri="https://docs.example/log",
                    description=oversized,
                ),
            ],
            structuredContent={"rows": oversized},
            isError=False,
        ),
        name="dump",
    )

    result = tool.handler()

    assert result.data["text"].endswith("\n[truncated]")
    assert len(result.data["text"].encode("utf-8")) <= MAX_TOOL_TEXT_BYTES
    # Large structured content and block metadata are kept as capped JSON text;
    # small values stay data (see the "ok" / metadata tests above).
    for key in ("content", "structured_content"):
        assert result.data[key].endswith("\n[truncated]")
        assert len(result.data[key].encode("utf-8")) <= MAX_TOOL_TEXT_BYTES


def test_every_png_block_becomes_a_tool_result_image(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    blobs = [b"\x89PNG\r\n\x1a\nfirst", b"\x89PNG\r\n\x1a\nsecond"]
    images = [
        SimpleNamespace(
            type="image",
            mimeType="image/png",
            data=base64.b64encode(blob).decode("ascii"),
        )
        for blob in blobs
    ]
    tool = _registered(
        monkeypatch,
        spec,
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="captured"), *images],
            structuredContent=None,
            isError=False,
        ),
        name="snapshot",
    )

    result = tool.handler()

    assert result.images == blobs
    assert result.data == {"text": "captured"}


def test_non_png_images_do_not_embed_base64(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    block = SimpleNamespace(
        type="image",
        mimeType="image/jpeg",
        data="jpegdata",
        model_dump=lambda **kwargs: {
            "type": "image",
            "mimeType": "image/jpeg",
            "data": "jpegdata",
        },
    )
    tool = _registered(
        monkeypatch,
        spec,
        SimpleNamespace(content=[block], structuredContent=None, isError=False),
        name="snapshot",
    )

    result = tool.handler()

    assert result.images == []
    assert result.data == {
        "content": [
            {
                "type": "image",
                "mimeType": "image/jpeg",
                "notice": "only text and image/png content is forwarded to the model",
            }
        ]
    }


@pytest.mark.parametrize("payload", ["abc", "AAAA@@@@"])
def test_invalid_base64_image_keeps_the_text_result(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec, payload: str
) -> None:
    tool = _registered(
        monkeypatch,
        spec,
        SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="captured"),
                SimpleNamespace(type="image", mimeType="image/png", data=payload),
            ],
            structuredContent=None,
            isError=False,
        ),
        name="snapshot",
    )

    result = tool.handler()

    assert result.images == []
    assert result.data == {"text": "captured"}


def test_binary_and_resource_blocks_keep_only_metadata() -> None:
    result = types.CallToolResult(
        content=[
            types.AudioContent(type="audio", data="YXVkaW8=", mimeType="audio/wav"),
            types.EmbeddedResource(
                type="resource",
                resource=types.BlobResourceContents(
                    uri="https://docs.example/archive",
                    blob="YmluYXJ5",
                    mimeType="application/octet-stream",
                ),
            ),
            types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(
                    uri="https://docs.example/notes",
                    text="not forwarded",
                    mimeType="text/plain",
                ),
            ),
        ],
        isError=False,
    )

    payload = mcp_tools.result_payload(result)

    # Trajectory saving dumps this payload with a strict json.dump.
    json.dumps(payload)
    audio, resource, notes = payload["content"]
    assert audio == {
        "type": "audio",
        "mimeType": "audio/wav",
        "notice": "only text and image/png content is forwarded to the model",
    }
    assert resource["resource"]["uri"] == "https://docs.example/archive"
    assert "blob" not in resource["resource"]
    assert notes["resource"]["uri"] == "https://docs.example/notes"
    assert "text" not in notes["resource"]


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
        "list_remote_tools",
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
    tool = toolkit.tools["status"]

    parameters = tool.args_schema.model_validate({"detail": "high"}, strict=True)
    tool.handler(
        **{name: getattr(parameters, name) for name in type(parameters).model_fields}
    )

    assert tool.input_schema == schema
    assert calls == [{"detail": "high"}]


def test_required_arguments_are_validated_locally() -> None:
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


def test_error_results_surface_as_errors() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="door is locked")],
        structuredContent=None,
        isError=True,
    )

    assert mcp_tools.result_payload(result) == {
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
        mcp_tools, "list_remote_tools", lambda server: [_remote_tool("status")]
    )

    with pytest.raises(ValueError, match=r"did not expose tool\(s\) \['dock'\]"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(), (spec,), tools={"facility": ["dock"]}
        )


@pytest.mark.parametrize("remote_name", ["finish", "Finish"])
def test_rejects_name_already_registered_on_the_toolkit(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec, remote_name: str
) -> None:
    # Compared case-insensitively, so an external ``Finish`` cannot shadow the
    # toolkit's ``finish``.
    monkeypatch.setattr(
        mcp_tools, "list_remote_tools", lambda server: [_remote_tool(remote_name)]
    )

    with pytest.raises(ValueError, match="already registered"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(existing=("finish",)),
            (spec,),
            tools={"facility": [remote_name]},
        )


def test_rejects_the_same_name_from_two_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mcp_tools, "list_remote_tools", lambda server: [_remote_tool("status")]
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


def test_additional_properties_follow_the_remote_schema() -> None:
    permissive = mcp_tools._args_model(
        "status", {"type": "object", "properties": {"detail": {"type": "string"}}}
    )
    strict = mcp_tools._args_model(
        "status",
        {
            "type": "object",
            "properties": {"detail": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    # The remote server stays the authority unless it forbids extra keys.
    assert permissive.model_validate(
        {"detail": "high", "bogus": 1}, strict=True
    ).model_dump(exclude_unset=True) == {"detail": "high", "bogus": 1}
    with pytest.raises(ValueError):
        strict.model_validate({"detail": "high", "bogus": 1}, strict=True)


def test_reserved_property_names_survive_a_strict_schema() -> None:
    # ``model_dump`` repeats a BaseModel member, ``schema`` shadows a deprecated
    # one, and ``_id``/``model_config`` cannot be fields at all. All of them are
    # legal arguments for the remote server, so none may abort registration.
    schema = {
        "type": "object",
        "properties": {
            "_id": {"type": "string"},
            "model_config": {"type": "string"},
            "schema": {"type": "string"},
            "model_dump": {"type": "string"},
        },
        "required": ["_id"],
        "additionalProperties": False,
    }
    args = mcp_tools._args_model("lookup", schema)
    arguments = {
        "_id": "abc",
        "model_config": "x",
        "schema": "public",
        "model_dump": "payload",
    }

    assert args.model_validate(arguments, strict=True).model_dump() == arguments
    with pytest.raises(ValueError, match="undeclared argument"):
        args.model_validate({"_id": "abc", "bogus": 1}, strict=True)


@pytest.mark.parametrize(
    "property_schema, accepted, rejected",
    [
        ({"type": ["string", "null"]}, ["high", None], [1]),
        ({"type": ["integer", "string"]}, [3, "three"], [1.5]),
        ({"type": "string", "enum": ["low", "high"]}, ["high"], ["medium"]),
        ({"type": "array", "items": {"type": "string"}}, [["a", "b"]], [[1]]),
    ],
)
def test_argument_types_follow_the_remote_schema(
    property_schema: dict[str, Any],
    accepted: list[Any],
    rejected: list[Any],
) -> None:
    args = mcp_tools._args_model(
        "status",
        {
            "type": "object",
            "properties": {"value": property_schema},
            "required": ["value"],
        },
    )

    for value in accepted:
        args.model_validate({"value": value}, strict=True)
    for value in rejected:
        with pytest.raises(ValueError):
            args.model_validate({"value": value}, strict=True)


class _ListSession:
    """MCP session whose ``tools/list`` answer comes from ``page``."""

    def __init__(self, page: Callable[[str | None], Any]) -> None:
        self.page = page
        self.cursors: list[str | None] = []

    async def list_tools(self, cursor: str | None = None) -> Any:
        self.cursors.append(cursor)
        return self.page(cursor)


def _serve_list(monkeypatch: pytest.MonkeyPatch, session: Any) -> Any:
    """Make ``list_remote_tools`` talk to ``session`` instead of a server."""
    monkeypatch.setattr(
        mcp_tools, "_run_remote", lambda server, action: asyncio.run(action(session))
    )
    return session


def test_tools_list_follows_server_pagination(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    pages = {
        None: SimpleNamespace(tools=[_remote_tool("first")], nextCursor="page-2"),
        "page-2": SimpleNamespace(tools=[_remote_tool("second")], nextCursor=None),
    }
    session = _serve_list(monkeypatch, _ListSession(pages.__getitem__))

    assert [tool.name for tool in mcp_tools.list_remote_tools(spec)] == [
        "first",
        "second",
    ]
    assert session.cursors == [None, "page-2"]


def test_a_repeated_cursor_ends_pagination(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    session = _serve_list(
        monkeypatch,
        _ListSession(
            lambda cursor: SimpleNamespace(
                tools=[_remote_tool("only")], nextCursor="page-2"
            )
        ),
    )

    # The second page repeats its own cursor; pagination must stop there
    # instead of requesting it forever.
    assert [tool.name for tool in mcp_tools.list_remote_tools(spec)] == [
        "only",
        "only",
    ]
    assert session.cursors == [None, "page-2"]


def test_tools_list_gives_up_after_the_request_deadline(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "0.05")

    class EndlessSession:
        async def list_tools(self, cursor: str | None = None) -> Any:
            await asyncio.sleep(0.01)
            return SimpleNamespace(tools=[], nextCursor=f"{cursor or 'page-0'}+1")

    _serve_list(monkeypatch, EndlessSession())

    with pytest.raises(TimeoutError, match="tools/list did not finish"):
        mcp_tools.list_remote_tools(spec)


def test_call_timeout_defaults_to_sixty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_CALL_TIMEOUT_S", raising=False)
    assert mcp_tools.call_timeout_s() == 60.0

    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "5")
    assert mcp_tools.call_timeout_s() == 5.0


@pytest.mark.parametrize("raw", ["abc", "0", "-1"])
def test_call_timeout_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", raw)

    with pytest.raises(ValueError, match="MCP_CALL_TIMEOUT_S"):
        mcp_tools.call_timeout_s()


class _ServerToolkit(Toolkit):
    """Toolkit whose state capture answers, so a real server can return results."""

    def get_env_state(self, *, command, result, elapsed_s):  # noqa: ANN001
        return ToolResult(data={"observed": True})


def _toolkit(root: Path) -> Toolkit:
    state = EnvState(root)
    state.start_task(instruction="Sense.", config=Payload("test", {}))
    return _ServerToolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(root / "memory"),
    )


def test_closing_a_session_does_not_terminate_it_on_the_server(
    monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    opened: list[dict[str, Any]] = []

    class FakeSession:
        def __init__(self, read: Any, write: Any, read_timeout_seconds: Any = None):
            self.read = read
            self.write = write
            self.read_timeout_seconds = read_timeout_seconds

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *exc_info: Any) -> bool:
            return False

        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def fake_client(url: str, headers: Any = None, **kwargs: Any):
        opened.append({"url": url, "headers": headers, **kwargs})
        yield object(), object(), None

    monkeypatch.setattr(mcp_tools, "_mcp_client", lambda: (FakeSession, fake_client))

    async def action(session: Any) -> str:
        return "done"

    assert asyncio.run(mcp_tools.run_remote(spec, action)) == "done"
    # The transport's DELETE waits for the server to finish the in-flight call,
    # so a timed-out call would otherwise keep running for the tool's runtime.
    assert opened[0]["terminate_on_close"] is False


def test_registered_tool_deadline_ends_a_slow_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "0.2")
    init_output_dir(tmp_path / "log")
    server_toolkit = _toolkit(tmp_path)

    @declare_tool(readonly=True)
    def slow_sense() -> ToolResult:
        time.sleep(3)
        return ToolResult(data={"observed": True})

    server_toolkit.add_tool(slow_sense)
    server = HttpMcpServer(server_toolkit)
    registered = _toolkit(tmp_path)
    try:
        url = server.start()
        mcp_tools.register_mcp_tools(
            registered,
            (McpServerSpec(name="facility", url=url),),
            tools={"facility": ["slow_sense"]},
        )
        started = time.monotonic()
        result = registered.execute_tool("slow_sense", {})
        elapsed = time.monotonic() - started
    finally:
        server.stop()
        server_toolkit.close()
        registered.close()

    assert "Timed out" in result.data["error"]
    assert elapsed < 3


def test_registered_tool_reaches_a_real_server(tmp_path: Path) -> None:
    init_output_dir(tmp_path / "log")
    server_toolkit = _toolkit(tmp_path)
    received: list[str] = []

    @declare_tool(readonly=True)
    def get_level(label: str) -> ToolResult:
        received.append(label)
        return ToolResult(data={"label": label})

    server_toolkit.add_tool(get_level)
    server = HttpMcpServer(server_toolkit)
    registered = _toolkit(tmp_path)
    try:
        url = server.start()
        mcp_tools.register_mcp_tools(
            registered,
            (McpServerSpec(name="facility", url=url),),
            tools={"facility": ["get_level"]},
        )
        result = registered.execute_tool("get_level", {"label": "high"})
    finally:
        server.stop()
        server_toolkit.close()
        registered.close()

    assert received == ["high"]
    assert not result.is_error


def test_registered_tool_forwards_arguments_the_schema_admits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: McpServerSpec
) -> None:
    """The remote schema, not the local model, decides which keys reach it."""
    init_output_dir(tmp_path / "log")
    registered = _toolkit(tmp_path)
    sent: list[dict[str, Any]] = []

    def call(server, name, arguments):  # noqa: ANN001
        sent.append(arguments)
        return SimpleNamespace(content=[], isError=False, structuredContent=None)

    try:
        # A schema without declared properties is how most servers express a
        # free form request; every argument the model sends is valid under it.
        _registered(monkeypatch, spec, name="set_level", call=call, toolkit=registered)
        result = registered.execute_tool("set_level", {"level": "high", "fade_s": 2})
    finally:
        registered.close()

    assert sent == [{"level": "high", "fade_s": 2}]
    assert not result.is_error


def test_rejects_registering_from_a_server_the_planner_attaches_whole() -> None:
    with pytest.raises(ValueError, match="planner_tools is '\\*'"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(),
            (
                McpServerSpec(
                    name="facility",
                    url="https://facility.example/mcp",
                    planner_tools="*",
                ),
            ),
            tools={"facility": ["dock"]},
        )


def test_rejects_registering_a_tool_the_planner_also_attaches() -> None:
    with pytest.raises(ValueError, match="also attached to the planner"):
        mcp_tools.register_mcp_tools(
            FakeToolkit(),
            (
                McpServerSpec(
                    name="facility",
                    url="https://facility.example/mcp",
                    planner_tools=("read_power_meter", "dock"),
                ),
            ),
            tools={"facility": ["dock"]},
        )
