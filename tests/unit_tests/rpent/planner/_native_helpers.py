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

import threading
from pathlib import Path
from typing import Any

from mcp import types

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import (
    ToolContext,
    Toolkit,
    ToolResult,
    readonly,
    tool,
)

PNG = b"\x89PNG\r\n\x1a\ncontract-image"


@tool
@readonly
def inspect_scene(detail: str = "low", *, ctx: ToolContext[Any]) -> ToolResult:
    """Inspect the current scene."""
    return ctx.robot.response()


@tool
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Return the outcome configured by the planner test."""
    return ctx.robot.finish(status, summary)


class FakeToolkit(Toolkit):
    def __init__(
        self,
        output: Path,
        result: dict[str, Any] | None = None,
        *,
        images: list[bytes] | None = None,
        accepted: dict[str, str] | None = None,
    ):
        self.result = result if result is not None else {"value": "ok"}
        self.images = images or []
        self.accepted = accepted
        self.calls = []
        self.cancel_calls = 0
        super().__init__(
            state=EnvState(output),
            memory=MemoryManager(output / "memory"),
            robot=self,
            output_dir=output,
            tools=(finish, inspect_scene),
        )

    def response(self):
        data = dict(self.result)
        error = data.pop("error", None)
        return ToolResult(data=data, error=error, images=self.images)

    def finish(self, status: str, summary: str) -> ToolResult:
        result = self.response()
        if not result.is_error:
            accepted = self.accepted or {
                "status": status,
                "summary": summary,
            }
            result.data.update({"_finish": True, **accepted})
        return result

    def execute_tool(self, name, args):
        self.calls.append((name, args))
        return super().execute_tool(name, args)

    def cancel_active_and_wait(self):
        self.cancel_calls += 1
        super().cancel_active_and_wait()


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


async def call_sdk_tool(config, name, arguments):
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    response = await config["instance"].request_handlers[types.CallToolRequest](request)
    return response.root


@tool
@readonly
def read(number: int = 0, *, ctx: ToolContext[threading.Barrier]) -> ToolResult:
    """Read concurrently and return the validated number."""
    ctx.robot.wait(timeout=3)
    return ToolResult(data={"number": number})
