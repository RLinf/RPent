# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

"""Per-decision RoboDojo MCP tools backed by isolated observation/action files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent


def create_server(snapshot_path: Path, action_path: Path) -> FastMCP:
    server = FastMCP("robodojo")

    @server.tool()
    def snapshot() -> CallToolResult:
        """View the latest RoboDojo cameras, robot state, feedback, and task demonstration."""
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        content = []
        for item in data["content"]:
            if item["type"] == "image":
                content.append(
                    ImageContent(type="image", data=item["data"], mimeType="image/jpeg")
                )
            else:
                content.append(TextContent(type="text", text=item["text"]))
        return CallToolResult(content=content)

    @server.tool()
    def submit_action(
        commands: list[str],
        scene: str = "",
        progress: str = "",
        memory: str = "",
        plan: str = "",
    ) -> str:
        """Submit up to four RoboDojo commands for execution after this decision."""
        if (
            not commands
            or len(commands) > 4
            or not all(isinstance(c, str) and c.strip() for c in commands)
        ):
            raise ValueError("submit 1-4 nonempty discrete command strings")
        payload = {
            "scene": scene,
            "progress": progress,
            "memory": memory,
            "plan": plan,
            "commands": commands,
        }
        pending = action_path.with_suffix(".pending")
        pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        pending.replace(action_path)
        return "Action submitted. Call finish now."

    return server


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m rpent.benchmarks.robodojo_mcp SNAPSHOT ACTION"
        )
    create_server(Path(sys.argv[1]), Path(sys.argv[2])).run(transport="stdio")


if __name__ == "__main__":
    main()
