"""Shared LIBERO prompt fragments."""

from __future__ import annotations

MCP_RUNTIME_ADAPTER = """
CURRENT MCP RUNTIME ADAPTER:
- The environment server is already running and managed by the runner.
- Do not start, stop, restart, or otherwise manage `env_server.py`.
- Do not issue file-based driver commands.
- Do not emit plain-text pseudo tool calls such as `<tool_call>`, `[tool_use:]`,
  or JSON action commands.
- For images, use Claude Code's structured `Read` tool.
- For physical actions, call the real `rpent` MCP tools exposed by the
  runtime.
- If a guide source mentions file-driver commands, action dictionaries, or
  legacy command files, translate the intended primitive into structured MCP
  tool calls. Preserve the guide's strategy, constraints, parameters, and
  recovery advice; only the command format is legacy.
- Only call `finish` with success status after the latest
  state has `libero_terminated == true`.
"""

GUIDE_READ_INSTRUCTIONS = """
GUIDE SOURCE FILES TO READ:
- robots/libero/guides/strict_hybrid_guide.md
- robots/libero/guides/pro_hybrid_guide.md
- robots/libero/guides/env_calibration.md

At the start of each run, read all three guide source files once with Claude
Code's structured `Read` tool before issuing the first physical command. Treat
them as strategy and calibration references. If they mention legacy command
examples or older command formats, use the MCP runtime adapter: keep the
strategy, but call current runtime tools by the names shown in the tool list.
"""
