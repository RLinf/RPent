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

"""Gemini-specific command, workspace, and credentials for the shared ACP driver."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rpent.planner.utils.acp import AcpPlanner, AcpSession
from rpent.planner.utils.agent_runtime import agent_environment

_MCP_TOOL_PREFIX = "mcp_rpent_"
_SECRET_ENV_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
# Gemini CLI treats an empty tools.core list as unset.
_NO_BUILTIN_TOOLS = ("__rpent_no_builtin_tools__",)


# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class GeminiCliPlanner(AcpPlanner):
    """Configure Gemini CLI while sharing ACP execution and interaction."""

    backend = "gemini_cli"
    label = "Gemini CLI"

    def _create_session(
        self, root, control, emit, stderr_path, *, mcp_url, max_turns, tool_names
    ) -> AcpSession:
        _write_settings(
            root, mcp_url=mcp_url, max_turns=max_turns, tool_names=tool_names
        )
        binary = os.environ.get("GEMINI_BIN") or shutil.which("gemini")
        if not binary:
            raise FileNotFoundError(
                "Gemini CLI is not installed; install @google/gemini-cli or set GEMINI_BIN"
            )
        command = [
            binary,
            "--acp",
            "--allowed-mcp-server-names",
            "rpent",
            "--extensions",
            "none",
        ]
        if self._model:
            command.extend(["--model", self._model])
        return AcpSession(
            command=command,
            cwd=root,
            model=None,
            env={
                **agent_environment(),
                "GEMINI_CLI_NO_RELAUNCH": "1",
                # Trust the generated workspace so Gemini loads its local settings.
                "GEMINI_CLI_TRUST_WORKSPACE": "true",
                "NO_COLOR": "1",
            },
            control=control,
            emit_event=emit,
            stderr_path=stderr_path,
            redact=self._redact,
        )

    @staticmethod
    def _redact(text: str) -> str:
        for name in _SECRET_ENV_NAMES:
            secret = os.environ.get(name) or _dotenv_value(name)
            if secret and len(secret) >= 8:
                text = text.replace(secret, f"<{name}>")
        return text


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


def _write_settings(
    root: Path,
    *,
    mcp_url: str | None,
    max_turns: int,
    tool_names: Iterable[str] = (),
) -> Path:
    """Write workspace-only settings without changing the user's Gemini config."""
    allowed_tools = [f"{_MCP_TOOL_PREFIX}{name}" for name in tool_names]
    settings: dict[str, Any] = {
        "context": {
            "fileName": "RPENT_NO_PROJECT_CONTEXT.md",
            "includeDirectoryTree": False,
            "memoryBoundaryMarkers": [],
        },
        "hooksConfig": {"enabled": False},
        "model": {"maxSessionTurns": max(1, int(max_turns))},
        "privacy": {"usageStatisticsEnabled": False},
        "skills": {"enabled": False},
        "tools": {"core": allowed_tools or list(_NO_BUILTIN_TOOLS)},
    }
    if mcp_url:
        settings["mcpServers"] = {
            "rpent": {
                "httpUrl": mcp_url,
                "trust": True,
            }
        }
    settings_dir = root / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return path


def _dotenv_value(name: str) -> str | None:
    """Read one Gemini CLI dotenv setting without logging neighboring secrets."""
    path = Path.home() / ".gemini" / ".env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip().removeprefix("export ").strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None
