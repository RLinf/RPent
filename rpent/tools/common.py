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

"""Shared file tools with run-owned memory authorization."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rpent.tools.base import ToolResult, tool
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory import MemoryManager


class CommonTools:
    """Bind common tools to the current run's memory access policy."""

    def __init__(self, *, memory: MemoryManager) -> None:
        self._memory = memory

    @tool(readonly=True)
    def read_text_file(self, path: str, max_chars: int = 40000) -> ToolResult:
        """Read a UTF-8 text file. Use for past recipe JSONLs, audit JSONs, and memory files. Large files are truncated. Published memory is read-only. During exploration, you may write only to your current memory inbox. Memory for other robots is unavailable.

        Args:
            path: Absolute or repo-relative path
            max_chars: Max chars (default 40000)
        """
        resolved = self._memory.authorize_read(path)
        if not resolved.exists():
            return ToolResult(error=f"file not found: {resolved}")
        if resolved.is_dir():
            return ToolResult(error=f"is a directory: {resolved}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        content = text
        if len(text) > max_chars:
            content = text[:max_chars] + (
                f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
            )
        return ToolResult(
            data={"path": str(resolved), "size": len(text), "content": content}
        )

    @tool(readonly=True)
    def write_text_file(self, path: str, content: str) -> ToolResult:
        """Write a UTF-8 text file (creates parent dirs). Use this to save the working recipe JSONL and the final audit JSON at the end of a successful run. Published memory is read-only. During exploration, you may write only to your current memory inbox. Memory for other robots is unavailable."""
        resolved = self._memory.authorize_write(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(
            data={"path": str(resolved), "bytes_written": len(content.encode("utf-8"))}
        )

    @tool(readonly=True)
    def list_dir(self, path: str = "") -> ToolResult:
        """List files in a directory (non-recursive). Default = {{output_dir}}. Use to inspect the working directory or discover existing resource files. Published memory is read-only. During exploration, you may write only to your current memory inbox. Memory for other robots is unavailable.

        Args:
            path: Default: {{output_dir}}
        """
        resolved = self._memory.authorize_read(path or get_output_dir())
        if not resolved.exists():
            return ToolResult(error=f"directory not found: {resolved}")
        files = sorted(os.listdir(resolved))
        return ToolResult(
            data={"path": str(resolved), "count": len(files), "files": files}
        )

    @tool(readonly=True)
    def finish(self, status: str, summary: str) -> ToolResult:
        """Call when the task is complete or unrecoverable. Halts the agent loop. Save any artifacts (recipe, audit) BEFORE calling finish.

        Args:
            status: Outcome, e.g. 'success', 'failure', or 'stuck'.
            summary: Short natural-language summary of the run.
        """
        return ToolResult(data={"_finish": True, "status": status, "summary": summary})
