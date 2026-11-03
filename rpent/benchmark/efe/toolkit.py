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

"""Session control for EFE coding agents."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from rpent.dashboard.events import NullDashboardEventSink
from rpent.tools import Toolkit, ToolResult, tool

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

# The shared tools describe robot artifacts. EFE sessions produce code and
# notes, so their descriptions must not point the model at recipes or audits.
_EFE_TOOL_DESCRIPTIONS = {
    "read_text_file": (
        "Read a UTF-8 text file. Use for task files, workspace code, and RPent "
        "memory files. Large files are truncated. Published memory is read-only; "
        "during exploration, write only to the current memory inbox.\n\n"
        "Args:\n"
        "    path: Absolute or workspace-relative path\n"
        "    max_chars: Max chars (default 40000)"
    ),
    "write_text_file": (
        "Write a UTF-8 text file (creates parent dirs). Use for working notes "
        "and other run artifacts. Published memory is read-only; during "
        "exploration, write only to the current memory inbox."
    ),
    "list_dir": (
        "List files in a directory (non-recursive). Default = the run output "
        "directory. Use to inspect the workspace or memory corpus.\n\n"
        "Args:\n"
        "    path: Default: the run output directory"
    ),
}


class EfeToolkit(Toolkit):
    """Expose session completion while the coding backend owns file and shell tools."""

    def __init__(self, *, memory: MemoryManager) -> None:
        """Create a toolkit that keeps all non-robot tools on the Codex backend.

        The shared file and memory tools are registered so the memory
        authorization and notes protocol apply. Native Codex tools still own
        shell commands.
        """
        super().__init__(dashboard_events=NullDashboardEventSink(), memory=memory)

    def _register_common_tools(self) -> None:
        super()._register_common_tools()
        for declaration in list(self._tools.values()):
            description = _EFE_TOOL_DESCRIPTIONS.get(declaration.name)
            if description is not None:
                self.add_tool(
                    replace(declaration, description=description), replace=True
                )
        # The EFE finish contract replaces the shared one.
        self.add_tool(self.finish, replace=True)

    @tool(readonly=True)
    def finish(self, status: Literal["completed", "stuck"], summary: str) -> ToolResult:
        """End the coding session. EFE's verifier determines task success independently.

        Args:
            status: Whether the agent completed its work or cannot continue.
            summary: Describe the work and any remaining limitation.
        """
        return ToolResult(data={"_finish": True, "status": status, "summary": summary})
