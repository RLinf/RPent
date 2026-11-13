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

"""EFE toolkit: raw service tools, optional skills, and state capture."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from robots.efe import tools as efe_tools
from robots.efe.env_client import EfeEnvClient
from robots.efe.skills import EfeSkills
from robots.efe.tools import EfeTools, view_env_state
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools, tool
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

logger = get_logger("efe")

# The shared tools describe robot artifacts. EFE sessions work on one live
# episode, so their descriptions must not point the model at recipes or audits.
_EFE_TOOL_DESCRIPTIONS = {
    "read_text_file": (
        "Read a UTF-8 text file. Use for the task instruction, RPent memory "
        "files, and notes from earlier sessions. Large files are truncated. "
        "Published memory is read-only; during exploration, write only to the "
        "current memory inbox.\n\n"
        "Args:\n"
        "    path: Absolute or workspace-relative path\n"
        "    max_chars: Max chars (default 40000)"
    ),
    "write_text_file": (
        "Write a UTF-8 text file (creates parent dirs). Use for working notes "
        "only. Published memory is read-only; during exploration, write only "
        "to the current memory inbox. Files under /app/output are collected "
        "with the trial."
    ),
    "list_dir": (
        "List files in a directory (non-recursive). Default = the run output "
        "directory. Use to inspect the workspace or memory corpus.\n\n"
        "Args:\n"
        "    path: Default: the run output directory"
    ),
}


def tool_capabilities(
    client: EfeEnvClient, *, enable_skills: bool = True
) -> dict[str, bool]:
    """Return the prompt flags that match the tools this task registers.

    The prompt and the toolkit must derive the tool set the same way: a task
    without the joint layout the skills need falls back to ``efe_submit``, and
    the service spec decides reset and hidden success.
    """
    return {
        "skills": enable_skills and EfeSkills.unavailable_reason(client.spec) is None,
        "reset": client.reset_budget > 0,
        "end_episode": client.success_hidden,
    }


class EfeToolkit(Toolkit):
    """Expose raw EFE tools and high-level skills to the planner."""

    def __init__(
        self,
        *,
        client: EfeEnvClient,
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str,
        enable_skills: bool = True,
    ) -> None:
        self._client = client
        self._solved = False
        self._call_counts: Counter[str] = Counter()
        self._failed_counts: Counter[str] = Counter()
        self._start_submits = client.submit_calls
        self._start_steps = client.control_steps
        self._start_observes = client.observe_calls
        state = EnvState(state_output_dir)
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._service = EfeTools(client=client, state=state)
        self._skills_error: str | None = None
        self._skills: EfeSkills | None = None
        if enable_skills:
            reason = EfeSkills.unavailable_reason(client.spec)
            if reason is None:
                self._skills = EfeSkills(
                    client=client,
                    check_cancelled=self.raise_if_cancelled,
                )
            else:
                # Keep the raw tools and record why no action skill is registered.
                self._skills_error = reason
                logger.warning("EFE skills unavailable: %s", reason)
        for declaration in iter_tools(self._service):
            if declaration.name == "efe_submit" and self._skills is not None:
                continue
            if declaration.name == "efe_reset" and client.reset_budget <= 0:
                continue
            if declaration.name == "efe_end_episode" and not client.success_hidden:
                continue
            self.add_tool(declaration)
        self.add_tool(view_env_state.with_handler(partial(view_env_state, state=state)))
        if self._skills is not None:
            self.add_tools(iter_tools(self._skills))
        self.add_tool(self.finish, replace=True)

    def _register_common_tools(self) -> None:
        super()._register_common_tools()
        for declaration in list(self._tools.values()):
            description = _EFE_TOOL_DESCRIPTIONS.get(declaration.name)
            if description is not None:
                self.add_tool(
                    replace(declaration, description=description), replace=True
                )

    @property
    def skills_enabled(self) -> bool:
        """Whether the high-level skills are registered for this task."""
        return self._skills is not None

    @tool(name="finish", readonly=True)
    def finish(
        self,
        status: Literal["completed", "stuck"],
        summary: str,
    ) -> ToolResult:
        """End the session. The EFE verifier scores the episode independently.

        Call it after the last check, not to claim success. Save artifacts and
        notes before calling it; nothing runs after it.

        Args:
            status: Whether the agent finished its work or cannot continue.
            summary: What was achieved and what remains uncertain.
        """
        return ToolResult(data={"_finish": True, "status": status, "summary": summary})

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Count calls and failures for the comparison record."""
        result = super().execute_tool(name, input_dict)
        self._call_counts[name] += 1
        if result.is_error:
            self._failed_counts[name] += 1
        return result

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Record public proprioception after a state-changing tool."""
        record = efe_tools.dump_state(
            self._client,
            self.state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            frames=self._service.latest_frames(),
        )
        out = ToolResult(data=record.to_blob())
        out.data["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.data.update(result)
        return out

    def solved(self) -> bool:
        """Whether the service itself reported success.

        A ``finish`` call is not a score. Only the service's own end reason is
        trusted, and the Verifier still grades the episode.
        """
        self._solved |= self._client.done and self._client.done_reason == "success"
        return self._solved

    def metrics(self) -> dict[str, Any]:
        """Return this session's counters for the comparison record.

        The service client lives for the whole trial, so counters are offsets
        from the values at session start.
        """
        skill_calls = sum(
            count
            for name, count in self._call_counts.items()
            if name in EfeSkills.skill_names
        )
        return {
            "tool_calls": sum(self._call_counts.values()),
            "failed_calls": sum(self._failed_counts.values()),
            "skill_calls": skill_calls,
            "efe_submit_calls": self._client.submit_calls - self._start_submits,
            "efe_control_steps": self._client.control_steps - self._start_steps,
            "efe_observe_calls": self._client.observe_calls - self._start_observes,
            "skills_enabled": self._skills is not None,
            "skills_error": self._skills_error,
        }
