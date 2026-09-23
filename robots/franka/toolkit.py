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

"""Franka toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.franka import perception as franka_perception
from robots.franka import tools as franka_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class FrankaToolkit(Toolkit):
    """Common RPent tools plus safe single-Franka planner primitives."""

    _tools_module = franka_tools
    _primitives_cls = franka_tools.FrankaPrimitives

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
    ) -> None:
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._primitives = self._primitives_cls(
            check_cancelled=self.raise_if_cancelled,
            **runtime_kwargs,
        )
        self._register_tools()
        self._state.reset()
        record = self._tools_module.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
        )
        self._publish_step(record)

    def _register_tools(self) -> None:
        self.add_tools(iter_tools(self._primitives))
        for definition in iter_tools(self._tools_module, franka_perception):
            self.add_tool(
                definition.with_handler(partial(definition, state=self._state))
            )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        record = self._tools_module.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = self._tools_module.view_env_state(record.step_idx, state=self._state)
        output.data["agent_elapsed_s"] = elapsed_s
        return output
