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

"""R2 toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.r2 import tools as r2_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class R2Toolkit(Toolkit):
    """Common RPent tools plus bounded R2 motion, perception, and state readers."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
    ) -> None:
        runtime_kwargs = dict(runtime_kwargs)
        sam3_client = runtime_kwargs.pop("sam3_client", None)
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._primitives = r2_tools.R2Primitives(
            check_cancelled=self.raise_if_cancelled, **runtime_kwargs
        )
        meta = self._primitives.env.meta
        self.add_tools(
            definition
            for definition in iter_tools(self._primitives)
            if definition.name != "move_base" or meta["base_motion"]
        )
        for definition in iter_tools(r2_tools):
            resources: dict[str, Any] = {"state": self._state}
            if definition.name == "segment":
                if sam3_client is None:
                    continue
                resources["sam3_client"] = sam3_client
            self.add_tool(definition.with_handler(partial(definition, **resources)))
        self._state.reset()
        record = r2_tools.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
            extra={"robot_info": meta},
        )
        self._publish_step(record)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        record = r2_tools.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = r2_tools.view_env_state(record.step_idx, state=self._state)
        output.data["agent_elapsed_s"] = elapsed_s
        return output

    def solved(self) -> bool:
        """Return ``False``: R2 has no automatic success check, so the operator judges."""
        return False
