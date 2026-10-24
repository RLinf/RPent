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

"""RoboDojo tool registration and observation capture."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools, tool
from rpent.utils.logging import get_output_dir

OBSERVATION_TOOLS = frozenset(
    {"view_env_state", "segment", "back_project", "sample_world_xyz", "query_world_map"}
)
MOTION_TOOLS = frozenset(
    {"move_to", "rotate_wrist", "set_gripper", "release", "vpp2_act"}
)

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment.

    Legacy policy code and composites are not exposed. ``eval_fair`` requires a mode-validated
    environment, removes file tools, and disables development perception
    recording. Scoring and safety diagnostics are never planner tools.
    """

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        eval_fair: bool = False,
        state_output_dir: Path | str | None = None,
    ) -> None:
        self.eval_fair = eval_fair
        if eval_fair and not getattr(primitives_kwargs.get("env"), "eval_fair", False):
            raise ValueError("eval-fair requires a mode-validated environment client")
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        self._register_robodojo_tools()
        if eval_fair:
            allowed = OBSERVATION_TOOLS | MOTION_TOOLS | {"finish", "render"}
            self._tools = {
                name: entry for name, entry in self._tools.items() if name in allowed
            }

    def execute_tool(self, name, input_dict):
        result = super().execute_tool(name, input_dict)
        if not self.eval_fair and name in {
            "segment",
            "back_project",
        }:
            import copy

            if isinstance(result.to_dict(), dict):
                # Read-only queries do not create EnvState steps. Attach them
                # to the next action for Flash grounding; actions themselves
                # are already recorded by the shared Toolkit/EnvState path.
                self._pending_perception.append(
                    copy.deepcopy(
                        {
                            "action": name,
                            "arguments": input_dict,
                            "result": result.to_dict(),
                        }
                    )
                )
        return result

    def flash_observation(self) -> dict:
        """Refresh the public observation used by live Flash grounding."""
        from robots.robodojo.access import public_observation

        if not self.eval_fair:
            raise RuntimeError("Flash requires eval-fair toolkit")
        obs = public_observation(self._primitives.env.get_obs())
        self._primitives._last_obs = obs
        return obs

    def init_primitives(self, *, primitives_kwargs: dict[str, Any]) -> None:
        """Start a new trace from the runtime-owned environment."""
        from robots.robodojo import tools as robodojo_tools

        self._pending_perception = []
        self._state.reset()
        self._primitives = RoboDojoPrimitives(
            env=primitives_kwargs["env"],
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=primitives_kwargs.get("vla_client"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )
        self._publish_step(
            robodojo_tools.dump_state(self._primitives, self._state, log=None)
        )

    def solved(self) -> bool:
        """Read official success for the runner after planner execution."""
        if self.eval_fair:
            return False  # Plan completion is not an official task verdict.
        return self._primitives.env.is_success()

    def _register_robodojo_tools(self) -> None:
        from robots.robodojo import tools as robodojo_tools

        for definition in iter_tools(robodojo_tools):
            if definition.name not in OBSERVATION_TOOLS:
                continue
            resources = {"state": self._state}
            if definition.name in {"sample_world_xyz", "query_world_map"}:
                resources = {"env_state": self._state}
            elif definition.name == "segment":
                resources["primitives"] = self._primitives
            self.add_tool(definition.with_handler(partial(definition, **resources)))
        for definition in iter_tools(self._primitives):
            if definition.name == "vpp2_act" and self._primitives.vla_client is None:
                continue
            self.add_tool(definition, replace=definition.name == "finish")
        self.add_tool(self.render)

    @tool
    def render(self) -> ToolResult:
        """Refresh the synchronized RGB-D, calibration and robot state record."""
        self.raise_if_cancelled()
        return ToolResult(data={"success": True})

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        from robots.robodojo import tools as robodojo_tools

        record = robodojo_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
            perception=self._pending_perception,
        )
        self._pending_perception = []
        return robodojo_tools.view_env_state(record.step_idx, state=self._state)
