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
from typing import TYPE_CHECKING, Any

from robots.franka import perception as franka_perception
from robots.franka import tools as franka_tools
from robots.franka.runtime_config import set_calibration_path
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class FrankaToolkit(Toolkit):
    """Common RPent tools plus safe single-Franka planner primitives."""

    _tools = franka_tools
    _primitives_cls = franka_tools.FrankaPrimitives

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
    ) -> None:
        state = EnvState(get_output_dir())
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        calibration_path = primitives_kwargs.pop("calibration_path", None)
        if calibration_path is not None:
            set_calibration_path(calibration_path)
        self._primitives = self._primitives_cls(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        self._register_tools()
        self._state.reset()
        self._primitives.reset()
        record = self._tools.dump_state(
            self._primitives,
            self._state,
            command=None,
            result=None,
            elapsed_s=None,
        )
        self._publish_step(record)

    def _register_tools(self) -> None:
        state_handlers = {
            "view_env_state": partial(franka_tools.view_env_state, state=self._state),
            "view_camera_meta": partial(
                franka_tools.view_camera_meta,
                state=self._state,
            ),
            "view_perception_setup": partial(
                franka_perception.view_perception_setup,
                state=self._state,
            ),
            "back_project": partial(
                franka_perception.back_project,
                state=self._state,
            ),
            "back_project_correspondence": partial(
                franka_perception.back_project_correspondence,
                state=self._state,
            ),
        }
        for spec in self._tools.TOOLS_SPEC:
            name = spec["name"]
            handler = state_handlers.get(name) or getattr(self._primitives, name)
            self.add_tool(name, spec, handler)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        record = self._tools.dump_state(
            self._primitives,
            self._state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )
        output = self._tools.view_env_state(record.step_idx, state=self._state)
        output["agent_elapsed_s"] = elapsed_s
        return output
