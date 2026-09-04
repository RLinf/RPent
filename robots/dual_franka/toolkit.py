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

"""Dual-Franka toolkit integrated with RPent's centralized environment state."""

from __future__ import annotations

from functools import partial

from robots.dual_franka import perception as dual_franka_perception
from robots.dual_franka import tools as dual_franka_tools
from robots.franka import tools as franka_tools
from robots.franka.toolkit import FrankaToolkit


class DualFrankaToolkit(FrankaToolkit):
    """Common RPent tools plus safe dual-Franka planner primitives."""

    _tools = dual_franka_tools
    _primitives_cls = dual_franka_tools.DualFrankaPrimitives

    def _register_tools(self) -> None:
        state_handlers = {
            "view_env_state": partial(
                dual_franka_tools.view_env_state, state=self._state
            ),
            "view_camera_meta": partial(
                franka_tools.view_camera_meta,
                state=self._state,
            ),
            "back_project_base_pixel": partial(
                dual_franka_perception.back_project_base_pixel,
                state=self._state,
            ),
            "back_project_d455_pixel": partial(
                dual_franka_perception.back_project_d455_pixel,
                state=self._state,
            ),
        }
        for spec in self._tools.TOOLS_SPEC:
            name = spec["name"]
            handler = state_handlers.get(name) or getattr(self._primitives, name)
            self.add_tool(name, spec, handler)
