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

"""Native dual-Franka toolkit with shared Franka session lifecycle."""

from typing import Any

from robots.dual_franka import tools
from robots.franka.toolkit import FrankaToolkit
from rpent.session import StepRecord
from rpent.tools import ToolResult


class DualFrankaToolkit(FrankaToolkit):
    """Common native tools plus dual-Franka motion and perception."""

    _robot_tools = tools.DUAL_FRANKA_TOOLS

    def _dump_state(
        self,
        *,
        command: dict[str, Any] | None,
        result: dict[str, Any] | None,
        elapsed_s: float | None,
    ) -> StepRecord:
        return tools.dump_state(
            self._robot,
            self.state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )

    def _build_observation(self, record: StepRecord) -> ToolResult:
        return tools.build_observation(self.state, record)
