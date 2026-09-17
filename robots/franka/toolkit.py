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

"""Native Franka toolkit and per-session robot resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.franka import tools as franka_tools
from robots.franka.runtime_config import set_calibration_path
from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.session import EnvState, StepRecord
from rpent.tools import Toolkit, ToolResult
from rpent.utils.logging import get_output_dir

if TYPE_CHECKING:
    from robots.franka.env_client import FrankaEnvClient
    from rpent.memory.manager import MemoryManager
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient


@dataclass
class FrankaRuntime:
    """Environment and optional VLA client shared by one Franka session."""

    env: FrankaEnvClient
    model: Pi05VLAClient | None
    task_description: str


class FrankaToolkit(Toolkit[FrankaRuntime]):
    """Common native tools plus single-Franka motion and perception."""

    _runtime_type = FrankaRuntime
    _robot_tools = franka_tools.FRANKA_TOOLS

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str | None = None,
    ) -> None:
        runtime_kwargs = dict(runtime_kwargs)
        calibration_path = runtime_kwargs.pop("calibration_path", None)
        if calibration_path is not None:
            set_calibration_path(calibration_path)
        output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(output_dir)
        super().__init__(
            state=state,
            memory=memory,
            robot=self._runtime_type(**runtime_kwargs),
            output_dir=output_dir,
            tools=self._robot_tools,
            dashboard_events=dashboard_events,
        )
        # The env client already resets the robot when it connects.
        state.reset()
        record = self._dump_state(
            command=None,
            result=None,
            elapsed_s=None,
        )
        self._dashboard_events.emit(StepRecordEvent(record=record, env_state=state))

    def _capture_observation(
        self,
        *,
        command: dict[str, Any],
        result: ToolResult,
        elapsed_s: float,
    ) -> tuple[dict[str, Any], list[bytes]]:
        record = self._dump_state(
            command=command,
            result=result.to_dict(),
            elapsed_s=elapsed_s,
        )
        observation = self._build_observation(record)
        observation.data["agent_elapsed_s"] = elapsed_s
        return observation.data, observation.images

    def _dump_state(
        self,
        *,
        command: dict[str, Any] | None,
        result: dict[str, Any] | None,
        elapsed_s: float | None,
    ) -> StepRecord:
        return franka_tools.dump_state(
            self._robot,
            self.state,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        )

    def _build_observation(self, record: StepRecord) -> ToolResult:
        return franka_tools.build_observation(self.state, record)
