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

"""Visual observations and bounded Cartesian primitives for MetaWorld."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic import Field

from robots.metaworld.geometry import MetaWorldGeometryObservations
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools, tool
from toolkits.back_project import back_project

if TYPE_CHECKING:
    from rpent.memory import MemoryManager


class MetaWorldToolkit(Toolkit):
    """Drive a single fixed episode without exposing native expert state."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        state_output_dir: Path | str,
    ) -> None:
        self._wam = None
        self.env = runtime_kwargs["env_client"]
        self.obs = self.env.last_obs
        super().__init__(
            dashboard_events=dashboard_events,
            state=EnvState(state_output_dir),
            memory=memory,
        )
        self.add_tools(t for t in iter_tools(self) if t.name != "wam_act")
        self.add_tool(
            back_project.with_handler(
                partial(
                    back_project,
                    observations=MetaWorldGeometryObservations(self._state),
                )
            )
        )
        self.get_env_state(command={}, result={}, elapsed_s=0)
        self._publish_step(self._state.latest_record())
        if endpoint := runtime_kwargs.get("am_endpoint"):
            from robots.metaworld.wam_client import MetaWorldWAMClient

            self._wam = MetaWorldWAMClient.connect(endpoint)
            self.add_tool(self.wam_act)

    @tool(readonly=True)
    def view_env_state(self) -> ToolResult:
        """Read the latest camera snapshot without advancing simulation."""
        return self._view()

    def _view(self) -> ToolResult:
        record = self._state.latest_record()
        return ToolResult(
            data={
                **record.state,
                "step": record.step_idx,
                "episode_ended": bool(record.terminated or record.truncated),
                "terminated": record.terminated,
                "truncated": record.truncated,
            },
            images=[self._state.load_bytes("camera.png", step=record.step_idx)],
        )

    @staticmethod
    def _gripper(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("gripper must be finite and in [-1, 1]")
        return value

    def _step(self, action: list[float]) -> None:
        self.raise_if_cancelled()
        if self.obs["terminated"] or self.obs["truncated"]:
            raise RuntimeError("episode ended; call finish")
        self.obs = self.env.step(action)[0]

    @tool
    def move_to(
        self,
        target_xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
        gripper: Annotated[float, Field(ge=-1, le=1)],
        max_steps: Annotated[int, Field(ge=1, le=100)] = 50,
    ) -> ToolResult:
        """Move toward world XYZ with fixed wrist orientation; gripper +1 closes, -1 opens. Check reached."""
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_xyz must be three finite world coordinates")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or not 1 <= max_steps <= 100
        ):
            raise ValueError("max_steps must be an integer in [1, 100]")
        gripper = self._gripper(gripper)
        count = 0
        while count < max_steps:
            delta = target - np.asarray(self.obs["eef_position"])
            if (
                np.linalg.norm(delta) < 0.005
                or self.obs["terminated"]
                or self.obs["truncated"]
            ):
                break
            # Native MetaWorld scales XYZ actions by 0.01 meters.
            self._step([*np.clip(delta / 0.01, -1, 1).tolist(), gripper])
            count += 1
        distance = float(np.linalg.norm(target - np.asarray(self.obs["eef_position"])))
        return ToolResult(
            data={"reached": distance < 0.005, "distance_m": distance, "steps": count}
        )

    @tool
    def set_gripper(
        self,
        gripper: Annotated[float, Field(ge=-1, le=1)],
        steps: Annotated[int, Field(ge=1, le=20)] = 10,
    ) -> ToolResult:
        """Apply a gripper command without requesting Cartesian displacement."""
        gripper = self._gripper(gripper)
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 1 <= steps <= 20
        ):
            raise ValueError("steps must be an integer in [1, 20]")
        count = 0
        while count < steps and not (self.obs["terminated"] or self.obs["truncated"]):
            self._step([0, 0, 0, gripper])
            count += 1
        return ToolResult(
            data={"steps": count, "gripper_opening": self.obs["gripper_opening"]}
        )

    def get_env_state(
        self, *, command: dict, result: dict, elapsed_s: float
    ) -> ToolResult:
        """Record the observation and attach the motion result to the tool response."""
        self.obs = self.env.get_obs()
        public = {
            k: self.obs[k]
            for k in [
                "eef_position",
                "gripper_opening",
                "instruction",
                "steps",
                "camera",
            ]
        }
        with self._state.record_step(
            state=public,
            terminated=self.obs["terminated"],
            truncated=self.obs["truncated"],
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            self._state.save("camera.png", self.obs["rgb"])
            self._state.save("depth.npy", self.obs["depth"])
        observed = self._view()
        observed.data["motion_result"] = result
        return observed

    @tool
    def wam_act(self, max_chunks: Annotated[int, Field(ge=1, le=20)] = 1) -> ToolResult:
        """Execute bounded WAM chunks for the task, then inspect the observed result."""
        if self._wam is None:
            raise ValueError("wam_act requires --am-endpoint")
        if not 1 <= max_chunks <= 20:
            raise ValueError("max_chunks must be in [1, 20]")
        steps = 0
        for _ in range(max_chunks):
            self.raise_if_cancelled()
            self.obs = self.env.get_obs()
            if self.obs["terminated"] or self.obs["truncated"]:
                break
            prediction = self._wam.predict(self.obs)
            for action in prediction["actions"]:
                self._step(action.tolist())
                steps += 1
                if self.obs["terminated"] or self.obs["truncated"]:
                    break
        return ToolResult(data={"steps": steps})

    def close(self) -> None:
        """Drain tool calls before releasing this client's borrowed model session."""
        super().close()
        if self._wam is not None:
            self._wam.close()
            self._wam = None

    def solved(self) -> bool:
        """Read the native success accumulator only for run finalization."""
        return self.env.is_success()
