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

"""LeKiwi planner tools, primitives, and canonical state capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import Field

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool


class LeKiwiPrimitives:
    """Safe agent-facing operations over a remote LeKiwi environment."""

    def __init__(
        self,
        *,
        env: Any,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self._check_cancelled = check_cancelled

    def reset(self) -> dict[str, Any]:
        return self.env.reset()

    @tool
    def drive(
        self,
        velocity: Annotated[list[float], Field(min_length=3, max_length=3)],
        duration_s: Annotated[float, Field(gt=0.0, le=10.0, allow_inf_nan=False)] = 1.0,
    ) -> ToolResult:
        """Drive at body velocity [forward, left, turn] for a duration, then stop.

        Units are m/s, m/s and deg/s. Velocity is clipped to the robot's
        configured limits; duration is at most 10 seconds. Waits for completion
        or cancellation. This is a timed command, not a distance guarantee.
        No obstacle avoidance; inspect status and the camera view before continuing.
        """
        self._check_cancelled()
        data = self.env.drive(velocity, duration_s=duration_s)
        self._check_cancelled()
        return ToolResult(data=data)

    @tool
    def rotate(
        self,
        angle_deg: Annotated[float, Field(ge=-360.0, le=360.0, allow_inf_nan=False)],
    ) -> ToolResult:
        """Turn in place by a relative angle in degrees, then stop.

        Positive angles turn left, negative angles turn right; at most one
        full turn per call. Waits for wheel-estimated angle and stopping,
        or failure/cancellation. No obstacle avoidance. Inspect the returned
        status and camera view before the next motion.
        """
        self._check_cancelled()
        data = self.env.rotate(angle_deg)
        self._check_cancelled()
        return ToolResult(data=data)

    @tool
    def stop(self) -> ToolResult:
        """Command zero wheel velocity immediately."""
        self._check_cancelled()
        data = self.env.stop()
        return ToolResult(data=data)


def dump_state(
    primitives: LeKiwiPrimitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture robot state and the post-action camera frame in ``EnvState``."""
    observation = primitives.env.get_observation()
    robot_state = primitives.env.get_robot_state()
    metadata = primitives.env.get_camera_meta()
    with state.record_step(
        state=robot_state,
        command=command,
        result=result,
        elapsed_s=elapsed_s,
    ) as step:
        main_image = observation.get("main_images")
        if main_image is not None:
            state.save("front.png", main_image, step=step)
        if metadata is not None:
            state.save("camera_meta.json", metadata, step=step)
    return state.get(step)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read a LeKiwi state snapshot and its recorded front-camera image."""
    images: list[bytes] = []
    record = state.get(step)
    output = record.to_blob()
    if state.exists("front.png", step=record.step_idx):
        output["image_front_path"] = str(
            state.artifact_path("front.png", step=record.step_idx)
        )
        images.append(state.load_bytes("front.png", step=record.step_idx))
    return ToolResult(data=output, images=images)
