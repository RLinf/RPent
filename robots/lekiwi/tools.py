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

from collections.abc import Callable, Sequence
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool


def coerce_vec3(value: Sequence[float], *, name: str) -> np.ndarray:
    """Return a finite float32 three-vector or raise a useful error."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class LeKiwiPrimitives:
    """Safe agent-facing operations over a remote LeKiwi environment."""

    def __init__(
        self,
        *,
        env: Any,
        task_description: str,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self.task_description = task_description
        self._check_cancelled = check_cancelled

    def reset(self) -> dict[str, Any]:
        return self.env.reset()

    @tool
    def drive(
        self,
        velocity: Annotated[list[float], Field(min_length=3, max_length=3)],
        duration_s: Annotated[float, Field(gt=0.0, le=10.0)] = 1.0,
    ) -> ToolResult:
        """Drive the base with one body velocity [x, y, theta] held for a bounded duration.

        Units are m/s, m/s, and deg/s in the base frame. The command is
        clipped to the configured velocity limits, and the wheels stop
        when the duration ends.
        """
        self._check_cancelled()
        data = self.env.drive(
            coerce_vec3(velocity, name="velocity"), duration_s=duration_s
        )
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
    """Capture robot state and the synchronized front-camera frame in ``EnvState``."""
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
    """Read a LeKiwi state snapshot and its synchronized front-camera image."""
    images: list[bytes] = []
    record = state.get(step)
    output = record.to_blob()
    if state.exists("front.png", step=record.step_idx):
        output["image_front_path"] = str(
            state.artifact_path("front.png", step=record.step_idx)
        )
        images.append(state.load_bytes("front.png", step=record.step_idx))
    return ToolResult(data=output, images=images)
