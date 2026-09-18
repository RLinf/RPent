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

"""Franka planner tools, primitives, and canonical state capture."""

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


class FrankaPrimitives:
    """Safe agent-facing operations over a remote Franka environment."""

    def __init__(
        self,
        *,
        env: Any,
        model: Any | None,
        task_description: str,
        check_cancelled: Callable[[], None],
    ) -> None:
        self.env = env
        self.model = model
        self.task_description = task_description
        self._check_cancelled = check_cancelled

    def reset(self) -> dict[str, Any]:
        return self.env.reset()

    @tool
    def move_delta(
        self, delta_xyz: Annotated[list[float], Field(min_length=3, max_length=3)]
    ) -> ToolResult:
        """Move the Franka TCP by a bounded base-frame xyz delta in meters."""
        self._check_cancelled()
        data = self.env.move_delta(coerce_vec3(delta_xyz, name="delta_xyz"))
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def rotate_delta(
        self, delta_rpy: Annotated[list[float], Field(min_length=3, max_length=3)]
    ) -> ToolResult:
        """Rotate the Franka TCP by a bounded base-frame rpy delta in radians."""
        self._check_cancelled()
        data = self.env.rotate_delta(coerce_vec3(delta_rpy, name="delta_rpy"))
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def open_gripper(self) -> ToolResult:
        """Open the Franka gripper and wait for the command to settle."""
        self._check_cancelled()
        data = self.env.set_gripper(open=True)
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def close_gripper(self) -> ToolResult:
        """Close the Franka gripper and wait for the command to settle."""
        self._check_cancelled()
        data = self.env.set_gripper(open=False)
        return ToolResult(data=data, error=data.pop("error", None))

    @tool
    def vla_grasp(
        self, prompt: str, max_chunks: Annotated[int, Field(ge=1, le=20)] = 4
    ) -> ToolResult:
        """Run bounded real-world VLA action chunks for a local grasp attempt."""
        if self.model is None:
            raise RuntimeError("vla_grasp requires --vla-endpoint")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not 1 <= int(max_chunks) <= 20:
            raise ValueError("max_chunks must be between 1 and 20")

        chunk_results: list[dict[str, Any]] = []
        observation: dict[str, Any] | None = None
        for _ in range(int(max_chunks)):
            self._check_cancelled()
            if observation is None:
                observation = dict(self.env.get_observation())
            observation["task_descriptions"] = prompt or self.task_description
            actions = self.model.predict(observation, options={"mode": "eval"})
            result = self.env.chunk_step(actions)
            chunk_results.append(result)
            if result.get("terminated") or result.get("truncated"):
                break
            # Reuse the obs chunk_step already returned instead of re-fetching it.
            next_obs = result.get("observation")
            observation = dict(next_obs) if isinstance(next_obs, dict) else None

        return ToolResult(
            data={
                "ok": True,
                "chunks_executed": len(chunk_results),
                "last_chunk": chunk_results[-1] if chunk_results else None,
                "robot_state": self.env.get_robot_state(),
            }
        )


def dump_state(
    primitives: FrankaPrimitives,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture robot state and synchronized camera artifacts in ``EnvState``."""
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
            state.save("wrist.png", main_image, step=step)
        extra_image = observation.get("extra_view_images")
        if extra_image is not None:
            extra_array = np.asarray(extra_image)
            if extra_array.ndim == 4:
                extra_array = extra_array[0]
            state.save("camera.png", extra_array, step=step)
        main_depth = observation.get("main_depths")
        if main_depth is not None:
            state.save("wrist_depth.npy", main_depth, step=step)
        extra_depth = observation.get("extra_view_depths")
        if extra_depth is not None:
            depth_array = np.asarray(extra_depth)
            if depth_array.ndim == 3:
                depth_array = depth_array[0]
            state.save("camera_depth.npy", depth_array, step=step)
        if metadata is not None:
            state.save("camera_meta.json", metadata, step=step)
    return state.get(step)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read a Franka state snapshot and its synchronized RGB images."""
    images: list[bytes] = []
    record = state.get(step)
    output = record.to_blob()
    if state.exists("wrist.png", step=record.step_idx):
        output["image_wrist_path"] = str(
            state.artifact_path("wrist.png", step=record.step_idx)
        )
        images.append(state.load_bytes("wrist.png", step=record.step_idx))
    if state.exists("camera.png", step=record.step_idx):
        output["image_cam_path"] = str(
            state.artifact_path("camera.png", step=record.step_idx)
        )
        images.insert(0, state.load_bytes("camera.png", step=record.step_idx))
    return ToolResult(data=output, error=output.pop("error", None), images=images)


@tool(readonly=True, exclude=("state",))
def view_camera_meta(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read camera intrinsics, crop, depth, and calibration metadata."""
    if not state.exists("camera_meta.json", step=step):
        return ToolResult(data={"step": step}, error="camera metadata is unavailable")
    return ToolResult(
        data={
            "step": state.get(step).step_idx,
            "camera_meta": state.load("camera_meta.json", step=step),
        }
    )
