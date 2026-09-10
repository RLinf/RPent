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

"""Native Franka tools and canonical RGB-D state capture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat

from robots.franka import perception
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolContext, ToolResult, readonly, tool

if TYPE_CHECKING:
    from robots.franka.toolkit import FrankaRuntime

Vec3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Pixel = Annotated[int, Field(ge=0)]


def _json_data(value: Any) -> Any:
    """Convert NumPy values received from the robot RPC into tool JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(item) for item in value]
    return value


def _result(data: dict[str, Any]) -> ToolResult:
    data = _json_data(data)
    error = data.pop("error", None)
    return ToolResult(data=data, error=error)


@tool
def move_delta(delta_xyz: Vec3, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Move the Franka TCP by a bounded base-frame xyz delta in meters.

    Args:
        delta_xyz: Base-frame x, y, z displacement in meters.
    """
    ctx.check_cancelled()
    return _result(ctx.robot.env.move_delta(np.asarray(delta_xyz, dtype=np.float32)))


@tool
def rotate_delta(delta_rpy: Vec3, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Rotate the Franka TCP by a bounded base-frame rpy delta in radians.

    Args:
        delta_rpy: Base-frame roll, pitch, yaw displacement in radians.
    """
    ctx.check_cancelled()
    return _result(ctx.robot.env.rotate_delta(np.asarray(delta_rpy, dtype=np.float32)))


@tool
def open_gripper(*, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Open the Franka gripper and wait for the command to settle."""
    ctx.check_cancelled()
    return _result(ctx.robot.env.set_gripper(open=True))


@tool
def close_gripper(*, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Close the Franka gripper and wait for the command to settle."""
    ctx.check_cancelled()
    return _result(ctx.robot.env.set_gripper(open=False))


@tool
def vla_grasp(
    prompt: str,
    max_chunks: Annotated[int, Field(ge=1, le=20)] = 4,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Run bounded real-world VLA action chunks for a local grasp attempt.

    Args:
        prompt: Non-empty instruction for this grasp attempt.
        max_chunks: Maximum number of VLA chunks to execute.
    """
    runtime = ctx.robot
    if runtime.model is None:
        return ToolResult(error="vla_grasp requires --vla-endpoint")
    if not prompt.strip():
        return ToolResult(error="prompt must be non-empty")
    observation = None
    for chunk in range(max_chunks):
        ctx.check_cancelled()
        if observation is None:
            observation = dict(runtime.env.get_observation())
        observation["task_descriptions"] = prompt
        actions = runtime.model.predict(observation, options={"mode": "eval"})
        result = runtime.env.chunk_step(actions)
        if result.get("terminated") or result.get("truncated"):
            break
        next_obs = result.get("observation")
        observation = dict(next_obs) if isinstance(next_obs, dict) else None
    # RGB-D arrays are captured as artifacts after execution, not repeated in JSON.
    return _result(
        {
            "ok": True,
            "chunks_executed": chunk + 1,
            "last_chunk": {
                key: value for key, value in result.items() if key != "observation"
            },
            "robot_state": runtime.env.get_robot_state(),
        }
    )


def dump_state(
    runtime: FrankaRuntime,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture robot state and synchronized camera artifacts in ``EnvState``."""
    observation = runtime.env.get_observation()
    robot_state = runtime.env.get_robot_state()
    metadata = runtime.env.get_camera_meta()
    with state.record_step(
        state=_json_data(robot_state),
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


def build_observation(state: EnvState, record: StepRecord) -> ToolResult:
    """Return recorded JSON and wrist/external PNGs in their declared order."""
    data = record.to_blob()
    data["images"] = []
    images = []
    for name in ("wrist", "camera"):
        if f"{name}.png" in record.artifacts:
            data["images"].append(name)
            images.append(state.load_bytes(f"{name}.png", step=record.step_idx))
    return ToolResult(data=data, images=images)


@tool
@readonly
def view_env_state(step: int = -1, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Read a Franka state snapshot and its synchronized RGB images.

    Args:
        step: Recorded step index, or -1 for the latest state.
    """
    return build_observation(ctx.state, ctx.state.get(step))


@tool
@readonly
def view_camera_meta(step: int = -1, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Read camera intrinsics, crop, depth, and calibration metadata.

    Args:
        step: Recorded step index, or -1 for the latest state.
    """
    if not ctx.state.exists("camera_meta.json", step=step):
        return ToolResult(data={"step": step}, error="camera metadata is unavailable")
    return _result(
        {
            "step": ctx.state.get(step).step_idx,
            "camera_meta": ctx.state.load("camera_meta.json", step=step),
        }
    )


@tool
@readonly
def view_perception_setup(
    step: int = -1, *, ctx: ToolContext[FrankaRuntime]
) -> ToolResult:
    """Read calibrated camera geometry and projection conventions.

    Args:
        step: Recorded step index, or -1 for the latest state.
    """
    return _result(perception.view_perception_setup(state=ctx.state, step=step))


@tool
@readonly
def back_project(
    row: Pixel,
    col: Pixel,
    step: int | None = None,
    camera: Literal["wrist", "third_person"] = "wrist",
    debug: bool = False,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Back-project one wrist or external-camera pixel into Franka base coordinates.

    Args:
        row: Pixel row in the recorded camera image.
        col: Pixel column in the recorded camera image.
        step: Recorded step index; omitted or -1 selects the latest state.
        camera: Camera containing the selected pixel.
        debug: Include projection diagnostics.
    """
    return _result(
        perception.back_project(
            row=row,
            col=col,
            step=step,
            camera=camera,
            debug=debug,
            state=ctx.state,
        )
    )


@tool
@readonly
def back_project_correspondence(
    third_person_row: Pixel | None = None,
    third_person_col: Pixel | None = None,
    wrist_row: Pixel | None = None,
    wrist_col: Pixel | None = None,
    pixels: list[dict[str, Any]] | None = None,
    step: int | None = None,
    debug: bool = False,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Fuse matched wrist and external-camera pixels into a Franka base point.

    Wrist pixels are required; matching external-camera pixels improve confidence.

    Args:
        third_person_row: Optional matched external-camera pixel row.
        third_person_col: Optional matched external-camera pixel column.
        wrist_row: Wrist-camera pixel row for a single correspondence.
        wrist_col: Wrist-camera pixel column for a single correspondence.
        pixels: Optional batch of pixel correspondences.
        step: Recorded step index; omitted or -1 selects the latest state.
        debug: Include projection diagnostics.
    """
    return _result(
        perception.back_project_correspondence(
            third_person_row=third_person_row,
            third_person_col=third_person_col,
            wrist_row=wrist_row,
            wrist_col=wrist_col,
            pixels=pixels,
            step=step,
            debug=debug,
            state=ctx.state,
        )
    )


FRANKA_TOOLS = (
    view_env_state,
    view_camera_meta,
    view_perception_setup,
    back_project,
    back_project_correspondence,
    move_delta,
    rotate_delta,
    open_gripper,
    close_gripper,
    vla_grasp,
)
