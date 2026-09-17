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


def _result(data: dict[str, Any]) -> ToolResult:
    data = dict(data)
    error = data.pop("error", None)
    return ToolResult(data=data, error=error)


@tool
def move_delta(
    delta_xyz: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)],
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Move the Franka TCP by a bounded base-frame xyz delta in meters."""
    ctx.check_cancelled()
    return _result(ctx.robot.env.move_delta(np.asarray(delta_xyz, dtype=np.float32)))


@tool
def rotate_delta(
    delta_rpy: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)],
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Rotate the Franka TCP by a bounded base-frame rpy delta in radians."""
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
    """Run bounded real-world VLA action chunks for a local grasp attempt."""
    runtime = ctx.robot
    if runtime.model is None:
        raise RuntimeError("vla_grasp requires --vla-endpoint")
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    observation = None
    for chunk in range(int(max_chunks)):
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
    return _result(
        {
            "ok": True,
            "chunks_executed": chunk + 1,
            "last_chunk": result,
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


def build_observation(state: EnvState, record: StepRecord) -> ToolResult:
    """Return recorded JSON and wrist/external PNGs in their declared order."""
    data = record.to_blob()
    images = []
    for name, field in (("camera", "image_cam_path"), ("wrist", "image_wrist_path")):
        if state.exists(f"{name}.png", step=record.step_idx):
            data[field] = str(state.artifact_path(f"{name}.png", step=record.step_idx))
            images.append(state.load_bytes(f"{name}.png", step=record.step_idx))
    return ToolResult(data=data, images=images)


@tool
@readonly
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Read a Franka state snapshot and its synchronized RGB images."""
    return build_observation(ctx.state, ctx.state.get(step))


@tool
@readonly
def view_camera_meta(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Read camera intrinsics, crop, depth, and calibration metadata."""
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
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Read calibrated camera geometry and projection conventions."""
    return _result(perception.view_perception_setup(state=ctx.state, step=step))


@tool
@readonly
def back_project(
    row: Annotated[int, Field(ge=0)],
    col: Annotated[int, Field(ge=0)],
    step: int | None = None,
    camera: Literal["wrist", "third_person"] = "wrist",
    debug: Annotated[bool, Field(json_schema_extra={"default": False})] = False,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Back-project one wrist or external-camera pixel into Franka base coordinates."""
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
    third_person_row: Annotated[int, Field(ge=0)] | None = None,
    third_person_col: Annotated[int, Field(ge=0)] | None = None,
    wrist_row: Annotated[int, Field(ge=0)] | None = None,
    wrist_col: Annotated[int, Field(ge=0)] | None = None,
    pixels: list[dict[str, Any]] | None = None,
    step: int | None = None,
    debug: Annotated[bool, Field(json_schema_extra={"default": False})] = False,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Fuse matched wrist and external-camera pixels into a Franka base point."""
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


@tool
@readonly
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Call when the task is complete or unrecoverable. Halts the agent loop. Save any artifacts (recipe, audit) BEFORE calling finish.

    Args:
        status: Outcome, e.g. 'success', 'failure', or 'stuck'.
        summary: Short natural-language summary of the run.
    """
    return _result({"_finish": True, "status": status, "summary": summary})


FRANKA_TOOLS = (
    finish,
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
