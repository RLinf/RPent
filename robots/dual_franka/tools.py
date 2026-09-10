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

"""Native dual-Franka tools and canonical RGB-D state capture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from robots.dual_franka import perception
from robots.franka.tools import (
    Pixel,
    Vec3,
    _json_data,
    _result,
    view_camera_meta,
    vla_grasp,
)
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolContext, ToolResult, readonly, tool

if TYPE_CHECKING:
    from robots.franka.toolkit import FrankaRuntime

Arm = Literal["left", "right"]


@tool
def move_delta(
    arm: Arm, delta_xyz: Vec3, *, ctx: ToolContext[FrankaRuntime]
) -> ToolResult:
    """Move one Franka TCP by a bounded world-frame xyz delta in meters.

    Args:
        arm: Arm to command; the other arm is left uncommanded.
        delta_xyz: World-frame x, y, z displacement in meters.
    """
    ctx.check_cancelled()
    return _result(
        ctx.robot.env.move_delta(arm, np.asarray(delta_xyz, dtype=np.float32))
    )


@tool
def rotate_delta(
    arm: Arm, delta_rpy: Vec3, *, ctx: ToolContext[FrankaRuntime]
) -> ToolResult:
    """Rotate one Franka TCP by a bounded world-frame rpy delta in radians.

    Args:
        arm: Arm to command; the other arm is left uncommanded.
        delta_rpy: World-frame roll, pitch, yaw displacement in radians.
    """
    ctx.check_cancelled()
    return _result(
        ctx.robot.env.rotate_delta(arm, np.asarray(delta_rpy, dtype=np.float32))
    )


@tool
def open_gripper(arm: Arm, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Open one Franka gripper and wait for the command to settle.

    Args:
        arm: Arm to command; the other arm is left uncommanded.
    """
    ctx.check_cancelled()
    return _result(ctx.robot.env.set_gripper(arm, open=True))


@tool
def close_gripper(arm: Arm, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Close one Franka gripper and wait for the command to settle.

    Args:
        arm: Arm to command; the other arm is left uncommanded.
    """
    ctx.check_cancelled()
    return _result(ctx.robot.env.set_gripper(arm, open=False))


# VLA policy-obs slots mapped to their EnvState stem and raw-frame key. The
# left wrist is ``main_images``; ``extra_view_images`` stacks [base, right wrist].
_VLA_CAMERAS = (
    ("left_wrist", "left_wrist_0_rgb"),
    ("base", "base_0_rgb"),
    ("right_wrist", "right_wrist_0_rgb"),
)


def _policy_views(main: Any, extra: Any, *, channels: bool) -> list[Any]:
    """Return policy views ordered ``[left_wrist, base, right_wrist]``.

    ``extra`` stacks the non-main views as ``[N,H,W,3]`` (RGB) or ``[N,H,W]``
    (depth), optionally behind a leading singleton batch dim stripped here.
    """
    views: list[Any] = [None if main is None else np.asarray(main)]
    if extra is None:
        return views
    array = np.asarray(extra)
    view_ndim = 4 if channels else 3
    if array.ndim == view_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == view_ndim:
        views.extend(array[index] for index in range(array.shape[0]))
    return views


def dump_state(
    runtime: FrankaRuntime,
    state: EnvState,
    *,
    command: dict[str, Any] | None,
    result: dict[str, Any] | None,
    elapsed_s: float | None,
) -> StepRecord:
    """Capture per-arm robot state and one canonical frame per camera.

    Each VLA camera stores its uncropped raw RGB-D as the single version; the
    policy-resolution view is derived on demand from it with RLinf's crop
    helpers (crop parameters live in ``camera_meta.json``). The policy view is
    persisted only as a fallback when a camera's raw frame is unavailable.
    """
    observation = runtime.env.get_observation()
    robot_state = runtime.env.get_robot_state()
    metadata = runtime.env.get_camera_meta()
    raw_frames = observation.get("raw_camera_frames") or {}
    raw_depths = observation.get("raw_camera_depths") or {}
    policy_frames = _policy_views(
        observation.get("main_images"),
        observation.get("extra_view_images"),
        channels=True,
    )
    policy_depths = _policy_views(
        observation.get("main_depths"),
        observation.get("extra_view_depths"),
        channels=False,
    )
    with state.record_step(
        state=_json_data(robot_state),
        command=command,
        result=result,
        elapsed_s=elapsed_s,
    ) as step:
        for index, (stem, raw_key) in enumerate(_VLA_CAMERAS):
            frame = raw_frames.get(raw_key)
            if frame is None and index < len(policy_frames):
                frame = policy_frames[index]
            if frame is not None:
                state.save(f"{stem}.png", np.asarray(frame), step=step)
            depth = raw_depths.get(raw_key)
            if depth is None and index < len(policy_depths):
                depth = policy_depths[index]
            if depth is not None:
                state.save(f"{stem}_depth.npy", np.asarray(depth), step=step)
        d455_image = observation.get("d455_images")
        if d455_image is not None:
            state.save("d455.png", np.asarray(d455_image), step=step)
        d455_depth = observation.get("d455_depths")
        if d455_depth is not None:
            state.save("d455_depth.npy", np.asarray(d455_depth), step=step)
        if metadata is not None:
            state.save("camera_meta.json", metadata, step=step)
    return state.get(step)


def build_observation(state: EnvState, record: StepRecord) -> ToolResult:
    """Return recorded JSON and left wrist/base/right wrist PNGs in order."""
    data = record.to_blob()
    data["images"] = []
    images = []
    for name in ("left_wrist", "base", "right_wrist"):
        if f"{name}.png" in record.artifacts:
            data["images"].append(name)
            images.append(state.load_bytes(f"{name}.png", step=record.step_idx))
    return ToolResult(data=data, images=images)


@tool
@readonly
def view_env_state(step: int = -1, *, ctx: ToolContext[FrankaRuntime]) -> ToolResult:
    """Read a dual-Franka state snapshot and its synchronized RGB images.

    Args:
        step: Recorded step index, or -1 for the latest state.
    """
    return build_observation(ctx.state, ctx.state.get(step))


@tool
@readonly
def back_project_base_pixel(
    row: Pixel,
    col: Pixel,
    target_name: str = "target",
    step: int | None = None,
    window_radius: Pixel = 2,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Back-project one base-camera pixel into shared right-base coordinates.

    Args:
        row: Pixel row in the recorded base-camera image.
        col: Pixel column in the recorded base-camera image.
        target_name: Target label for the diagnostic image.
        step: Recorded step index; omitted or -1 selects the latest state.
        window_radius: Depth sampling radius around the selected pixel.
    """
    return _result(
        perception.back_project_base_pixel(
            row=row,
            col=col,
            target_name=target_name,
            step=step,
            window_radius=window_radius,
            state=ctx.state,
        )
    )


@tool
@readonly
def back_project_d455_pixel(
    row: Pixel,
    col: Pixel,
    target_name: str = "target",
    step: int | None = None,
    window_radius: Pixel = 2,
    *,
    ctx: ToolContext[FrankaRuntime],
) -> ToolResult:
    """Back-project one D455 pixel into shared right-base coordinates.

    Args:
        row: Pixel row in the recorded D455 image.
        col: Pixel column in the recorded D455 image.
        target_name: Target label for the diagnostic image.
        step: Recorded step index; omitted or -1 selects the latest state.
        window_radius: Depth sampling radius around the selected pixel.
    """
    return _result(
        perception.back_project_d455_pixel(
            row=row,
            col=col,
            target_name=target_name,
            step=step,
            window_radius=window_radius,
            state=ctx.state,
        )
    )


DUAL_FRANKA_TOOLS = (
    view_env_state,
    view_camera_meta,
    back_project_base_pixel,
    back_project_d455_pixel,
    move_delta,
    rotate_delta,
    open_gripper,
    close_gripper,
    vla_grasp,
)
