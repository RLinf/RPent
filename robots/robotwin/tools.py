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

"""Native RoboTwin action and perception tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field

from robots.robotwin.robot_spec import MODEL_SPEC, ROBOTWIN_CAMERA_NAMES
from rpent.session import EnvState
from rpent.tools import ToolContext, ToolResult, readonly, tool

if TYPE_CHECKING:
    from robots.robotwin.toolkit import RoboTwinRuntime


def _tool_error(code: str, message: str, **details: Any) -> ToolResult:
    return ToolResult(data={"success": False, "code": code, **details}, error=message)


def _load_world_xyz(
    env_state: EnvState,
    *,
    view: str,
    step: int | None,
) -> tuple[dict[str, Any], np.ndarray] | ToolResult:
    """Load one persisted agent-visible world map without touching the env."""
    from robots.robotwin.toolkit import _artifact_name

    requested_step = -1 if step is None else step
    try:
        record = env_state.get(requested_step)
    except (LookupError, ValueError):
        return _tool_error(
            "state_not_found",
            "The requested RoboTwin state artifact does not exist.",
            step=requested_step,
        )
    state = record.state
    views = state["artifacts"]
    if view not in views:
        return _tool_error(
            "view_not_found",
            "The requested view is unavailable in this state.",
            view=view,
            available_views=sorted(views),
        )
    world_name = _artifact_name(view, "world_xyz")
    if world_name not in record.artifacts:
        return _tool_error(
            "world_xyz_not_found",
            "The requested view has no persisted world map.",
            view=view,
            step_idx=record.step_idx,
        )
    try:
        world = env_state.load(world_name, step=record.step_idx)
    except Exception as error:
        return _tool_error(
            "world_xyz_invalid",
            "The persisted world map cannot be read.",
            detail=str(error),
        )
    if world.ndim != 3 or world.shape[2] != 3:
        return _tool_error(
            "world_xyz_shape",
            "A RoboTwin world map must have shape [H,W,3].",
            actual_shape=list(world.shape),
        )
    return state, world


@tool
@readonly
def sample_world_xyz(
    view: str,
    pixels: Annotated[
        list[Annotated[list[int], Field(min_length=2, max_length=2)]],
        Field(min_length=1, max_length=256),
    ],
    step: int | None = None,
    neighborhood: Annotated[
        int, Field(ge=0, le=32, json_schema_extra={"default": 1})
    ] = 1,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Read persisted same-frame world xyz around [row,col] pixels. The view is also the pixel coordinate space: use the exact view whose RGB supplied the pixels. The current state's view_specs gives each view's [height,width]. This is read-only and does not render or move the robot.

    Args:
        view: Artifact view and pixel coordinate space. It must match the RGB image used to choose pixels.
    """
    loaded = _load_world_xyz(ctx.state, view=view, step=step)
    if isinstance(loaded, ToolResult):
        return loaded
    state, world = loaded
    radius = neighborhood
    height, width = world.shape[:2]
    samples: list[dict[str, Any]] = []
    for pixel in pixels:
        row, col = pixel
        if row < 0 or row >= height or col < 0 or col >= width:
            return _tool_error(
                "pixel_out_of_bounds",
                "The pixel is outside this view's world map. Use the exact "
                "artifact view whose RGB supplied the pixel; do not reuse "
                "high-resolution pixels with a base-resolution view.",
                pixel=[row, col],
                shape=[height, width],
                view=view,
                coordinate_space=view,
                valid_row_range=[0, height - 1],
                valid_col_range=[0, width - 1],
            )
        row_start = max(0, row - radius)
        row_end = min(height, row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(width, col + radius + 1)
        region = world[row_start:row_end, col_start:col_end].reshape(-1, 3)
        finite_counts = np.isfinite(region).sum(axis=0)
        if np.any(finite_counts == 0):
            return _tool_error(
                "no_valid_world_points",
                "The requested pixel neighborhood has no finite xyz coordinate.",
                pixel=[row, col],
                neighborhood=radius,
            )
        xyz = np.nanmedian(region, axis=0)
        samples.append(
            {
                "pixel": [row, col],
                "valid": True,
                "xyz": xyz.tolist(),
                "valid_points": int(np.isfinite(region).all(axis=1).sum()),
                "valid_coordinates": finite_counts.tolist(),
            }
        )
    return ToolResult(
        data={
            "success": True,
            "step_idx": state["step_idx"],
            "view": view,
            "coordinate_space": view,
            "image_shape": [height, width],
            "pixel_order": "row_col",
            "coordinate_order": "xyz",
            "frame": "world",
            "unit": "metre",
            "neighborhood": radius,
            "samples": samples,
        }
    )


@tool
@readonly
def query_world_map(
    view: str,
    bbox: Annotated[list[int], Field(min_length=4, max_length=4)],
    step: int | None = None,
    max_points: Annotated[
        int, Field(ge=1, le=4096, json_schema_extra={"default": 256})
    ] = 256,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Read deterministic world-xyz samples from a half-open [row_start,col_start,row_end,col_end] region. The view is also the bbox coordinate space and must match the source RGB artifact; view_specs gives [height,width]. This is read-only.

    Args:
        view: Artifact view and bbox coordinate space. It must match the RGB image used to choose the bbox.
    """
    loaded = _load_world_xyz(ctx.state, view=view, step=step)
    if isinstance(loaded, ToolResult):
        return loaded
    state, world = loaded
    row_start, col_start, row_end, col_end = bbox
    height, width = world.shape[:2]
    if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
        return _tool_error(
            "bbox_out_of_bounds",
            "bbox must be a non-empty half-open region inside this view's "
            "world map. Use the exact artifact view whose RGB supplied the "
            "bbox coordinates.",
            bbox=bbox,
            shape=[height, width],
            view=view,
            coordinate_space=view,
            valid_bbox=[0, 0, height, width],
        )
    limit = max_points
    region = world[row_start:row_end, col_start:col_end]
    valid_mask = np.isfinite(region).all(axis=2)
    local_rows, local_cols = np.nonzero(valid_mask)
    if not len(local_rows):
        return _tool_error(
            "no_valid_world_points",
            "The requested region contains no finite world coordinates.",
            bbox=bbox,
        )
    xyz = region[local_rows, local_cols]
    if len(xyz) > limit:
        indices = np.linspace(0, len(xyz) - 1, limit).astype(int)
    else:
        indices = np.arange(len(xyz))
    points = [
        {
            "pixel": [
                int(row_start + local_rows[index]),
                int(col_start + local_cols[index]),
            ],
            "xyz": xyz[index].tolist(),
        }
        for index in indices
    ]
    return ToolResult(
        data={
            "success": True,
            "step_idx": state["step_idx"],
            "view": view,
            "coordinate_space": view,
            "image_shape": [height, width],
            "bbox": [row_start, col_start, row_end, col_end],
            "bbox_interval": "half_open",
            "pixel_order": "row_col",
            "coordinate_order": "xyz",
            "frame": "world",
            "unit": "metre",
            "valid_points": int(len(xyz)),
            "returned_points": len(points),
            "xyz_min": np.min(xyz, axis=0).tolist(),
            "xyz_max": np.max(xyz, axis=0).tolist(),
            "xyz_median": np.median(xyz, axis=0).tolist(),
            "points": points,
        }
    )


def _qmult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _completion(
    *, requested: int, executed: int, status: dict[str, Any]
) -> dict[str, Any]:
    step_lim = status.get("step_lim")
    budget_exhausted = step_lim is not None and int(
        status.get("take_action_cnt", 0)
    ) >= int(step_lim)
    completed = executed == requested
    if status.get("eval_success") is True:
        stop_reason = "native_success"
    elif budget_exhausted:
        stop_reason = "budget_exhausted"
    elif completed:
        stop_reason = "completed"
    else:
        stop_reason = "runtime_failure"
    return {
        "completed": completed,
        "requested_steps": requested,
        "executed_steps": executed,
        "stop_reason": stop_reason,
    }


def _apply_qpos_updates(
    ctx: ToolContext[RoboTwinRuntime], updates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compose each waypoint from fresh state and account for it before cancellation."""
    runtime = ctx.robot
    env = runtime.env
    executed = 0
    for update in updates:
        ctx.check_cancelled()
        action = np.asarray(
            env.last_info["robot_state"]["qpos_target14"], dtype=np.float64
        ).copy()
        offset = 0 if update["arm"] == "left" else 7
        if "arm_qpos" in update:
            action[offset : offset + 6] = update["arm_qpos"]
        if update.get("gripper") is not None:
            action[offset + 6] = update["gripper"]
        obs, _, _, _, info = env.step(action, action_type="qpos")
        count = int(info["executed_actions"])
        executed += count
        runtime.native_actions += count
        ctx.record_frame(obs["main_images"])
        ctx.check_cancelled()
        if env.terminated or env.truncated:
            break
    return {
        "action_type": "qpos",
        "requested_actions": len(updates),
        "executed_actions": executed,
        "episode_status": env.last_info["episode_status"],
    }


@tool
def lingbot_act(
    chunks: Annotated[int, Field(ge=1, json_schema_extra={"default": 4})] = 4,
    use_length: Annotated[Literal[50], Field(json_schema_extra={"default": 50})] = 50,
    prompt: str | None = None,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Run LingBot-VLA eef16 actions using the native task instruction. The optional prompt is recorded but never sent to the policy."""
    runtime = ctx.robot
    env = runtime.env
    executed = 0
    requested = chunks * MODEL_SPEC.use_length
    native_prompt = None
    for _ in range(chunks):
        ctx.check_cancelled()
        status = env.last_info["episode_status"]
        step_lim = status.get("step_lim")
        budget_exhausted = step_lim is not None and int(
            status["take_action_cnt"]
        ) >= int(step_lim)
        if status["eval_success"] is True or budget_exhausted:
            break
        observation = {
            "views": {
                camera: {"rgb": np.asarray(env.render_camera(camera))}
                for camera in ROBOTWIN_CAMERA_NAMES
            },
            "robot_state": env.last_info["robot_state"],
            "task_language": env.get_task_language(),
        }
        native_prompt = observation["task_language"]
        actions = runtime.model.infer(observation)[: MODEL_SPEC.use_length]
        ctx.check_cancelled()
        all_frames = env.execution_capabilities.get("chunk_step_all_frames") is True
        payload, _, _, _, info = env.chunk_step(
            actions,
            action_type="ee",
            return_all_frames=all_frames,
        )
        count = int(info["executed_actions"])
        executed += count
        runtime.policy_actions += count
        runtime.native_actions += count
        if all_frames:
            for frame in payload["frames"]:
                ctx.record_frame(frame)
        else:
            ctx.record_frame(payload["main_images"])
        ctx.check_cancelled()
    status = env.last_info["episode_status"]
    return ToolResult(
        data={
            **_completion(requested=requested, executed=executed, status=status),
            "success": True,
            "prompt": native_prompt,
            "agent_prompt_ignored": prompt is not None,
            "ignored_agent_prompt": prompt,
            "episode_status": status,
        }
    )


def _move_to(
    ctx: ToolContext[RoboTwinRuntime],
    *,
    arm: str,
    xyz: list[float],
    quat: list[float] | None,
    gripper: float | None,
    substeps: int,
) -> ToolResult:
    ctx.check_cancelled()
    env = ctx.robot.env
    state = env.last_info["robot_state"]
    if quat is None:
        quat = np.asarray(state[f"{arm}_eef_pose"], dtype=np.float64)[3:].tolist()
    target = np.asarray([*xyz, *quat], dtype=np.float64)
    planned = env.plan_arm_path(arm, target)
    ctx.check_cancelled()
    if (
        planned["status"] != "Success"
        or planned.get("position") is None
        or not len(planned["position"])
    ):
        return ToolResult(
            data={
                "completed": False,
                "requested_steps": 0,
                "executed_steps": 0,
                "stop_reason": "plan_failed",
                "success": False,
                "plan_status": planned["status"],
                "hint": "target may be unreachable or in collision",
            },
            error="Arm path planning failed.",
        )
    path = np.asarray(planned["position"], dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 6:
        raise ValueError("RoboTwin planned path must have shape [N,6]")
    if substeps == 1:
        path = path[-1:]
    elif substeps >= 2 and len(path) > substeps:
        indices = np.linspace(0, len(path) - 1, substeps).astype(int)
        path = path[indices]
    updates = [
        {"arm": arm, "arm_qpos": waypoint, "gripper": gripper} for waypoint in path
    ]
    execution = _apply_qpos_updates(ctx, updates)
    final_pose = np.asarray(
        env.last_info["robot_state"][f"{arm}_eef_pose"], dtype=np.float64
    )
    return ToolResult(
        data={
            **execution,
            **_completion(
                requested=len(updates),
                executed=execution["executed_actions"],
                status=execution["episode_status"],
            ),
            "success": True,
            "plan_status": planned["status"],
            "waypoints": len(path),
            "final_eef_xyz": final_pose[:3].tolist(),
            "final_dist_m": float(
                np.linalg.norm(final_pose[:3] - np.asarray(xyz, dtype=np.float64))
            ),
        }
    )


@tool
def move_to(
    arm: Literal["left", "right"],
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    quat: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
    gripper: float | None = None,
    substeps: Annotated[int, Field(ge=0, json_schema_extra={"default": 25})] = 25,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Plan and move one arm to a world-frame xyz and wxyz orientation. The native planner returns qpos waypoints executed with fresh state."""
    return _move_to(
        ctx, arm=arm, xyz=xyz, quat=quat, gripper=gripper, substeps=substeps
    )


@tool
def rotate_wrist(
    arm: Literal["left", "right"],
    delta_yaw_deg: float,
    gripper: float | None = None,
    substeps: Annotated[int, Field(ge=0, json_schema_extra={"default": 25})] = 25,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Rotate one EEF about world Z by a relative angle in degrees."""
    state = ctx.robot.env.last_info["robot_state"]
    pose = np.asarray(state[f"{arm}_eef_pose"], dtype=np.float64)
    yaw = np.deg2rad(delta_yaw_deg)
    world_z = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    result = _move_to(
        ctx,
        arm=arm,
        xyz=pose[:3].tolist(),
        quat=_qmult(world_z, pose[3:]).tolist(),
        gripper=gripper,
        substeps=substeps,
    )
    result.data["requested_delta_yaw_deg"] = delta_yaw_deg
    return result


def _set_gripper(
    ctx: ToolContext[RoboTwinRuntime], *, arm: str, val: float, steps: int
) -> ToolResult:
    ctx.check_cancelled()
    env = ctx.robot.env
    current = float(env.last_info["robot_state"][f"{arm}_gripper"])
    values = [
        current + (val - current) * index / steps for index in range(1, steps + 1)
    ]
    execution = _apply_qpos_updates(
        ctx, [{"arm": arm, "gripper": value} for value in values]
    )
    return ToolResult(
        data={
            **execution,
            **_completion(
                requested=steps,
                executed=execution["executed_actions"],
                status=execution["episode_status"],
            ),
            "success": True,
            "gripper_val": float(env.last_info["robot_state"][f"{arm}_gripper"]),
        }
    )


@tool
def set_gripper(
    arm: Literal["left", "right"],
    val: Annotated[float, Field(ge=0, le=1)],
    steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 10})] = 10,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Linearly move one normalized gripper to val over the requested number of actions."""
    return _set_gripper(ctx, arm=arm, val=val, steps=steps)


@tool
def release(
    arm: Literal["left", "right"],
    val: Annotated[float, Field(json_schema_extra={"default": 1.0})] = 1.0,
    steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 10})] = 10,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Open one gripper to 1.0 over 10 native actions by default."""
    return _set_gripper(ctx, arm=arm, val=val, steps=steps)


@tool
def finish(
    status: str, summary: str, *, ctx: ToolContext[RoboTwinRuntime]
) -> ToolResult:
    """Stop the run using native episode status as authority. Requesting success cannot override TASK_ENV.eval_success.

    Args:
        status: Requested task outcome.
        summary: Summary of what worked and what failed.
    """
    requested_success = status.lower() == "success"
    try:
        native = ctx.robot.status()
    except Exception as error:  # The terminal tool must still stop the Planner.
        return ToolResult(
            data={
                "_finish": True,
                "status": "error",
                "summary": summary,
                "requested_status": status,
                "requested_success": requested_success,
                "runtime_error": f"{type(error).__name__}: {error}",
            }
        )
    verified_success = native.get("eval_success") is True
    if verified_success:
        reported_status = "success"
    elif requested_success:
        reported_status = "failure"
    else:
        reported_status = status
    return ToolResult(
        data={
            "_finish": True,
            "status": reported_status,
            "summary": summary,
            "requested_success": requested_success,
            "success": verified_success,
            "episode_status": native,
        }
    )


@tool
@readonly
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    ctx: ToolContext[RoboTwinRuntime],
) -> ToolResult:
    """Read one EnvState step and its synchronized RoboTwin observation artifacts. Step -1 selects the latest entry. Embeds the head, left wrist, and right wrist RGB images when available.

    Args:
        step: Step number; 0 = initial, -1 = latest.
    """
    from robots.robotwin.toolkit import build_observation

    try:
        record = ctx.state.get(step)
    except (LookupError, ValueError) as error:
        return ToolResult(error=f"state step not available: {error}")
    data, images = build_observation(ctx.state, record)
    return ToolResult(data=data, images=images)


@tool
def render(*, ctx: ToolContext[RoboTwinRuntime]) -> ToolResult:
    """Capture a fresh synchronized RoboTwin agent observation."""
    return ToolResult(data={"success": True})


ROBOTWIN_TOOLS = (
    view_env_state,
    render,
    sample_world_xyz,
    query_world_map,
    lingbot_act,
    move_to,
    rotate_wrist,
    set_gripper,
    release,
    finish,
)
