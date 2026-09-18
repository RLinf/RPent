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

"""RoboTwin tool schemas and observation artifact helpers."""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import Field

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult
from rpent.tools.base import tool


def _tool_error(code: str, message: str, **details: Any) -> ToolResult:
    return ToolResult(data={"success": False, "code": code, **details}, error=message)


def _artifact_name(view: str, field: str) -> str:
    suffix = {
        "rgb": ".png",
        "depth": ".npy",
        "world_xyz": ".npy",
        "camera_meta": ".json",
    }[field]
    return f"{view}_{field}{suffix}"


def _load_world_xyz(
    env_state: EnvState,
    *,
    view: str,
    step: int | None,
) -> tuple[dict[str, Any] | None, np.ndarray | None, ToolResult | None]:
    """Load one persisted agent-visible world map without touching the env."""
    requested_step = -1 if step is None else int(step)
    try:
        record = env_state.get(requested_step)
    except Exception:
        return (
            None,
            None,
            _tool_error(
                "state_not_found",
                "The requested RoboTwin state artifact does not exist.",
                step=requested_step,
            ),
        )
    state = record.state
    actual_step = record.step_idx
    views = state.get("artifacts", {})
    if view not in views:
        return (
            None,
            None,
            _tool_error(
                "view_not_found",
                "The requested view is unavailable in this state.",
                view=view,
                available_views=sorted(views) if isinstance(views, dict) else [],
            ),
        )
    world_name = _artifact_name(view, "world_xyz")
    if world_name not in record.artifacts:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_not_found",
                "The requested view has no persisted world map.",
                view=view,
                step_idx=actual_step,
            ),
        )
    try:
        world = env_state.load(world_name, step=actual_step)
    except Exception as error:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_invalid",
                "The persisted world map cannot be read.",
                detail=str(error),
            ),
        )
    if world.ndim != 3 or world.shape[2] != 3:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_shape",
                "A RoboTwin world map must have shape [H,W,3].",
                actual_shape=list(world.shape),
            ),
        )
    return state, np.asarray(world), None


@tool(readonly=True, exclude=("env_state",))
def sample_world_xyz(
    env_state: EnvState,
    *,
    view: str,
    pixels: Annotated[
        list[Annotated[list[int], Field(min_length=2, max_length=2)]],
        Field(min_length=1, max_length=256),
    ],
    step: int | None = None,
    neighborhood: Annotated[
        int, Field(ge=0, le=32, json_schema_extra={"default": 1})
    ] = 1,
) -> ToolResult:
    """Read persisted same-frame world xyz around [row,col] pixels. The view is also the pixel coordinate space: use the exact view whose RGB supplied the pixels. The current state's view_specs gives each view's [height,width]. This is read-only and does not render or move the robot.

    Args:
        view: Artifact view and pixel coordinate space. It must match the RGB image used to choose pixels.
    """
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    radius = int(neighborhood)
    if radius < 0 or radius > 32:
        return _tool_error(
            "invalid_neighborhood",
            "neighborhood must be an integer from 0 through 32.",
        )
    if not isinstance(pixels, list) or not pixels or len(pixels) > 256:
        return _tool_error(
            "invalid_pixels",
            "pixels must contain between 1 and 256 [row,col] pairs.",
        )
    height, width = world.shape[:2]
    samples: list[dict[str, Any]] = []
    for pixel in pixels:
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or not all(isinstance(value, (int, np.integer)) for value in pixel)
        ):
            return _tool_error(
                "invalid_pixel",
                "Every pixel must be an integer [row,col] pair.",
                pixel=pixel,
            )
        row, col = (int(pixel[0]), int(pixel[1]))
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


@tool(readonly=True, exclude=("env_state",))
def query_world_map(
    env_state: EnvState,
    *,
    view: str,
    bbox: Annotated[list[int], Field(min_length=4, max_length=4)],
    step: int | None = None,
    max_points: Annotated[
        int, Field(ge=1, le=4096, json_schema_extra={"default": 256})
    ] = 256,
) -> ToolResult:
    """Read deterministic world-xyz samples from a half-open [row_start,col_start,row_end,col_end] region. The view is also the bbox coordinate space and must match the source RGB artifact; view_specs gives [height,width]. This is read-only.

    Args:
        view: Artifact view and bbox coordinate space. It must match the RGB image used to choose the bbox.
    """
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(isinstance(value, (int, np.integer)) for value in bbox)
    ):
        return _tool_error(
            "invalid_bbox",
            "bbox must be [row_start,col_start,row_end,col_end].",
        )
    row_start, col_start, row_end, col_end = map(int, bbox)
    height, width = world.shape[:2]
    if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
        return _tool_error(
            "bbox_out_of_bounds",
            "bbox must be a non-empty half-open region inside this view's "
            "world map. Use the exact artifact view whose RGB supplied the "
            "bbox coordinates.",
            bbox=list(map(int, bbox)),
            shape=[height, width],
            view=view,
            coordinate_space=view,
            valid_bbox=[0, 0, height, width],
        )
    limit = int(max_points)
    if limit < 1 or limit > 4096:
        return _tool_error(
            "invalid_max_points",
            "max_points must be an integer from 1 through 4096.",
        )
    region = world[row_start:row_end, col_start:col_end]
    valid_mask = np.isfinite(region).all(axis=2)
    local_rows, local_cols = np.nonzero(valid_mask)
    if not len(local_rows):
        return _tool_error(
            "no_valid_world_points",
            "The requested region contains no finite world coordinates.",
            bbox=list(map(int, bbox)),
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


def dump_observation(
    observation: dict[str, Any],
    *,
    env_state: EnvState,
    status: dict[str, Any],
    log: dict[str, Any] | None,
) -> StepRecord:
    """Persist one agent-visible observation without simulator oracle state."""
    step_idx = 0 if env_state.latest_step is None else env_state.latest_step + 1
    paths: dict[str, dict[str, str]] = {}
    view_specs: dict[str, dict[str, Any]] = {}
    for view_name, view in observation["views"].items():
        view_paths: dict[str, str] = {}
        if "rgb" in view:
            name = _artifact_name(view_name, "rgb")
            view_paths["rgb"] = str(env_state.artifact_path(name, step=step_idx))
        for field in ("depth", "world_xyz"):
            if field in view:
                name = _artifact_name(view_name, field)
                view_paths[field] = str(env_state.artifact_path(name, step=step_idx))
        if "camera_meta" in view:
            name = _artifact_name(view_name, "camera_meta")
            view_paths["camera_meta"] = str(
                env_state.artifact_path(name, step=step_idx)
            )
        paths[view_name] = view_paths
        shape_source = next(
            (
                np.asarray(view[field])
                for field in ("rgb", "world_xyz", "depth")
                if field in view
            ),
            None,
        )
        if shape_source is not None and shape_source.ndim >= 2:
            view_specs[view_name] = {
                "coordinate_space": view_name,
                "image_shape": [
                    int(shape_source.shape[0]),
                    int(shape_source.shape[1]),
                ],
                "pixel_order": "row_col",
            }

    state = {
        "step_idx": step_idx,
        "task_name": observation["task_name"],
        "task_language": observation["task_language"],
        "robot_state": observation["robot_state"],
        "episode_status": status,
        "artifacts": paths,
        "view_specs": view_specs,
        "log": log,
    }
    eval_success = status.get("eval_success") is True
    with env_state.record_step(
        state=state,
        terminated=eval_success,
        truncated=False,
        command=(log or {}).get("command"),
        result=(log or {}).get("result"),
        elapsed_s=(log or {}).get("elapsed_s"),
        extras={"task_language": observation.get("task_language")},
    ) as recorded_step:
        for view_name, view in observation["views"].items():
            for field in ("rgb", "depth", "world_xyz", "camera_meta"):
                if field in view:
                    env_state.save(
                        _artifact_name(view_name, field),
                        view[field],
                        step=recorded_step,
                    )
    return env_state.get(step_idx)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Read one EnvState step and its synchronized RoboTwin observation artifacts. Step -1 selects the latest entry. Embeds the head, left wrist, and right wrist RGB images when available.

    Args:
        step: Step number; 0 = initial, -1 = latest.
    """
    images: list[bytes] = []
    try:
        record = state.get(step)
    except Exception as error:
        return ToolResult(error=f"state step not available: {error}")
    result: dict[str, Any] = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
    }
    result["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    for view in ("head", "left_wrist", "right_wrist"):
        name = _artifact_name(view, "rgb")
        if name in record.artifacts:
            try:
                images.append(state.load_bytes(name, step=record.step_idx))
            except FileNotFoundError:
                pass
    return ToolResult(data=result, images=images)
