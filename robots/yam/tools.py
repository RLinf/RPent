# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""YAM tool schemas and observation artifact helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.session import EnvState, StepRecord
from rpent.tools.toolkit import readonly


def _tool_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message, **details}}


def _artifact_name(view: str, field: str) -> str:
    suffix = {
        "rgb": ".png",
        "depth": ".npy",
        "world_xyz": ".npy",
        "camera_meta": ".json",
    }[field]
    return f"{view}_{field}{suffix}"


def _load_world_xyz(env_state: EnvState, *, view: str, step: int | None):
    requested_step = -1 if step is None else int(step)
    try:
        record = env_state.get(requested_step)
    except Exception:
        return (
            None,
            None,
            _tool_error(
                "state_not_found",
                "The requested YAM state does not exist.",
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
                "The requested view is unavailable.",
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
                "This view has no persisted world map.",
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
                "A YAM world map must have shape [H,W,3].",
                actual_shape=list(world.shape),
            ),
        )
    return state, np.asarray(world), None


@readonly
def sample_world_xyz(
    env_state: EnvState,
    *,
    view: str,
    pixels: list[list[int]],
    step: int | None = None,
    neighborhood: int = 1,
) -> dict[str, Any]:
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    radius = int(neighborhood)
    if radius < 0 or radius > 32:
        return _tool_error("invalid_neighborhood", "neighborhood must be 0 through 32.")
    if not isinstance(pixels, list) or not pixels or len(pixels) > 256:
        return _tool_error(
            "invalid_pixels", "pixels must contain 1 to 256 [row,col] pairs."
        )
    height, width = world.shape[:2]
    samples = []
    for pixel in pixels:
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            return _tool_error(
                "invalid_pixel", "Every pixel must be [row,col].", pixel=pixel
            )
        row, col = int(pixel[0]), int(pixel[1])
        if row < 0 or row >= height or col < 0 or col >= width:
            return _tool_error(
                "pixel_out_of_bounds",
                "The pixel is outside this view's world map.",
                pixel=[row, col],
                shape=[height, width],
                view=view,
                valid_row_range=[0, height - 1],
                valid_col_range=[0, width - 1],
            )
        region = world[
            max(0, row - radius) : min(height, row + radius + 1),
            max(0, col - radius) : min(width, col + radius + 1),
        ].reshape(-1, 3)
        finite = np.isfinite(region).all(axis=1)
        if not np.any(finite):
            return _tool_error(
                "no_valid_world_points",
                "The requested pixel neighborhood has no finite xyz.",
                pixel=[row, col],
            )
        xyz = np.nanmedian(region[finite], axis=0)
        samples.append(
            {
                "pixel": [row, col],
                "valid": True,
                "xyz": xyz.tolist(),
                "valid_points": int(finite.sum()),
            }
        )
    return {
        "success": True,
        "step_idx": state["step_idx"],
        "view": view,
        "coordinate_space": view,
        "image_shape": [height, width],
        "pixel_order": "row_col",
        "coordinate_order": "xyz",
        "frame": "left_base",
        "unit": "metre",
        "neighborhood": radius,
        "samples": samples,
    }


@readonly
def query_world_map(
    env_state: EnvState,
    *,
    view: str,
    bbox: list[int],
    step: int | None = None,
    max_points: int = 256,
) -> dict[str, Any]:
    state, world, error = _load_world_xyz(env_state, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return _tool_error(
            "invalid_bbox", "bbox must be [row_start,col_start,row_end,col_end]."
        )
    row_start, col_start, row_end, col_end = map(int, bbox)
    height, width = world.shape[:2]
    if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
        return _tool_error(
            "bbox_out_of_bounds",
            "bbox must be a non-empty half-open region inside the view.",
            bbox=list(map(int, bbox)),
            shape=[height, width],
        )
    limit = int(max_points)
    if limit < 1 or limit > 4096:
        return _tool_error("invalid_max_points", "max_points must be 1 through 4096.")
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
    indices = (
        np.linspace(0, len(xyz) - 1, limit).astype(int)
        if len(xyz) > limit
        else np.arange(len(xyz))
    )
    points = [
        {
            "pixel": [int(row_start + local_rows[i]), int(col_start + local_cols[i])],
            "xyz": xyz[i].tolist(),
        }
        for i in indices
    ]
    return {
        "success": True,
        "step_idx": state["step_idx"],
        "view": view,
        "coordinate_space": view,
        "image_shape": [height, width],
        "bbox": [row_start, col_start, row_end, col_end],
        "bbox_interval": "half_open",
        "pixel_order": "row_col",
        "coordinate_order": "xyz",
        "frame": "left_base",
        "unit": "metre",
        "valid_points": int(len(xyz)),
        "returned_points": len(points),
        "xyz_min": np.min(xyz, axis=0).tolist(),
        "xyz_max": np.max(xyz, axis=0).tolist(),
        "xyz_median": np.median(xyz, axis=0).tolist(),
        "points": points,
    }


def dump_observation(
    observation: dict[str, Any],
    *,
    env_state: EnvState,
    status: dict[str, Any],
    log: dict[str, Any] | None,
) -> StepRecord:
    step_idx = 0 if env_state.latest_step is None else env_state.latest_step + 1
    paths: dict[str, dict[str, str]] = {}
    view_specs: dict[str, dict[str, Any]] = {}
    for view_name, view in observation["views"].items():
        view_paths: dict[str, str] = {}
        for field in ("rgb", "depth", "world_xyz", "camera_meta"):
            if field in view:
                name = _artifact_name(view_name, field)
                view_paths[field] = str(env_state.artifact_path(name, step=step_idx))
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
                "image_shape": [int(shape_source.shape[0]), int(shape_source.shape[1])],
                "pixel_order": "row_col",
            }
            if "world_xyz_limitation" in view:
                view_specs[view_name]["world_xyz_limitation"] = view[
                    "world_xyz_limitation"
                ]

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
    with env_state.record_step(
        state=state,
        terminated=status.get("eval_success") is True,
        truncated=bool(
            status.get("terminal_event") in {"failure", "abort"}
            or status.get("stop_requested")
            or int(status["take_action_cnt"]) >= int(status["step_lim"])
        ),
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


@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict[str, Any]:
    try:
        record = state.get(step)
    except Exception as error:
        return {"error": f"state step not available: {error}"}
    result = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
    }
    for slot, view in (
        ("_image_bytes", "top"),
        ("_image_cam_bytes", "left"),
        ("_image_wrist_bytes", "right"),
    ):
        name = _artifact_name(view, "rgb")
        if name in record.artifacts:
            try:
                result[slot] = state.load_bytes(name, step=record.step_idx)
            except FileNotFoundError:
                pass
    return result


TOOLS_SPEC = [
    {
        "name": "status",
        "description": "Read fresh episode rules: ready, awaiting ready, terminal, stop or budget. Does not establish visual clearance or drive hardware.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "view_env_state",
        "description": "Read one synchronized YAM state and image artifacts.",
        "input_schema": {
            "type": "object",
            "properties": {"step": {"type": "integer", "default": -1}},
        },
    },
    {
        "name": "render",
        "description": "Capture a fresh synchronized YAM observation.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "reset",
        "description": "Operator-approved reset for exploration or a fresh task attempt.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "sample_world_xyz",
        "description": "Read persisted same-frame YAM world xyz around [row,col] pixels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["top", "left", "right"]},
                "pixels": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 256,
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "step": {"type": ["integer", "null"]},
                "neighborhood": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32,
                    "default": 1,
                },
            },
            "required": ["view", "pixels"],
        },
    },
    {
        "name": "query_world_map",
        "description": "Read deterministic world-xyz samples from a half-open bbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["top", "left", "right"]},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "step": {"type": ["integer", "null"]},
                "max_points": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4096,
                    "default": 256,
                },
            },
            "required": ["view", "bbox"],
        },
    },
    {
        "name": "pi05_act",
        "description": "Run the YAM Pi0.5 qpos14 policy for one or more short chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "integer", "minimum": 1, "default": 1},
                "use_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 30,
                },
                "prompt": {"type": ["string", "null"]},
            },
        },
    },
    {
        "name": "move_to",
        "description": "Plan and move one YAM arm to a left-base xyz and wxyz orientation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "xyz_bounds": {
                    "type": ["array", "null"],
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "description": "Optional task-valid [lower_xyz, upper_xyz] containing xyz. Try at most three plans within it; execute only one. Reobserve after recoverable motion failure before choosing another point. Does not relax arrival tolerance or collision checks.",
                },
                "quat": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "gripper": {"type": ["number", "null"]},
                "substeps": {"type": "integer", "minimum": 0, "default": 25},
            },
            "required": ["arm", "xyz"],
        },
    },
    {
        "name": "rotate_wrist",
        "description": "Rotate one wrist about world Z by a relative angle in degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "delta_yaw_deg": {"type": "number"},
                "gripper": {"type": ["number", "null"]},
                "substeps": {"type": "integer", "minimum": 0, "default": 25},
            },
            "required": ["arm", "delta_yaw_deg"],
        },
    },
    {
        "name": "set_gripper",
        "description": "Linearly move one normalized YAM gripper, where 0=closed and 1=open.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "val": {"type": "number", "minimum": 0, "maximum": 1},
                "steps": {"type": "integer", "minimum": 1, "default": 10},
            },
            "required": ["arm", "val"],
        },
    },
    {
        "name": "release",
        "description": "Open one YAM gripper to 1.0.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "val": {"type": "number", "default": 1.0},
                "steps": {"type": "integer", "minimum": 1, "default": 10},
            },
            "required": ["arm"],
        },
    },
    {
        "name": "finish",
        "description": "Stop the run. Fresh env eval_success is authoritative.",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}, "summary": {"type": "string"}},
            "required": ["status", "summary"],
        },
    },
]
