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

"""RoboDojo tool specs and backend-specific control adapters."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from robots.robodojo.access import public_episode_budget
from rpent.session import EnvState
from rpent.tools import ToolResult, tool, tool_error


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
            tool_error(
                "state_not_found",
                "The requested RoboDojo state artifact does not exist.",
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
            tool_error(
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
            tool_error(
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
            tool_error(
                "world_xyz_invalid",
                "The persisted world map cannot be read.",
                detail=str(error),
            ),
        )
    if world.ndim != 3 or world.shape[2] != 3:
        return (
            None,
            None,
            tool_error(
                "world_xyz_shape",
                "A RoboDojo world map must have shape [H,W,3].",
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
        return tool_error(
            "invalid_neighborhood",
            "neighborhood must be an integer from 0 through 32.",
        )
    if not isinstance(pixels, list) or not pixels or len(pixels) > 256:
        return tool_error(
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
            return tool_error(
                "invalid_pixel",
                "Every pixel must be an integer [row,col] pair.",
                pixel=pixel,
            )
        row, col = (int(pixel[0]), int(pixel[1]))
        if row < 0 or row >= height or col < 0 or col >= width:
            return tool_error(
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
        region = region[np.isfinite(region).all(axis=1)]
        finite_counts = np.full(3, len(region))
        if not len(region):
            return tool_error(
                "no_valid_world_points",
                "The requested pixel neighborhood has no finite xyz coordinate.",
                pixel=[row, col],
                neighborhood=radius,
            )
        xyz = np.median(region, axis=0)
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
        return tool_error(
            "invalid_bbox",
            "bbox must be [row_start,col_start,row_end,col_end].",
        )
    row_start, col_start, row_end, col_end = map(int, bbox)
    height, width = world.shape[:2]
    if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
        return tool_error(
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
        return tool_error(
            "invalid_max_points",
            "max_points must be an integer from 1 through 4096.",
        )
    region = world[row_start:row_end, col_start:col_end]
    valid_mask = np.isfinite(region).all(axis=2)
    local_rows, local_cols = np.nonzero(valid_mask)
    if not len(local_rows):
        return tool_error(
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


# (camera key in the obs dict, artifact base name). Mirrors the ``frame_channels``
# declared in ``robot_spec.ROBODOJO_DASHBOARD_SPEC``.
CAMERA_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("cam_head", "cam_head.png"),
    ("cam_left_wrist", "cam_left_wrist.png"),
    ("cam_right_wrist", "cam_right_wrist.png"),
)


@tool(readonly=True, exclude=("state",))
def view_env_state(step: int = -1, *, state: Any) -> ToolResult:
    """Read one recorded RoboDojo state: robot joint/ee state, camera shapes and calibration, the task instruction, action budget and the three camera RGB images (cam_head, cam_left_wrist, cam_right_wrist). Step -1 selects the latest record. episode_status contains native step, step_limit and remaining_steps at capture time; the top-level step is the observation index.

    Args:
        step: Recorded step to read (-1 = latest)"""
    try:
        record = state.get(step)
    except Exception as error:
        payload = {"error": f"state step not available: {error}"}
        return ToolResult(data=payload)
    result: dict[str, Any] = {
        "step": record.step_idx,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "task_language": record.extras.get("task_language"),
        "episode_status": record.extras.get("episode_status"),
    }
    result["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    images: list[bytes] = []
    for _, artifact in CAMERA_ARTIFACTS:
        if artifact in record.artifacts:
            try:
                images.append(state.load_bytes(artifact, step=record.step_idx))
            except FileNotFoundError:
                pass
    return ToolResult(data=result, images=images)


def _summarize_obs(obs: dict) -> dict:
    from robots.robodojo.access import public_observation

    obs = public_observation(obs)
    vision = obs.get("vision", {})
    state_data = obs.get("state", {})
    return {
        "instruction": obs.get("instruction"),
        "cameras": {
            name: {
                "shape": cam.get("shape"),
                "has_depth": cam.get("distance_to_image_plane") is not None
                or cam.get("depth") is not None,
                "intrinsic_matrix": _jsonable(cam.get("intrinsic_matrix")),
                "extrinsic_matrix": _jsonable(cam.get("extrinsic_matrix")),
                "color_dtype": str(cam.get("color", type(None)).dtype)
                if hasattr(cam.get("color"), "dtype")
                else None,
            }
            for name, cam in vision.items()
        },
        "state": _jsonable(state_data),
        "eef": _jsonable(
            {
                "left": state_data.get("left_ee_pose"),
                "right": state_data.get("right_ee_pose"),
            }
        ),
    }


@tool(readonly=True, exclude=("state",))
def back_project(
    state: EnvState,
    row: int,
    col: int,
    camera: Literal["cam_head", "cam_left_wrist", "cam_right_wrist"] = "cam_head",
    step: int | None = None,
) -> ToolResult:
    """Read one pixel's world xyz from the selected camera's saved world map. Use the same camera and step as the source image. Returns world coordinates in metres.

    Args:
        row: Pixel row (0=top)
        col: Pixel col (0=left)
        camera: Camera to project from"""
    view = camera.removeprefix("cam_")
    record, world, error = _load_world_xyz(state, view=view, step=step)
    if error is not None:
        return error
    step_idx = record["step_idx"]
    if type(row) is not int or type(col) is not int:
        return ToolResult(data={"error": "pixel must be integer [row,col]"})
    h, w = world.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        payload = {"error": f"pixel out of range: row 0..{h - 1}, col 0..{w - 1}"}
        return ToolResult(data=payload)
    try:
        d = float(state.load(_artifact_name(view, "depth"), step=step_idx)[row, col])
    except Exception as exc:
        return ToolResult(data={"error": f"saved depth unavailable: {exc}"})
    if not np.isfinite(d) or d <= 0:
        payload = {"error": f"invalid depth at pixel: {d}"}
        return ToolResult(data=payload)
    point = world[row, col]
    if not np.isfinite(point).all():
        return ToolResult(data={"error": f"invalid world xyz at pixel ({row},{col})"})
    payload = {
        "camera": camera,
        "step_idx": step_idx,
        "pixel": [row, col],
        "depth_m": round(d, 4),
        "world_xyz": [round(float(v), 4) for v in point],
    }
    return ToolResult(data=payload)


@tool(readonly=True, exclude=("primitives", "state"))
def segment(
    primitives: Any,
    state: Any,
    text_prompt: str,
    camera: Literal["cam_head", "cam_left_wrist", "cam_right_wrist"] = "cam_head",
    min_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.2,
    step: int | None = None,
) -> ToolResult:
    """Segment a saved camera image with SAM3. Inspect the returned red-mask overlay before using point_rc with back_project at the same camera and step_idx. This tool does not move the robot or compute a grasp pose.

    Args:
        text_prompt: e.g. 'the bottle'
        camera: Camera to segment (default cam_head)
        min_score: Min score (default 0.2)"""
    if not text_prompt.strip() or not np.isfinite(min_score) or not 0 <= min_score <= 1:
        return ToolResult(
            data={"error": "provide nonempty text and finite min_score in [0,1]"}
        )
    cam, step_idx = _recorded_camera(primitives, state, camera, step)
    if cam is None:
        payload = {"error": f"camera {camera!r} not in observation"}
        return ToolResult(data=payload)
    color = cam.get("color")
    if color is None:
        payload = {"error": "camera color image missing"}
        return ToolResult(data=payload)
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        payload = {"error": "sam3_client not configured"}
        return ToolResult(data=payload)
    fallback = "Use a visible pixel with back_project or sample_world_xyz on the same saved frame."
    try:
        result = sam3.segment(color, text_prompt=text_prompt, min_score=min_score)
    except Exception as exc:
        return ToolResult(
            data={
                "error": f"segmentation service call failed: {exc}",
                "camera": camera,
                "step_idx": step_idx,
                "fallback": fallback,
            }
        )
    if not result.found:
        payload = {
            "camera": camera,
            "step_idx": step_idx,
            "found": False,
            "text_prompt": text_prompt,
            "score": result.score,
            "reason": getattr(result, "reason", None) or "SAM3 found no mask",
            "fallback": fallback,
        }
        return ToolResult(data=payload)
    from robots.robodojo.flash.grounding import mask_geometry

    mask = np.asarray(result.mask, dtype=bool)
    if mask.shape != np.asarray(color).shape[:2] or not mask.any():
        return ToolResult(
            data={"error": "SAM3 mask must match the RGB image and contain foreground"}
        )
    geometry = mask_geometry(mask)
    rows, cols = np.nonzero(mask)
    centroid = geometry["centroid_rc"]
    index = np.argmin((rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2)
    point = [int(rows[index]), int(cols[index])]
    payload = {
        "camera": camera,
        "step_idx": step_idx,
        "text_prompt": text_prompt,
        "min_score": min_score,
        "found": True,
        "score": result.score,
        "box_px": _jsonable(result.box),
        **geometry,
        "point_rc": point,
        "point_method": "nearest_foreground_to_centroid",
        "centroid_in_mask": bool(mask[tuple(centroid)]),
    }
    import imageio.v2 as imageio

    overlay = np.asarray(color).copy()
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)
    overlay_bytes = imageio.imwrite("<bytes>", overlay, format="png")
    if state is not None:
        index = primitives.reserve_segment_index()
        overlay_name = f"segment_overlay_{index:02d}.png"
        segment_name = f"segment_{index:02d}.json"
        saved_overlay = state.save(overlay_name, overlay, step=step_idx)
        saved_segment = state.save(segment_name, payload, step=step_idx)
        if saved_overlay is None or saved_segment is None:
            return ToolResult(
                data={
                    **payload,
                    "error": "could not persist segmentation evidence",
                    "fallback": fallback,
                },
                images=[overlay_bytes],
            )
        payload.update(overlay_artifact=saved_overlay, segment_artifact=saved_segment)
    return ToolResult(data=payload, images=[overlay_bytes])


def _recorded_camera(
    primitives, state, camera: str, step: int | None
) -> tuple[dict | None, int | None]:
    """Load a saved frame, never silently render during a read-only query."""
    if state is None:
        # Direct legacy callers may supply their own immutable observation.
        return (primitives._last_obs or {}).get("vision", {}).get(camera), None
    record = state.get(-1 if step is None else step)
    cam = dict(record.state.get("cameras", {}).get(camera, {}))
    name = f"{camera}.png"
    if name in record.artifacts:
        cam["color"] = state.load(name, step=record.step_idx)
    return cam or None, record.step_idx


def _world_from_depth(depth, intrinsic, cam2world) -> np.ndarray:
    """Project optical-axis metres using USD camera axes (+X, +Y up, -Z)."""
    depth = np.asarray(depth, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    cam2world = np.asarray(cam2world, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H,W]")
    if (
        intrinsic.shape != (3, 3)
        or cam2world.shape != (4, 4)
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(cam2world).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
    ):
        raise ValueError(
            "camera calibration must be finite with positive focal lengths"
        )
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, np.nan)
    rows, cols = np.indices(depth.shape)
    points = np.stack(
        (
            (cols - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            -(rows - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            -depth,
        ),
        axis=-1,
    )
    return (points @ cam2world[:3, :3].T + cam2world[:3, 3]).astype(np.float32)


def _refresh_obs(primitives) -> dict:
    primitives._last_obs = primitives.env.get_obs()
    return primitives._last_obs


@tool(
    readonly=False,
    exclude=("primitives", "state", "lift_thresh", "gripper_closed_thresh"),
    json_schema_extra={"required": []},
)
def pi0_pick(
    primitives: Any,
    state: Any,
    prompt: str | SkipJsonSchema[None] = None,
    arm: Literal["left", "right"] = "right",
    max_chunks: int = 8,
    lift_thresh: Any = 0.04,
    gripper_closed_thresh: Any = 0.55,
) -> ToolResult:
    """Closed-loop Pi_05 pick: feed the observation to the Pi_05 policy, apply its action chunk, and detect grasp success by eef lift + gripper closure.

    Args:
        prompt: Optional specific target instruction; omit to use resolved official task language
        max_chunks: Max policy chunks"""
    import numpy as np

    from robots.robodojo.tasks import validate_instruction

    prompt = validate_instruction(
        primitives.env.get_task_language() if prompt is None else prompt
    )

    vla = getattr(primitives, "vla_client", None)
    if vla is None:
        payload = {"error": "vla_client not configured"}
        return ToolResult(data=payload)
    arms = ("left", "right")
    track = {
        a: {
            "start_z": None,
            "min_z": None,
            "peak": None,
            "start_grip": 1.0,
            "last_grip": 1.0,
        }
        for a in arms
    }
    chunks_used = 0
    success = False
    terminated = False
    for c in range(max_chunks):
        primitives._check_cancelled()
        obs = _refresh_obs(primitives)
        obs["instruction"] = prompt
        for a in arms:
            t = track[a]
            if t["start_z"] is None:
                pose = np.asarray(obs["state"][f"{a}_ee_pose"], dtype=np.float64)
                t["start_z"] = float(pose[2])
                t["min_z"] = t["start_z"]
                t["peak"] = t["start_z"]
                t["start_grip"] = float(
                    np.asarray(obs["state"][f"{a}_ee_joint_state"]).reshape(-1)[0]
                )
                t["last_grip"] = t["start_grip"]
        actions = vla.predict(obs)
        chunks_used = c + 1
        for action in actions:
            obs_step, _reward, _done, info = primitives.env.step(action)
            st = obs_step["state"]
            for a in arms:
                t = track[a]
                z = float(np.asarray(st[f"{a}_ee_pose"], dtype=np.float64)[2])
                grip = float(np.asarray(st[f"{a}_ee_joint_state"]).reshape(-1)[0])
                t["last_grip"] = grip
                if z < t["min_z"]:
                    t["min_z"] = z
                    t["peak"] = z
                else:
                    t["peak"] = max(t["peak"], z)
                descended = (t["start_z"] - t["min_z"]) >= 0.06
                if descended:
                    ascended = (t["peak"] - t["min_z"]) >= lift_thresh
                    closed = grip < gripper_closed_thresh
                    if ascended and closed:
                        success = True
                        break
            if success:
                # Stop the chunk at once: the remaining actions were predicted
                # for an ungrasped state and would reopen the gripper, dropping
                # the object while ``success`` stays true.
                break
            if info["status"]["step"] >= info["status"]["step_limit"]:
                terminated = True
                break
        if success or terminated:
            break
    per_arm = {}
    for a in arms:
        t = track[a]
        per_arm[a] = {
            "start_eef_z": round(t["start_z"], 4),
            "min_eef_z": round(t["min_z"], 4),
            "peak_eef_z": round(t["peak"], 4),
            "peak_lift_m": round(t["peak"] - t["min_z"], 4),
            "start_gripper": round(t["start_grip"], 3),
            "final_gripper": round(t["last_grip"], 3),
        }
    payload = {
        "monitored_arm": arm,
        "instruction": prompt,
        "success": success,
        "chunks_used": chunks_used,
        "arms": per_arm,
        "terminated": terminated,
    }
    return ToolResult(data=payload)


def get_reward_details(primitives, state) -> dict:
    """Read the current reward/score breakdown without changing the env."""
    return primitives.env.get_reward_details()


def get_safety_status(primitives, state) -> dict:
    """Read the env safety monitor (rolling / off-table alarms)."""
    return primitives.env.get_safety_status()


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_state(primitives, state, *, log: dict | None = None, perception=None):
    """Record one RoboDojo observation and save its camera artifacts.

    Fetches a live observation, appends a :class:`StepRecord` through
    :meth:`EnvState.record_step`, writes the three camera RGB PNGs for that
    step, and returns the record.
    """
    obs = _refresh_obs(primitives)
    status = primitives.env.get_status()
    log = log or {}
    summary = _summarize_obs(obs)
    step_idx = 0 if state.latest_step is None else state.latest_step + 1
    summary.update(step_idx=step_idx, artifacts={}, view_specs={})
    views = {}
    for camera, cam in obs.get("vision", {}).items():
        if camera not in dict(CAMERA_ARTIFACTS):
            continue
        view = camera.removeprefix("cam_")
        color = cam.get("color")
        if color is None:
            continue
        shape = np.asarray(color).shape[:2]
        summary["view_specs"][view] = {
            "coordinate_space": view,
            "image_shape": list(shape),
            "pixel_order": "row_col",
        }
        summary["artifacts"][view] = {"rgb": f"{camera}.png"}
        depth = cam.get("distance_to_image_plane")
        if depth is None:
            depth = cam.get("depth")
        if depth is not None:
            if np.asarray(depth).shape != shape:
                raise ValueError("RGB and depth must have the same image dimensions")
            world = _world_from_depth(
                depth, cam.get("intrinsic_matrix"), cam.get("extrinsic_matrix")
            )
            views[view] = {
                "depth": np.asarray(depth),
                "world_xyz": world,
                "camera_meta": {
                    "intrinsic_matrix": _jsonable(cam.get("intrinsic_matrix")),
                    "extrinsic_matrix": _jsonable(cam.get("extrinsic_matrix")),
                },
            }
            summary["artifacts"][view].update(
                {field: _artifact_name(view, field) for field in views[view]}
            )
    with state.record_step(
        state=summary,
        # Official success is read by the runner after planner execution,
        # not written into planner-readable state artifacts.
        terminated=False,
        truncated=int(status.get("step", 0)) >= int(status.get("step_limit", 0) or 0),
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": obs.get("instruction")
            or primitives.env.get_task_language(),
            "episode_status": public_episode_budget(status),
            **({"perception": perception} if perception else {}),
        },
    ) as step_idx:
        vision = obs.get("vision", {})
        for camera, artifact in CAMERA_ARTIFACTS:
            color = (vision.get(camera) or {}).get("color")
            if color is not None:
                state.save(artifact, color, step=step_idx)
        for view, fields in views.items():
            for field, value in fields.items():
                state.save(_artifact_name(view, field), value, step=step_idx)
    return state.get(step_idx)
