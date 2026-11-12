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

"""BEHAVIOR agent tools, public observations and camera-to-base geometry.

State inspection, projection, segmentation and run completion share the same
saved observations. Robot actions are declared by ``BehaviorPrimitives``;
``BehaviorToolkit`` registers both groups through RPent's tool interface.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool
from rpent.utils.transforms import transform_points

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Client


# The public camera-pose slots follow evaluator.py::_preprocess_obs / r1pro.yaml.
R1PRO_EVAL_CAMERA_ORDER = ("left_wrist", "right_wrist", "head")
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
_CAMERA_SUFFIXES = {
    "head": "zed_link:Camera:0",
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
}
# Official CAMERA_INTRINSICS: native RGBDFullResWrapper and 224² DefaultWrapper.
_INTRINSICS = {
    "head": (720, 306.0),
    "left_wrist": (480, 388.6639),
    "right_wrist": (480, 388.6639),
}


# Observation parsing and measured robot state.


def extract_public_proprio(
    raw: dict[str, Any], *, task_id: int | None = None
) -> np.ndarray:
    """Read public61 without cameras or array copies.

    Accept one unbatched or single-slot vector with any robot prefix. Check
    a received task ID when ``task_id`` is supplied. Treat the returned view
    as read-only; copy it when retaining independent control-frame history.
    """
    if task_id is not None:
        received_ids = [raw[key] for key in ("task_id", b"task_id") if key in raw]
        if len(received_ids) > 1:
            raise ValueError("duplicate observation key: task_id")
        if received_ids:
            received_id = np.asarray(received_ids[0])
            if received_id.size != 1 or received_id.reshape(-1)[0] != task_id:
                raise ValueError(
                    "observation task_id does not match the configured task"
                )
    matches = [
        value
        for key, value in raw.items()
        if (key.decode() if isinstance(key, bytes) else str(key)).endswith("::proprio")
    ]
    if len(matches) != 1:
        raise ValueError("expected one public sensor ending in ::proprio")
    state = np.asarray(matches[0])
    if state.shape == (1, 61):
        state = state[0]
    if (
        state.shape != (61,)
        or state.dtype.kind not in "fiu"
        or not np.isfinite(state).all()
    ):
        raise ValueError("public R1Pro state must contain one finite 61D proprio")
    return state


def decode_robot_state(proprio: Any) -> dict[str, Any]:
    """Decode the public R1Pro 61D proprio vector without using VLA state padding.

    Positions are metres and joint angles are radians. EEF poses use xyzw
    quaternions in the physical robot base frame at this observation.
    """
    values = np.asarray(proprio, dtype=np.float64)
    if values.shape != (61,) or not np.isfinite(values).all():
        raise ValueError("robot state requires the original finite 61D proprio")
    arms = {}
    for arm, offset in (("left", 3), ("right", 28)):
        quaternion = values[offset + 17 : offset + 21]
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3):
            raise ValueError(f"{arm} public EEF quaternion must be unit xyzw")
        arms[arm] = {
            "arm_qpos": values[offset : offset + 7].tolist(),
            "arm_qvel": values[offset + 7 : offset + 14].tolist(),
            "eef_position": values[offset + 14 : offset + 17].tolist(),
            "eef_quaternion_xyzw": quaternion.tolist(),
            "finger_qpos": values[offset + 21 : offset + 23].tolist(),
            "finger_qvel": values[offset + 23 : offset + 25].tolist(),
        }
    return {
        "frame": "robot_base",
        "position_unit": "metre",
        "joint_unit": "radian",
        "base_qvel": values[:3].tolist(),
        "trunk_qpos": values[53:57].tolist(),
        "trunk_qvel": values[57:61].tolist(),
        **arms,
    }


def build_camera_metadata(
    camera: str,
    image_shape: tuple[int, int],
    pose_xyz_xyzw: Any,
    *,
    pose_source: str,
    depth_type: str | None,
) -> dict[str, Any]:
    """Describe a public evaluator camera without changing RGB/depth arrays.

    Only the shipped native and 224x224 evaluator profiles receive an intrinsic
    matrix. Other resolutions retain their pose with calibration unavailable;
    a crop or a custom camera configuration must not silently reuse these K's.

    Args:
        camera: Canonical R1Pro evaluator camera name.
        image_shape: Stored RGB height and width.
        pose_xyz_xyzw: Seven values from this camera's public cam_rel_poses slot.
        pose_source: Original observation field containing the camera poses.
        depth_type: Public annotator definition, or None for RGB-only data.

    Returns:
        JSON-serializable calibration and robot-base pose metadata.

    Raises:
        ValueError: Camera identity or its xyz+xyzw pose is invalid.
    """
    if camera not in _INTRINSICS:
        raise ValueError(f"unknown R1Pro evaluator camera: {camera}")
    pose = np.asarray(pose_xyz_xyzw, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("camera pose must be finite xyz+xyzw with shape (7,)")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-3):
        raise ValueError("camera pose must contain a unit xyzw quaternion")
    height, width = image_shape
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    metadata = {
        "camera_name": camera,
        "height": height,
        "width": width,
        "pixel_order": "row_col",
        "camera_frame": "opengl",
        "target_frame": "robot_base",
        "camera_to_base_gl": transform.tolist(),
        "pose_xyz_xyzw": pose.tolist(),
        "pose_source": pose_source,
        "pose_index": R1PRO_EVAL_CAMERA_ORDER.index(camera),
        "calibration_source": "omnigibson.eval.utils.eval_utils.CAMERA_INTRINSICS.R1Pro",
        "depth_type": depth_type,
        "depth_unit": "metre" if depth_type is not None else None,
    }
    native_size, native_focal = _INTRINSICS[camera]
    if image_shape not in ((native_size, native_size), (224, 224)):
        metadata["calibration_available"] = False
        return metadata
    scale = width / native_size
    metadata["intrinsic_K"] = [
        [native_focal * scale, 0.0, width / 2.0],
        [0.0, native_focal * scale, height / 2.0],
        [0.0, 0.0, 1.0],
    ]
    metadata["calibration_available"] = True
    return metadata


def prepare_behavior_observation(
    raw: dict[str, Any], *, task_id: int, task_language: str
) -> dict[str, Any]:
    """Validate and normalize one official observation for tools and VLA input.

    Arrays share the input buffers where possible and must be treated as
    read-only. RGB views are made contiguous when their layout requires it.

    Args:
        raw: Flattened sensor fields, unbatched or with a single leading slot.
        task_id: Configured BEHAVIOR task, checked against the received task ID.
        task_language: Explicit task instruction from the run configuration.

    Returns:
        Three RGB views, optional raw depth/public camera metadata, proprioception
        and a filtered, unbatched ``policy_observation`` with original field names.

    Raises:
        ValueError: A sensor is missing, ambiguous or has an unsupported shape.
    """
    values: dict[str, Any] = {}
    for key, value in raw.items():
        name = key.decode() if isinstance(key, bytes) else str(key)
        if name in values:
            raise ValueError(f"duplicate observation key: {name}")
        values[name] = value

    def find_key(suffix: str) -> str:
        matches = [key for key in values if key.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one sensor ending in {suffix!r}: {matches}")
        return matches[0]

    proprio_key = find_key("::proprio")
    proprio = np.asarray(values[proprio_key])
    batched = proprio.ndim == 2
    if batched:
        if proprio.shape[0] != 1:
            raise ValueError("BEHAVIOR supports one environment at a time")
        proprio = proprio[0]
    if proprio.ndim != 1 or proprio.size == 0 or not np.isfinite(proprio).all():
        raise ValueError("proprio must be a nonempty finite vector")

    if "task_id" in values:
        received_id = np.asarray(values["task_id"])
        if received_id.size != 1 or received_id.reshape(-1)[0] != task_id:
            raise ValueError("observation task_id does not match the configured task")

    policy_observation: dict[str, Any] = {proprio_key: proprio}
    pose_key = proprio_key.removesuffix("::proprio") + "::cam_rel_poses"
    camera_poses = None
    if pose_key in values:
        camera_poses = np.asarray(values[pose_key], dtype=np.float64)
        if batched:
            if camera_poses.shape != (1, 21):
                raise ValueError(
                    "cam_rel_poses must match the single-slot [1,21] batch"
                )
            camera_poses = camera_poses[0]
        if camera_poses.shape != (21,) or not np.isfinite(camera_poses).all():
            raise ValueError("cam_rel_poses must contain three finite xyz+xyzw poses")
        camera_poses = camera_poses.reshape(3, 7)
    views: dict[str, dict[str, Any]] = {}
    for name, suffix in _CAMERA_SUFFIXES.items():
        rgb_key = find_key(f"{suffix}::rgb")
        rgb = np.asarray(values[rgb_key])
        if batched:
            if rgb.ndim != 4 or rgb.shape[0] != 1:
                raise ValueError(f"{name} RGB must match the single-slot batch")
            rgb = rgb[0]
        if (
            rgb.ndim != 3
            or rgb.shape[-1] not in (3, 4)
            or min(rgb.shape[:2]) < 1
            or rgb.dtype != np.uint8
        ):
            raise ValueError(f"{name} RGB must be a uint8 HWC RGB/RGBA image")
        policy_observation[rgb_key] = rgb
        views[name] = {"rgb": np.ascontiguousarray(rgb[..., :3])}
        depth_type = None
        for depth_suffix in ("::depth", "::depth_linear"):
            matches = [key for key in values if key.endswith(suffix + depth_suffix)]
            if len(matches) > 1:
                raise ValueError(f"ambiguous {name} depth sensor")
            if not matches:
                continue
            if "depth" in views[name]:
                raise ValueError(f"multiple depth definitions for {name}")
            depth_key = matches[0]
            depth = np.asarray(values[depth_key])
            if batched:
                if depth.ndim < 3 or depth.shape[0] != 1:
                    raise ValueError(f"{name} depth must match the single-slot batch")
                depth = depth[0]
            native_depth = depth
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
                raise ValueError(f"{name} depth dimensions must match RGB")
            views[name]["depth"] = depth
            policy_observation[depth_key] = native_depth
            depth_type = (
                "distance_to_image_plane"
                if depth_suffix == "::depth_linear"
                else "distance_to_camera"
            )
        if camera_poses is not None:
            views[name]["camera_meta"] = build_camera_metadata(
                name,
                tuple(rgb.shape[:2]),
                camera_poses[R1PRO_EVAL_CAMERA_ORDER.index(name)],
                pose_source=pose_key,
                depth_type=depth_type,
            )

    return {
        "task_id": task_id,
        "task_language": task_language,
        "proprio": proprio,
        "views": views,
        "policy_observation": policy_observation,
    }


# Camera pixels and masks in the observed robot-base frame.


def _validate_pixel_to_3d_inputs(
    view: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Validate matching metric RGB-D, calibration and coordinate conventions."""
    metadata = view.get("camera_meta")
    if not isinstance(metadata, dict) or not metadata.get("calibration_available"):
        raise ValueError("this view has no verified public camera calibration")
    if (
        metadata.get("target_frame") != "robot_base"
        or metadata.get("camera_frame") != "opengl"
        or metadata.get("pixel_order") != "row_col"
        or metadata.get("depth_unit") != "metre"
    ):
        raise ValueError(
            "camera metadata must describe metric depth in robot-base geometry"
        )
    if metadata.get("depth_type") not in (
        "distance_to_image_plane",
        "distance_to_camera",
    ):
        raise ValueError("unsupported or missing depth definition")
    depth = np.asarray(view.get("depth"))
    rgb = np.asarray(view.get("rgb"))
    if (
        depth.ndim != 2
        or depth.dtype.kind != "f"
        or rgb.ndim != 3
        or depth.shape != rgb.shape[:2]
        or depth.shape != (metadata.get("height"), metadata.get("width"))
    ):
        raise ValueError(
            "metric floating-point depth, RGB and calibration dimensions must agree"
        )
    intrinsic = np.asarray(metadata.get("intrinsic_K"), dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.isclose(intrinsic[1, 0], 0)
        or not np.allclose(intrinsic[2], [0, 0, 1])
    ):
        raise ValueError("intrinsic_K must be a finite pinhole camera matrix")
    return depth, metadata, intrinsic


def _pixels_to_robot_base_points(
    pixels: np.ndarray,
    depths: np.ndarray,
    metadata: dict[str, Any],
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Project validated [row, col] pixels and matching depths to base XYZ."""
    rays = np.linalg.solve(
        intrinsic, np.column_stack((pixels[:, 1], pixels[:, 0], np.ones(len(pixels)))).T
    ).T
    if metadata["depth_type"] == "distance_to_camera":
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return transform_points(
        metadata.get("camera_to_base_gl"), rays * depths[:, None] * [1.0, -1.0, -1.0]
    )


def locate_pixel_in_robot_base(
    view: dict[str, Any],
    pixel: list[int] | tuple[int, int],
    *,
    neighborhood: int = 0,
) -> dict[str, Any]:
    """Project one [row,col] pixel from its matching RGB-D view into robot base.

    Args:
        view: One snapshot view containing RGB, native metric depth and the
            same observation's public ``camera_meta``.
        pixel: Integer [row,col] in that view's stored RGB resolution.
        neighborhood: Radius for an optional median depth estimate. Zero uses
            the exact pixel. Zero, negative, NaN and infinite depths are invalid.

    Returns:
        The robot-base xyz point in metres, depth sampling details and camera
        identity. This pure calculation neither moves nor queries the robot.

    Raises:
        ValueError: Calibration, depth, or the requested pixel is unavailable.
    """
    depth, metadata, intrinsic = _validate_pixel_to_3d_inputs(view)
    if (
        not isinstance(pixel, (list, tuple))
        or len(pixel) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in pixel
        )
    ):
        raise ValueError("pixel must be an integer [row,col] pair")
    if (
        isinstance(neighborhood, (bool, np.bool_))
        or not isinstance(neighborhood, (int, np.integer))
        or not 0 <= neighborhood <= 32
    ):
        raise ValueError("neighborhood must be an integer radius between 0 and 32")
    row, col = map(int, pixel)
    height, width = depth.shape
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f"pixel [row,col] lies outside this view's {height}x{width} RGB image"
        )
    radius = int(neighborhood)
    region = depth[
        max(0, row - radius) : min(height, row + radius + 1),
        max(0, col - radius) : min(width, col + radius + 1),
    ]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size == 0:
        raise ValueError("no finite positive depth at the requested pixel/neighborhood")
    depth_m = float(np.median(valid))
    point_base = _pixels_to_robot_base_points(
        np.asarray([[row, col]]), np.asarray([depth_m]), metadata, intrinsic
    )[0]
    return {
        "point_robot_base": point_base.tolist(),
        "frame": "robot_base",
        "unit": "metre",
        "camera": metadata.get("camera_name"),
        "pixel": [row, col],
        "pixel_order": "row_col",
        "image_shape": [height, width],
        "depth_m": depth_m,
        "depth_type": metadata["depth_type"],
        "neighborhood": radius,
        "valid_depth_pixels": int(valid.size),
        "depth_sampling": "single_pixel" if radius == 0 else "median_neighborhood",
    }


def estimate_mask_position_in_robot_base(
    view: dict[str, Any], mask: np.ndarray
) -> dict[str, Any]:
    """Return the per-axis median of valid mask points in the observed robot base."""
    depth, metadata, intrinsic = _validate_pixel_to_3d_inputs(view)
    if mask.dtype != np.bool_ or mask.shape != depth.shape:
        raise ValueError("SAM mask, saved RGB and depth must have identical dimensions")
    rows, cols = np.where(mask & np.isfinite(depth) & (depth > 0))
    if not rows.size:
        raise ValueError("no finite positive depth inside the mask")
    points = _pixels_to_robot_base_points(
        np.column_stack((rows, cols)), depth[rows, cols], metadata, intrinsic
    )
    return {
        "point_robot_base": np.median(points, axis=0).tolist(),
        "valid_depth_pixels": int(rows.size),
    }


# Saved observations and read-only tools.


def _summarize_robot_state(proprio: Any, *, step: int) -> dict[str, Any]:
    """Summarize measured R1Pro state for decisions; preserve raw artifacts.

    Omit velocities of joints that the Planner cannot command directly. Display
    values are rounded to 0.1 mm / 0.0001 rad, with six-decimal quaternions;
    motion tools continue to use the original observation at full precision.
    """
    try:
        state = decode_robot_state(proprio)
    except (TypeError, ValueError) as error:
        return {"available": False, "source_step": step, "error": str(error)}

    def rounded(values: list[float], decimals: int = 4) -> list[float]:
        return [round(value, decimals) for value in values]

    return {
        "available": True,
        "source_step": step,
        "frame": "robot_base",
        "units": {
            "eef_position": "metre",
            "eef_quaternion_xyzw": "unit quaternion",
            "arm_qpos": "radian",
            "finger_qpos": "metre per finger",
            # The sourced R1Pro torso_joint1..4 are all revolute joints.
            "trunk_qpos": "radian",
            "base_qvel": ["metre/second", "metre/second", "radian/second"],
        },
        "base_qvel": rounded(state["base_qvel"]),
        "trunk_qpos": rounded(state["trunk_qpos"]),
        **{
            arm: {
                key: rounded(state[arm][key], 6 if "quaternion" in key else 4)
                for key in (
                    "eef_position",
                    "eef_quaternion_xyzw",
                    "arm_qpos",
                    "finger_qpos",
                )
            }
            for arm in ("left", "right")
        },
    }


def save_observation_record(
    snapshot: dict[str, Any],
    *,
    env_state: EnvState,
    log: dict[str, Any] | None,
) -> StepRecord:
    """Save a validated observation; record indexes do not count evaluator actions."""
    step_idx = 0 if env_state.latest_step is None else env_state.latest_step + 1
    artifacts: dict[str, Any] = {
        "proprio": str(env_state.artifact_path("proprio.npy", step=step_idx)),
        "views": {},
    }
    view_specs: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        view = snapshot["views"][name]
        artifacts["views"][name] = {
            field: str(
                env_state.artifact_path(
                    f"{name}_{field}.{'png' if field == 'rgb' else 'npy'}",
                    step=step_idx,
                )
            )
            for field in ("rgb", "depth")
            if field in view
        }
        view_specs[name] = {"image_shape": list(view["rgb"].shape[:2])}
        if "camera_meta" in view:
            view_specs[name]["camera_meta"] = deepcopy(view["camera_meta"])
    state = {
        "step_idx": step_idx,
        "step_kind": "observation_record",
        "task_id": snapshot["task_id"],
        "task_language": snapshot["task_language"],
        "proprio_shape": list(snapshot["proprio"].shape),
        "robot_state": _summarize_robot_state(snapshot["proprio"], step=step_idx),
        "artifacts": artifacts,
        "view_specs": view_specs,
    }
    result = (log or {}).get("result") or {}
    with env_state.record_step(
        state=state,
        terminated=result.get("terminated") is True,
        truncated=result.get("truncated") is True,
        command=(log or {}).get("command"),
        result=(log or {}).get("result"),
        elapsed_s=(log or {}).get("elapsed_s"),
    ) as recorded_step:
        if (
            env_state.save("proprio.npy", snapshot["proprio"], step=recorded_step)
            is None
        ):
            raise RuntimeError("failed to save BEHAVIOR proprio artifact")
        for name in CAMERA_NAMES:
            view = snapshot["views"][name]
            for field in ("rgb", "depth"):
                if field in view:
                    artifact = f"{name}_{field}.{'png' if field == 'rgb' else 'npy'}"
                    if (
                        env_state.save(artifact, view[field], step=recorded_step)
                        is None
                    ):
                        raise RuntimeError(
                            f"failed to save BEHAVIOR artifact: {artifact}"
                        )
    return env_state.get(step_idx)


@tool(readonly=True, exclude=("state",))
def view_env_state(
    step: Annotated[int, Field(ge=-1, json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Inspect the robot, target and last action using saved images and measured state.

    Returns head, left-wrist and right-wrist RGB, compact joint/EEF measurements
    in state.robot_state, and the preceding action's execution feedback. Action
    tools already return this observation; call again to inspect an earlier step.
    Reading a saved record does not move the robot or refresh the cameras.

    Args:
        step: Observation record index; 0 is initial and -1 is latest, not an action count.
    """
    try:
        record = state.get(step)
    except (KeyError, LookupError, ValueError) as error:
        return ToolResult(data={"error": f"state step not available: {error}"})
    images = [
        state.load_bytes(f"{name}_rgb.png", step=record.step_idx)
        for name in CAMERA_NAMES
    ]
    state_data = dict(record.state)
    if "robot_state" not in state_data:
        # Historical records retain their original stored format. Add the same
        # read-only view without rewriting them or requiring another tool call.
        try:
            state_data["robot_state"] = _summarize_robot_state(
                state.load("proprio.npy", step=record.step_idx), step=record.step_idx
            )
        except (LookupError, OSError, TypeError, ValueError) as error:
            state_data["robot_state"] = {
                "available": False,
                "source_step": record.step_idx,
                "error": str(error),
            }
    return ToolResult(
        data={
            "step": record.step_idx,
            "step_kind": "observation_record",
            "state": state_data,
            "log": {
                "command": record.command,
                "result": record.result,
                "elapsed_s": record.elapsed_s,
            },
        },
        images=images,
    )


@tool(readonly=True, exclude=("state",))
def read_robot_state(
    step: Annotated[int, Field(ge=-1, json_schema_extra={"default": -1})] = -1,
    *,
    state: EnvState,
) -> ToolResult:
    """Inspect full measured joints, velocities and EEF poses in a saved observation.

    The usual observation already includes compact robot state. This historical
    reader also returns joint velocities and unrounded values without moving.

    Args:
        step: Observation record index; -1 selects the latest saved record.
    """
    try:
        record = state.get(step)
        robot_state = decode_robot_state(
            state.load("proprio.npy", step=record.step_idx)
        )
    except (KeyError, LookupError, OSError, TypeError, ValueError) as error:
        return ToolResult(data={"error": f"robot state unavailable: {error}"})
    return ToolResult(
        data={"step": record.step_idx, "robot_state": robot_state, "readonly": True}
    )


@tool(readonly=True, exclude=("state",))
def pixel_to_robot_base(
    camera: Literal["head", "left_wrist", "right_wrist"],
    pixel: Annotated[list[int], Field(min_length=2, max_length=2)],
    step: Annotated[int, Field(ge=-1, json_schema_extra={"default": -1})] = -1,
    neighborhood: Annotated[
        int, Field(ge=0, le=32, json_schema_extra={"default": 0})
    ] = 0,
    *,
    state: EnvState,
) -> ToolResult:
    """Locate a visible surface in metres from its saved RGB pixel and matching depth.

    This reads matching RGB, depth and camera metadata from one saved step.
    Use this when you can identify a target pixel directly. If segment is
    available, use it first when the target boundary is ambiguous. A surface
    point is not a grasp pose.
    It neither queries the current cameras nor moves the robot. Historical
    coordinates refer to the robot base at that step, not its current pose.

    Args:
        camera: The exact camera whose saved RGB supplied the pixel.
        pixel: Integer [row,col] in the original saved RGB resolution, not a resized preview.
        step: Observation record index; -1 selects latest. Use the step that supplied the pixel.
        neighborhood: Radius for median positive depth; 0 uses only the selected pixel.
    """
    try:
        record = state.get(step)
        view_spec = record.state.get("view_specs", {}).get(camera, {})
        metadata = view_spec.get("camera_meta")
        if not isinstance(metadata, dict) or not metadata.get("calibration_available"):
            raise ValueError(
                "this observation has no verified public camera calibration"
            )
        rgb_name, depth_name = f"{camera}_rgb.png", f"{camera}_depth.npy"
        if rgb_name not in record.artifacts or depth_name not in record.artifacts:
            raise ValueError(
                "matching RGB and depth are not available at this observation"
            )
        view = {
            "rgb": state.load(rgb_name, step=record.step_idx),
            "depth": state.load(depth_name, step=record.step_idx),
            "camera_meta": metadata,
        }
        projected = locate_pixel_in_robot_base(view, pixel, neighborhood=neighborhood)
    except (KeyError, LookupError, OSError, TypeError, ValueError) as error:
        return ToolResult(
            data={
                "error": str(error),
                "code": "geometry_unavailable",
                "camera": camera,
                "requested_step": step,
            }
        )
    return ToolResult(
        data={
            **projected,
            "step": record.step_idx,
            "coordinate_frame_step": record.step_idx,
            "source_artifacts": {
                "rgb": str(state.artifact_path(rgb_name, step=record.step_idx)),
                "depth": str(state.artifact_path(depth_name, step=record.step_idx)),
            },
        }
    )


class SavedImageSegmenter:
    """Run native SAM3 segmentation and save artifacts without advancing state.

    Each toolkit owns one instance. Calls may run concurrently; artifact indices
    are reserved after inference and are not reused when persistence fails.
    """

    def __init__(self, client: Sam3Client) -> None:
        self._client = client
        self._index_lock = threading.Lock()
        self._index = 0

    def segment(
        self,
        *,
        state: EnvState,
        step: int,
        camera: str,
        image_artifact: str,
        prompt: str = "",
        point: list[int] | None = None,
        min_score: float = 0.2,
        project_mask: Callable[[np.ndarray | None, str | None], dict[str, Any]]
        | None = None,
        fallback: str = "Use manual visual localization and back_project.",
    ) -> ToolResult:
        """Segment one saved RGB image and attach results to its source step.

        Args:
            state: Session artifact store; no environment access is performed.
            step: Concrete source step already resolved by the robot adapter.
            camera: Camera label recorded in the result.
            image_artifact: RGB artifact selected by the robot adapter.
            prompt: Text prompt, mutually exclusive with point.
            point: One positive pixel as [row, col].
            min_score: Minimum accepted SAM3 score.
            project_mask: Robot-specific projection of a mask, or of None and
                the native SAM3 reason when no mask was found. The returned fields
                are included in both the saved artifact and tool response.
            fallback: Existing tool guidance used for service/persistence errors.

        Returns:
            Native segmentation fields, projection fields, saved artifact names,
            and the overlay image when its save succeeds.
        """
        prompt = prompt.strip()
        has_prompt = bool(prompt)
        if has_prompt == (point is not None):
            return ToolResult(
                data={"error": "segment needs exactly one of prompt or point"}
            )
        source = {
            "step": step,
            "camera": camera,
            "image_artifact": image_artifact,
        }
        try:
            data = self._client.segment(
                state.load_bytes(image_artifact, step=step),
                text_prompt=prompt if has_prompt else None,
                point=point,
                min_score=min_score,
            )
        except ValueError as exc:
            return ToolResult(data={"error": str(exc), **source})
        except Exception as exc:
            return ToolResult(
                data={
                    "error": f"segmentation service call failed: {exc}",
                    **source,
                    "fallback": fallback,
                }
            )

        # Preserve the native reservation order; failed saves leave gaps.
        with self._index_lock:
            segment_index = self._index
            self._index += 1
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        mask = data.mask if data.found and isinstance(data.mask, np.ndarray) else None
        projection = project_mask(mask, data.reason) if project_mask is not None else {}
        saved_overlay = None
        if mask is not None:
            overlay = _make_segment_overlay(state.load(image_artifact, step=step), mask)
            if overlay is not None and state.save(overlay_name, overlay, step=step):
                saved_overlay = overlay_name

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": step,
            "segment_index": segment_index,
            "image_artifact": image_artifact,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(projection)
        saved_segment = state.save(segment_name, segment_blob, step=step)

        result = {
            "found": data.found,
            **source,
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            **projection,
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = fallback
        else:
            result["segment_artifact"] = saved_segment
            if "error" in segment_blob:
                result["error"] = segment_blob["error"]
                result["fallback"] = fallback
        images: list[bytes] = []
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            images.append(state.load_bytes(saved_overlay, step=step))
        return ToolResult(data=result, images=images)


def _make_segment_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay


@tool(readonly=True, exclude=("state", "segmenter"))
def segment(
    prompt: str = "",
    camera: Literal["head", "left_wrist", "right_wrist"] = "head",
    step: Annotated[int, Field(ge=-1, json_schema_extra={"default": -1})] = -1,
    point: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None,
    min_score: Annotated[float, Field(ge=0, le=1)] = 0.2,
    *,
    state: EnvState,
    segmenter: SavedImageSegmenter,
) -> ToolResult:
    """Isolate a visible object or part with SAM3 in a saved camera image.

    Provide one text prompt or positive point to isolate a visible object or part.
    Inspect the returned mask overlay before using its geometry. Matching depth
    gives a robot-base point summary at that saved step, not a grasp pose.
    This reads the saved view without rendering or moving the robot.

    Args:
        prompt: Object/text prompt to segment; prefer a short identifying phrase.
        camera: Saved RGB camera containing the target.
        step: Saved observation record; -1 selects latest.
        point: Positive [row,col] in the original image, mutually exclusive with prompt.
        min_score: Minimum accepted mask score, not task-success confidence.
    """
    try:
        record = state.get(step)
        rgb_name, depth_name = f"{camera}_rgb.png", f"{camera}_depth.npy"
        if rgb_name not in record.artifacts:
            raise ValueError("requested camera RGB is not in the saved observation")
    except (LookupError, ValueError) as error:
        return ToolResult(data={"error": str(error)})

    def project_mask(mask: np.ndarray | None, reason: str | None) -> dict[str, Any]:
        result = {
            "geometry_valid": False,
            "point_robot_base": None,
            "geometry_error": reason,
            "frame": "robot_base",
            "unit": "metre",
            "coordinate_frame_step": record.step_idx,
            "point_semantics": "mask point median, not a contact or grasp pose",
        }
        if mask is not None:
            try:
                result.update(
                    estimate_mask_position_in_robot_base(
                        {
                            "rgb": state.load(rgb_name, step=record.step_idx),
                            "depth": state.load(depth_name, step=record.step_idx),
                            "camera_meta": record.state.get("view_specs", {})
                            .get(camera, {})
                            .get("camera_meta"),
                        },
                        mask,
                    )
                )
                result.update(geometry_valid=True, geometry_error=None)
            except (LookupError, OSError, TypeError, ValueError) as error:
                result["geometry_error"] = str(error)
        return result

    return segmenter.segment(
        state=state,
        step=record.step_idx,
        camera=camera,
        image_artifact=rgb_name,
        prompt=prompt,
        point=point,
        min_score=min_score,
        project_mask=project_mask,
        fallback=(
            "Inspect view_env_state. If pixel_to_robot_base is available, use it "
            "on the matching image for a visible target pixel."
        ),
    )


@tool(readonly=True)
def finish(
    *,
    status: Literal["success", "failure", "blocked"],
    summary: str,
) -> ToolResult:
    """End the agent run; official task success is not available to this adapter."""
    return ToolResult(
        data={
            "_finish": True,
            "status": "unverified" if status == "success" else status,
            "requested_status": status,
            "summary": summary,
            "success": None,
        }
    )
