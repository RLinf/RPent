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

"""Dual-Franka RGBD perception helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rotation

from robots.franka.perception import _resolve_step
from robots.franka.runtime_config import get_calibration_path, load_mapping
from rpent.session import EnvState, StepRecord
from rpent.tools.toolkit import readonly

ROBOT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "example.yaml"


class DualFrankaPerceptionError(ValueError):
    """Raised when a dual-Franka perception artifact is missing or invalid."""


@readonly
def back_project(
    *,
    # PhysicalAgent alignment note: D455 is the deployed clean-desk primary
    # metric view.  The implementation accepts any camera registered in
    # perception.projection_views; this default should become config-driven
    # before treating dual_franka as a portable robot extension.
    camera: str = "d455",
    row: int,
    col: int,
    target_name: str = "target",
    step: int | None = None,
    window_radius: int = 2,
    state: EnvState | None = None,
) -> dict[str, Any]:
    """Back-project one registered camera pixel into the shared right_base frame."""
    return _back_project_camera_pixel(
        camera=camera,
        row=row,
        col=col,
        target_name=target_name,
        step=step,
        window_radius=window_radius,
        state=state,
    )


@readonly
def segment(
    *,
    # Same deployment default as back_project: SAM3 can run on any registered
    # RGBD projection view, but D455 is the live clean-desk reference camera
    # used for log alignment with PhysicalAgent.
    camera: str = "d455",
    prompt: str = "",
    point: list[int] | None = None,
    target_name: str = "target",
    step: int | None = None,
    min_score: float = 0.2,
    min_valid_depth_pixels: int = 25,
    state: EnvState | None = None,
    sam3_client: Any | None = None,
) -> dict[str, Any]:
    """Segment one registered RGBD camera with SAM3 and localize in right_base."""
    camera = _resolve_projection_camera_alias(camera)
    if state is None:
        return {"ok": False, "found": False, "error": "state is required"}
    if sam3_client is None:
        return {
            "ok": False,
            "found": False,
            "error": (
                "SAM3 client is not configured. Start RPent with --sam3-endpoint "
                "or set SAM3_CHECKPOINT_PATH for local SAM3 auto-start."
            ),
            "fallback": f"Use manual {camera} image inspection and back_project.",
        }

    text_prompt = prompt.strip()
    has_prompt = bool(text_prompt)
    has_point = point is not None
    if has_prompt == has_point:
        return {
            "ok": False,
            "found": False,
            "error": "segment needs exactly one of prompt or point",
        }
    if has_point:
        if not isinstance(point, list) or len(point) != 2:
            return {
                "ok": False,
                "found": False,
                "error": "point must be [row, col]",
            }
        point = [int(point[0]), int(point[1])]

    try:
        step_idx, _record_state = _resolve_step(state, step)
        image_name = f"{camera}.png"
        depth_name = f"{camera}_depth.npy"
        if not state.exists(image_name, step=step_idx):
            raise DualFrankaPerceptionError(f"{camera} image artifact is missing")
        if not state.exists(depth_name, step=step_idx):
            raise DualFrankaPerceptionError(f"{camera} depth artifact is missing")
        image_bytes = state.load_bytes(image_name, step=step_idx)
        data = sam3_client.segment(
            image_bytes,
            text_prompt=text_prompt if has_prompt else None,
            point=point,
            min_score=float(min_score),
        )
    except DualFrankaPerceptionError as exc:
        return {"ok": False, "found": False, "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "found": False, "error": str(exc)}
    except Exception as exc:
        return {
            "ok": False,
            "found": False,
            "error": f"segmentation service call failed: {exc}",
            "fallback": f"Use manual {camera} image inspection and back_project.",
        }

    segment_index = _next_named_artifact_index(
        state.get(step_idx), prefix=f"{camera}_segment", suffix=".json"
    )
    segment_name = f"{camera}_segment_{segment_index:02d}.json"
    overlay_name = f"{camera}_segment_overlay_{segment_index:02d}.png"
    mode = "text" if has_prompt else "point"

    localization: dict[str, Any]
    mask = data.mask
    saved_overlay = None
    if data.found and isinstance(mask, np.ndarray):
        try:
            depth = np.asarray(np.load(state.artifact_path(depth_name, step=step_idx)))
            localization = _mask_to_camera_world(
                mask,
                depth.squeeze(),
                camera=camera,
                state=state,
                step_idx=step_idx,
                min_valid=max(1, int(min_valid_depth_pixels)),
            )
            overlay = _make_segment_overlay(
                state.load(image_name, step=step_idx),
                mask,
                localization=localization,
            )
            if overlay is not None and state.save(overlay_name, overlay, step=step_idx):
                saved_overlay = overlay_name
        except Exception as exc:
            localization = {
                "point_xyz": None,
                "world_error": f"{type(exc).__name__}: {exc}",
            }
    else:
        localization = {
            "point_xyz": None,
            "world_error": data.reason or "SAM3 found no mask",
        }

    segment_blob = {
        "ok": bool(data.found and localization.get("point_xyz") is not None),
        "found": bool(data.found),
        "mode": mode,
        "target_name": str(target_name).strip() or "target",
        "camera": camera,
        "source_step": step_idx,
        "segment_index": segment_index,
        "image_artifact": image_name,
        "depth_artifact": depth_name,
        "min_score": float(min_score),
        "score": round(float(data.score), 3) if data.score is not None else None,
        "box": data.box,
        "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        "coordinate_frame": "right_base",
        "coordinate_contract": (
            f"point_xyz is the median valid {camera}-mask point expressed in the "
            "shared right_base world frame."
        ),
        "selection_contract": (
            f"Inspect the returned {camera} mask overlay. The highlighted mask and "
            "median marker must cover the intended visible material, not the rim, "
            "wire basket, wall, table, or background. Retry with a point prompt "
            "or a more specific text prompt if it is wrong."
        ),
    }
    if has_prompt:
        segment_blob["prompt"] = text_prompt
    else:
        segment_blob["point"] = point
    if not data.found:
        segment_blob["error"] = data.reason or "SAM3 found no mask"
    segment_blob.update(localization)

    saved_segment = state.save(segment_name, segment_blob, step=step_idx)
    result = {
        "ok": segment_blob["ok"],
        "found": segment_blob["found"],
        "step": step_idx,
        "camera": camera,
        "target_name": segment_blob["target_name"],
        "mode": mode,
        "score": segment_blob["score"],
        "box": segment_blob["box"],
        "mask_shape": segment_blob["mask_shape"],
        "coordinate_frame": "right_base",
        "point_xyz": segment_blob.get("point_xyz"),
        "world_error": segment_blob.get("world_error"),
        "centroid_pixel": segment_blob.get("centroid_pixel"),
        "mask_pixels": segment_blob.get("mask_pixels"),
        "valid_depth_pixels": segment_blob.get("valid_depth_pixels"),
        "valid_localization_pixels": segment_blob.get("valid_localization_pixels"),
        "selection_valid": segment_blob.get("selection_valid"),
        "rejection_reasons": segment_blob.get("rejection_reasons"),
        "left_tcp_xyz": segment_blob.get("left_tcp_xyz"),
        "right_tcp_xyz": segment_blob.get("right_tcp_xyz"),
        "delta_left_tcp_to_point_xyz": segment_blob.get(
            "delta_left_tcp_to_point_xyz"
        ),
        "delta_right_tcp_to_point_xyz": segment_blob.get(
            "delta_right_tcp_to_point_xyz"
        ),
        "tcp_delta_coordinate_frame": segment_blob.get(
            "tcp_delta_coordinate_frame"
        ),
        "tcp_delta_contract": segment_blob.get("tcp_delta_contract"),
        "selection_contract": segment_blob["selection_contract"],
    }
    if saved_segment is None:
        result["error"] = f"failed to persist segment artifact {segment_name}"
    else:
        result["segment_artifact"] = saved_segment
    if saved_overlay is not None:
        result["overlay_artifact"] = saved_overlay
        result["_image_cam_bytes"] = state.load_bytes(saved_overlay, step=step_idx)
        result["image_block_order"] = [f"{camera}_segment_overlay"]
        result["image_delivery"] = (
            f"sam3_{camera}_segment_overlay_returned_for_verification"
        )
    if segment_blob.get("error"):
        result["error"] = segment_blob["error"]
        result["fallback"] = f"Use manual {camera} image inspection and back_project."
    return result


def _back_project_camera_pixel(
    *,
    camera: str,
    row: int,
    col: int,
    target_name: str,
    step: int | None,
    window_radius: int,
    state: EnvState | None = None,
) -> dict[str, Any]:
    camera = _resolve_projection_camera_alias(camera)
    if state is None:
        raise DualFrankaPerceptionError("state is required")
    step_idx, record_state = _resolve_step(state, step)
    projection_cameras = _projection_cameras_for_state(state, step_idx)
    camera_config = projection_cameras.get(camera)
    if camera_config is None:
        known = ", ".join(sorted(projection_cameras)) or "<none>"
        raise DualFrankaPerceptionError(
            f"unsupported projection camera: {camera!r}; registered={known}"
        )
    depth_name = f"{camera}_depth.npy"
    if not state.exists(depth_name, step=step_idx):
        raise DualFrankaPerceptionError(
            f"{camera_config['display_name']} depth artifact is missing. "
            "Restart the env server with the camera and depth enabled, then "
            "call view_driver_state/reset again."
        )
    depth_path = state.artifact_path(depth_name, step=step_idx)
    depth = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise DualFrankaPerceptionError(
            f"expected 2D {camera} depth, got {depth.shape}"
        )
    r = int(row)
    c = int(col)
    if not (0 <= r < depth.shape[0] and 0 <= c < depth.shape[1]):
        raise DualFrankaPerceptionError(
            f"pixel row/col {[r, c]} out of depth bounds {list(depth.shape)}"
        )

    z, valid_pixels = _median_depth(depth, r, c, radius=max(0, int(window_radius)))
    meta = _camera_meta(
        state,
        step_idx,
        camera_alias=camera,
        raw_key=str(camera_config["raw_key"]),
    )
    intr = meta.get("color_intrinsics") or {}
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr.get("ppx", intr.get("cx")))
    cy = float(intr.get("ppy", intr.get("cy")))
    point_camera = np.array([(c - cx) * z / fx, (r - cy) * z / fy, z], dtype=np.float64)

    calibration = load_calibration_bundle()
    calibration_key = str(camera_config["calibration_key"])
    camera_calibration = calibration.get(calibration_key)
    if not isinstance(camera_calibration, dict):
        raise DualFrankaPerceptionError(
            f"calibration entry {calibration_key!r} is missing"
        )
    t_right_camera = _transform_to_matrix(camera_calibration["transformation"])
    point_right = _transform_point(t_right_camera, point_camera)
    selection_valid, rejection_reasons, validity_contract = (
        _validate_localization_point(
            camera_calibration=camera_calibration,
            depth_m=z,
            point_right=point_right,
        )
    )

    out = {
        "ok": selection_valid,
        "selection_valid": selection_valid,
        "target_name": str(target_name).strip() or "target",
        "camera": camera,
        "pixel": [r, c],
        "coordinate_frame": "right_base",
        "step": step_idx,
        "depth_m": round(float(z), 5),
        "depth_window_radius": int(window_radius),
        "valid_depth_pixels_in_window": int(valid_pixels),
        "point_camera_xyz": _round(point_camera),
        "point_xyz": _round(point_right),
        "camera_extrinsic_frame": "right_base",
        "coordinate_contract": (
            "All returned points and deltas are expressed in the shared "
            "right_base world frame. Use the same delta convention for both "
            "left and right rule-based arm tools."
        ),
        "source_artifact": str(depth_path),
        "selection_contract": (
            "The selected RGB pixel must lie well inside visible material of "
            "the named target object. Never select image-space air/background "
            "above the object. Compute robot z approach offsets only for "
            "explicit grasp/approach poses after projecting the object surface "
            "into right_base. For placement staging, use projected x/y only "
            "and keep the carried-object TCP z unchanged by default."
        ),
        "validity_contract": validity_contract,
    }
    if rejection_reasons:
        out["error"] = (
            "Rejected localization point: "
            + "; ".join(rejection_reasons)
            + ". Select a new pixel well inside the visible target surface."
        )
        out["rejection_reasons"] = rejection_reasons
    out.update(_tcp_delta_fields(record_state, point_right, calibration=calibration))
    try:
        out["diagnostic_artifacts"] = _save_back_project_diagnostic(
            state=state,
            step_idx=step_idx,
            depth=depth,
            projection=out,
            transform_right_camera=t_right_camera,
            camera_alias=camera,
            calibration_key=calibration_key,
        )
        annotated_path = out["diagnostic_artifacts"].get("annotated_image")
        if annotated_path:
            out["_image_cam_bytes"] = Path(annotated_path).read_bytes()
            out["image_block_order"] = [f"{camera}_selection_diagnostic"]
            out["image_delivery"] = (
                f"annotated_{camera}_selection_returned_for_verification"
            )
    except Exception as exc:
        out["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _load_perception_config() -> dict[str, Any]:
    """Load the RPent perception section from ``example.yaml``.

    Holds the machine config ``easy_handeye`` does not produce: the tabletop
    ``localization_validity`` bounds and the inter-base ``base_frames``.
    """
    raw = load_mapping(ROBOT_CONFIG_PATH)
    perception = raw.get("perception")
    if not isinstance(perception, dict):
        raise DualFrankaPerceptionError("example.yaml missing the 'perception' section")
    return perception


def _projection_cameras() -> dict[str, dict[str, Any]]:
    perception = _load_perception_config()
    configured = perception.get("projection_views") or {}
    return _coerce_projection_views(configured)


def _projection_cameras_for_state(
    state: EnvState,
    step_idx: int,
) -> dict[str, dict[str, Any]]:
    if state.exists("camera_meta.json", step=step_idx):
        meta = state.load("camera_meta.json", step=step_idx)
        if isinstance(meta, dict) and isinstance(meta.get("projection_views"), dict):
            return _coerce_projection_views(meta["projection_views"])
    return _projection_cameras()


def _coerce_projection_views(configured: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(configured, dict):
        raise DualFrankaPerceptionError("perception.projection_views must be a mapping")
    cameras: dict[str, dict[str, Any]] = {}
    for alias, raw_config in configured.items():
        if not isinstance(raw_config, dict):
            raise DualFrankaPerceptionError(
                f"perception.projection_views.{alias} must be a mapping"
            )
        alias_str = str(alias)
        cameras[alias_str] = {
            "raw_key": str(raw_config.get("raw_key", f"{alias_str}_rgb")),
            "calibration_key": str(
                raw_config.get("calibration_key", f"{alias_str}_camera")
            ),
            "display_name": str(raw_config.get("display_name", alias_str)),
        }
    return cameras


def _resolve_projection_camera_alias(camera: str) -> str:
    alias = str(camera or "").strip()
    if not alias:
        raise DualFrankaPerceptionError("camera must be a non-empty string")
    return alias


def load_calibration_bundle(path: str | Path | None = None) -> dict[str, Any]:
    """Load the dual-Franka perception calibration.

    Merges the ``easy_handeye`` hand-eye transforms (``hand_eye_calibration.json``)
    with the RPent perception config (localization validity + base frames) from
    ``example.yaml``, keeping the historical consumer shape:
    ``<camera>.transformation``, ``<camera>.localization_validity``, and
    ``base_frames``.
    """
    bundle_path = Path(path or get_calibration_path())
    data = json.loads(bundle_path.read_text(errors="replace"))
    if not isinstance(data, dict):
        raise DualFrankaPerceptionError(f"invalid calibration bundle: {bundle_path}")
    perception = _load_perception_config()
    bundle = dict(data)
    bundle["base_frames"] = perception.get("base_frames") or {}
    for camera_key, validity in (perception.get("localization_validity") or {}).items():
        if camera_key in bundle and isinstance(bundle[camera_key], dict):
            bundle[camera_key] = {
                **bundle[camera_key],
                "localization_validity": validity,
            }
    return bundle


def transform_point_between_base_frames(
    point: Any,
    *,
    target: str,
    source: str,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    point_arr = np.asarray(point, dtype=np.float64).reshape(3)
    if target == source:
        return point_arr
    t_target_source = _base_frame_transform(
        calibration or load_calibration_bundle(),
        target=target,
        source=source,
    )
    return _transform_point(t_target_source, point_arr)


def transform_pose_between_base_frames(
    pose: Any,
    *,
    target: str,
    source: str,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    """Transform an xyz+xyzw TCP pose between robot base frames."""
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose_arr.size < 7:
        raise DualFrankaPerceptionError(
            f"expected xyz+quat pose with at least 7 values, got {pose_arr.shape}"
        )
    pose_arr = pose_arr[:7].copy()
    if target == source:
        return pose_arr
    t_target_source = _base_frame_transform(
        calibration or load_calibration_bundle(),
        target=target,
        source=source,
    )
    t_source_tcp = np.eye(4, dtype=np.float64)
    t_source_tcp[:3, 3] = pose_arr[:3]
    quat = pose_arr[3:7] / np.linalg.norm(pose_arr[3:7])
    t_source_tcp[:3, :3] = Rotation.from_quat(quat).as_matrix()
    t_target_tcp = t_target_source @ t_source_tcp
    return np.concatenate(
        [
            t_target_tcp[:3, 3],
            Rotation.from_matrix(t_target_tcp[:3, :3]).as_quat(),
        ]
    )


def _record_arm_state(record_state: dict[str, Any], arm: str) -> dict[str, Any]:
    if not isinstance(record_state, dict):
        return {}
    state_blob = record_state.get("state")
    if isinstance(state_blob, dict):
        record_state = state_blob
    raw = record_state.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get(arm), dict):
        return raw[arm]
    value = record_state.get(f"{arm}_arm")
    return value if isinstance(value, dict) else {}


def _record_tcp_frame(
    record_state: dict[str, Any], arm: str, arm_state: dict[str, Any]
) -> str:
    frame = arm_state.get("tcp_pose_frame") or arm_state.get("coordinate_frame")
    if isinstance(frame, str) and frame:
        return frame
    state_frame = record_state.get("coordinate_frame") if isinstance(record_state, dict) else None
    if isinstance(state_frame, str) and state_frame:
        return state_frame
    return "right_base" if arm == "right" else "left_base"


def _tcp_xyz_in_right_base(
    record_state: dict[str, Any],
    arm: str,
    *,
    calibration: dict[str, Any],
) -> np.ndarray | None:
    arm_state = _record_arm_state(record_state, arm)
    pose = arm_state.get("tcp_pose")
    xyz = _tcp_xyz(pose)
    if xyz is None:
        return None
    source = _record_tcp_frame(record_state, arm, arm_state)
    if source == "right_base":
        return xyz
    return transform_point_between_base_frames(
        xyz,
        target="right_base",
        source=source,
        calibration=calibration,
    )


def _tcp_delta_fields(
    record_state: dict[str, Any],
    point_right: np.ndarray,
    *,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tcp_delta_coordinate_frame": "right_base",
        "tcp_delta_contract": (
            "left_tcp_xyz, right_tcp_xyz, and both TCP-to-point deltas are "
            "expressed in the shared right_base world frame."
        ),
    }
    for arm in ("left", "right"):
        tcp = _tcp_xyz_in_right_base(record_state, arm, calibration=calibration)
        if tcp is None:
            continue
        out[f"{arm}_tcp_xyz"] = _round(tcp)
        out[f"delta_{arm}_tcp_to_point_xyz"] = _round(point_right - tcp)
    return out


def _camera_meta(
    state: EnvState,
    step_idx: int,
    *,
    camera_alias: str,
    raw_key: str,
) -> dict[str, Any]:
    if not state.exists("camera_meta.json", step=step_idx):
        raise DualFrankaPerceptionError(f"{camera_alias} camera metadata not found")
    meta = state.load("camera_meta.json", step=step_idx)
    aliases = [raw_key, camera_alias]
    if camera_alias == "base":
        aliases.append("extra_0")
    for key in aliases:
        value = meta.get(key)
        if isinstance(value, dict) and value.get("color_intrinsics"):
            return value
    raise DualFrankaPerceptionError(
        f"{camera_alias} RealSense color intrinsics not found"
    )


def _validate_localization_point(
    *,
    camera_calibration: dict[str, Any],
    depth_m: float,
    point_right: np.ndarray,
) -> tuple[bool, list[str], dict[str, Any]]:
    config = camera_calibration.get("localization_validity") or {}
    depth_bounds = np.asarray(
        config.get("depth_m", [0.15, 1.25]),
        dtype=np.float64,
    ).reshape(2)
    xyz_min = np.asarray(
        config.get("right_base_xyz_min", [0.10, -0.85, 0.00]),
        dtype=np.float64,
    ).reshape(3)
    xyz_max = np.asarray(
        config.get("right_base_xyz_max", [1.15, 0.85, 0.85]),
        dtype=np.float64,
    ).reshape(3)

    reasons: list[str] = []
    if not depth_bounds[0] <= float(depth_m) <= depth_bounds[1]:
        reasons.append(
            f"depth {float(depth_m):.3f}m is outside configured target range "
            f"[{depth_bounds[0]:.3f}, {depth_bounds[1]:.3f}]m"
        )
    outside_axes = [
        axis
        for axis, value, lower, upper in zip(
            "xyz",
            point_right,
            xyz_min,
            xyz_max,
            strict=True,
        )
        if not lower <= value <= upper
    ]
    if outside_axes:
        reasons.append(
            "right_base point is outside the configured tabletop localization "
            f"volume on axis/axes {','.join(outside_axes)}"
        )

    contract = {
        "depth_m": _round(depth_bounds),
        "right_base_xyz_min": _round(xyz_min),
        "right_base_xyz_max": _round(xyz_max),
    }
    return not reasons, reasons, contract


def _save_back_project_diagnostic(
    *,
    state: EnvState,
    step_idx: int,
    depth: np.ndarray,
    projection: dict[str, Any],
    transform_right_camera: np.ndarray,
    camera_alias: str,
    calibration_key: str,
) -> dict[str, str]:
    """Persist a marked camera image and JSON report for one projection call."""
    artifact_index = _next_named_artifact_index(
        state.get(step_idx), prefix=f"{camera_alias}_back_project", suffix=".json"
    )
    image_name = f"{camera_alias}.png"
    if not state.exists(image_name, step=step_idx):
        raise DualFrankaPerceptionError(f"{camera_alias} image artifact is missing")
    image_path = state.artifact_path(image_name, step=step_idx)

    pixel = projection.get("pixel") or []
    if len(pixel) != 2:
        raise DualFrankaPerceptionError(f"invalid projection pixel: {pixel!r}")
    row, col = int(pixel[0]), int(pixel[1])
    radius = int(projection.get("depth_window_radius") or 0)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x = max(0, min(width - 1, col))
    y = max(0, min(height - 1, row))
    color = (0, 255, 0) if projection.get("selection_valid", True) else (255, 0, 0)

    draw.line((x - 15, y, x + 15, y), fill=color, width=2)
    draw.line((x, y - 15, x, y + 15), fill=color, width=2)
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=2)
    label = (
        f"r{row},c{col} cam={_short_xyz(projection.get('point_camera_xyz'))} "
        f"rb={_short_xyz(projection.get('point_xyz'))}"
    )
    label_x = min(col + 14, max(0, width - 390))
    label_y = max(20, row - 14)
    draw.text((label_x, label_y), label, fill=color)

    annotated_artifact = f"{camera_alias}_back_project_{artifact_index:02d}_annotated.png"
    annotated_name = state.save(
        annotated_artifact, np.asarray(image), step=step_idx
    )
    if annotated_name is None:
        raise DualFrankaPerceptionError("failed to save annotated image")
    annotated_path = state.artifact_path(annotated_name, step=step_idx)

    report = {
        "ok": bool(projection.get("ok")),
        "snapshot_step": step_idx,
        "projection_index": artifact_index,
        "image_path": str(image_path),
        "depth_path": str(projection.get("source_artifact")),
        "annotated_image": str(annotated_path),
        "calibration_path": str(get_calibration_path()),
        "coordinate_convention": (
            f"point_camera_xyz is in the {camera_alias} color optical frame; "
            "point_xyz is in the shared right_base world frame."
        ),
        "calibration_key": calibration_key,
        f"T_right_base_{camera_alias}_camera": np.round(
            transform_right_camera, 8
        ).tolist(),
        "projection": projection,
        "depth_patch": _depth_patch_stats(depth, row=row, col=col, radius=radius),
    }
    report_artifact = f"{camera_alias}_back_project_{artifact_index:02d}.json"
    report_name = state.save(report_artifact, report, step=step_idx)
    if report_name is None:
        raise DualFrankaPerceptionError("failed to save back-project report")
    return {
        "annotated_image": str(annotated_path),
        "report_json": str(state.artifact_path(report_name, step=step_idx)),
    }


def _mask_to_camera_world(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    camera: str,
    state: EnvState,
    step_idx: int,
    min_valid: int,
) -> dict[str, Any]:
    camera = _resolve_projection_camera_alias(camera)
    projection_cameras = _projection_cameras_for_state(state, step_idx)
    camera_config = projection_cameras.get(camera)
    if camera_config is None:
        known = ", ".join(sorted(projection_cameras)) or "<none>"
        raise DualFrankaPerceptionError(
            f"unsupported projection camera: {camera!r}; registered={known}"
        )
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        return {
            "point_xyz": None,
            "world_error": f"expected 2D {camera} depth, got {tuple(depth.shape)}",
            "mask_pixels": int(mask.sum()),
            "valid_depth_pixels": 0,
            "valid_localization_pixels": 0,
        }
    if mask.shape != depth.shape:
        return {
            "point_xyz": None,
            "world_error": (
                f"mask/depth shape mismatch: mask={tuple(mask.shape)}, "
                f"depth={tuple(depth.shape)}"
            ),
            "mask_pixels": int(mask.sum()),
            "valid_depth_pixels": 0,
            "valid_localization_pixels": 0,
        }

    rows, cols = np.where(mask)
    result: dict[str, Any] = {
        "mask_pixels": int(rows.size),
        "valid_depth_pixels": 0,
        "valid_localization_pixels": 0,
        "mask_resized_to_depth_shape": False,
    }
    if rows.size == 0:
        result.update({"point_xyz": None, "world_error": "empty mask"})
        return result

    depths = depth[rows, cols].astype(np.float64)
    valid_depth = np.isfinite(depths) & (depths > 0.0)
    rows = rows[valid_depth]
    cols = cols[valid_depth]
    depths = depths[valid_depth]
    result["valid_depth_pixels"] = int(depths.size)
    if depths.size < min_valid:
        result.update(
            {
                "point_xyz": None,
                "world_error": f"too few valid {camera} depth pixels ({int(depths.size)})",
            }
        )
        return result

    meta = _camera_meta(
        state,
        step_idx,
        camera_alias=camera,
        raw_key=str(camera_config["raw_key"]),
    )
    intr = meta.get("color_intrinsics") or {}
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr.get("ppx", intr.get("cx")))
    cy = float(intr.get("ppy", intr.get("cy")))
    points_camera = np.stack(
        [
            (cols.astype(np.float64) - cx) * depths / fx,
            (rows.astype(np.float64) - cy) * depths / fy,
            depths,
        ],
        axis=1,
    )

    calibration = load_calibration_bundle()
    calibration_key = str(camera_config["calibration_key"])
    camera_calibration = calibration.get(calibration_key)
    if not isinstance(camera_calibration, dict):
        raise DualFrankaPerceptionError(
            f"calibration entry {calibration_key!r} is missing"
        )
    t_right_camera = _transform_to_matrix(camera_calibration["transformation"])
    points_right = _transform_points(t_right_camera, points_camera)
    valid_localization, validity_contract = _localization_validity_mask(
        camera_calibration=camera_calibration,
        depths=depths,
        points_right=points_right,
    )
    result["validity_contract"] = validity_contract
    rows_valid = rows[valid_localization]
    cols_valid = cols[valid_localization]
    depths_valid = depths[valid_localization]
    points_camera_valid = points_camera[valid_localization]
    points_right_valid = points_right[valid_localization]
    result["valid_localization_pixels"] = int(points_right_valid.shape[0])
    if points_right_valid.shape[0] < min_valid:
        result.update(
            {
                "point_xyz": None,
                "world_error": (
                    f"too few mask pixels remain inside configured {camera} localization "
                    f"volume ({int(points_right_valid.shape[0])})"
                ),
            }
        )
        return result

    point_right = np.median(points_right_valid, axis=0)
    point_camera = np.median(points_camera_valid, axis=0)
    depth_m = float(np.median(depths_valid))
    centroid_row = int(round(float(np.median(rows_valid))))
    centroid_col = int(round(float(np.median(cols_valid))))
    selection_valid, rejection_reasons, _contract = _validate_localization_point(
        camera_calibration=camera_calibration,
        depth_m=depth_m,
        point_right=point_right,
    )

    result.update(
        {
            "selection_valid": bool(selection_valid),
            "centroid_pixel": [centroid_row, centroid_col],
            "depth_m": round(depth_m, 5),
            "point_camera_xyz": _round(point_camera),
            "point_xyz": _round(point_right) if selection_valid else None,
            "raw_median_point_xyz": _round(point_right),
            "camera_extrinsic_frame": "right_base",
            "calibration_path": str(get_calibration_path()),
            "calibration_key": calibration_key,
        }
    )
    if selection_valid:
        _step_idx, record_state = _resolve_step(state, step_idx)
        result.update(
            _tcp_delta_fields(record_state, point_right, calibration=calibration)
        )
    if rejection_reasons:
        result["rejection_reasons"] = rejection_reasons
        result["world_error"] = "Rejected SAM3 mask localization: " + "; ".join(
            rejection_reasons
        )
    return result


def _localization_validity_mask(
    *,
    camera_calibration: dict[str, Any],
    depths: np.ndarray,
    points_right: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = camera_calibration.get("localization_validity") or {}
    depth_bounds = np.asarray(config.get("depth_m", [0.15, 1.25]), dtype=np.float64)
    xyz_min = np.asarray(
        config.get("right_base_xyz_min", [0.10, -0.85, 0.00]), dtype=np.float64
    )
    xyz_max = np.asarray(
        config.get("right_base_xyz_max", [1.15, 0.85, 0.85]), dtype=np.float64
    )
    valid = (
        np.isfinite(depths)
        & np.isfinite(points_right).all(axis=1)
        & (depths >= depth_bounds[0])
        & (depths <= depth_bounds[1])
        & (points_right[:, 0] >= xyz_min[0])
        & (points_right[:, 0] <= xyz_max[0])
        & (points_right[:, 1] >= xyz_min[1])
        & (points_right[:, 1] <= xyz_max[1])
        & (points_right[:, 2] >= xyz_min[2])
        & (points_right[:, 2] <= xyz_max[2])
    )
    contract = {
        "depth_m": _round(depth_bounds),
        "right_base_xyz_min": _round(xyz_min),
        "right_base_xyz_max": _round(xyz_max),
    }
    return valid, contract


def _next_named_artifact_index(
    record: StepRecord,
    *,
    prefix: str,
    suffix: str,
) -> int:
    idx = 0
    while f"{prefix}_{idx:02d}{suffix}" in record.artifacts:
        idx += 1
    return idx


def _make_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    localization: dict[str, Any],
) -> np.ndarray | None:
    image = np.asarray(image)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image[..., :3].copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)

    centroid = localization.get("centroid_pixel")
    if isinstance(centroid, list) and len(centroid) == 2:
        row, col = int(centroid[0]), int(centroid[1])
        # Draw on a fresh PIL image and copy back so imageio receives ndarray.
        pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil)
        width, height = pil.size
        x = max(0, min(width - 1, col))
        y = max(0, min(height - 1, row))
        color = (
            (0, 255, 0) if localization.get("point_xyz") is not None else (255, 0, 0)
        )
        draw.line((x - 15, y, x + 15, y), fill=color, width=2)
        draw.line((x, y - 15, x, y + 15), fill=color, width=2)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=2)
        label = f"SAM3 r{row},c{col} rb={_short_xyz(localization.get('point_xyz') or localization.get('raw_median_point_xyz'))}"
        label_x = min(x + 14, max(0, width - 380))
        label_y = max(20, y - 14)
        draw.text((label_x, label_y), label, fill=color)
        overlay = np.asarray(pil)
    return overlay


def _depth_patch_stats(
    depth: np.ndarray, *, row: int, col: int, radius: int
) -> dict[str, Any]:
    r0 = max(0, row - radius)
    r1 = min(depth.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(depth.shape[1], col + radius + 1)
    patch = np.asarray(depth[r0:r1, c0:c1], dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    raw_depth = float(depth[row, col]) if np.isfinite(depth[row, col]) else None
    stats: dict[str, Any] = {
        "window_bounds_rc": [int(r0), int(r1), int(c0), int(c1)],
        "window_shape": list(patch.shape),
        "raw_depth_at_pixel_m": round(raw_depth, 6) if raw_depth is not None else None,
        "valid_pixels": int(valid.size),
        "total_pixels": int(patch.size),
    }
    if valid.size:
        stats.update(
            {
                "min_m": round(float(valid.min()), 6),
                "max_m": round(float(valid.max()), 6),
                "mean_m": round(float(valid.mean()), 6),
                "median_m": round(float(np.median(valid)), 6),
                "std_m": round(float(valid.std()), 6),
            }
        )
    return stats


def _short_xyz(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 3:
        return "n/a"
    return f"{float(value[0]):.3f},{float(value[1]):.3f},{float(value[2]):.3f}"


def _median_depth(
    depth: np.ndarray, row: int, col: int, *, radius: int
) -> tuple[float, int]:
    r0 = max(0, row - radius)
    r1 = min(depth.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(depth.shape[1], col + radius + 1)
    patch = depth[r0:r1, c0:c1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise DualFrankaPerceptionError(
            f"no valid depth near pixel row={row} col={col} radius={radius}"
        )
    return float(np.median(valid)), int(valid.size)


def _transform_to_matrix(transform: dict[str, Any]) -> np.ndarray:
    if "matrix" in transform:
        mat = np.asarray(transform["matrix"], dtype=np.float64)
        if mat.shape != (4, 4):
            raise DualFrankaPerceptionError(
                f"expected 4x4 transform matrix, got {mat.shape}"
            )
        return mat
    qw = float(transform["qw"])
    qx = float(transform["qx"])
    qy = float(transform["qy"])
    qz = float(transform["qz"])
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    rot = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [float(transform["x"]), float(transform["y"]), float(transform["z"])]
    return mat


def _base_frame_transform(
    calibration: dict[str, Any],
    *,
    target: str,
    source: str,
) -> np.ndarray:
    frames = calibration.get("base_frames") or {}
    key = f"T_{target}_{source}"
    if isinstance(frames.get(key), dict):
        return _transform_to_matrix(frames[key])
    inverse_key = f"T_{source}_{target}"
    if isinstance(frames.get(inverse_key), dict):
        return np.linalg.inv(_transform_to_matrix(frames[inverse_key]))
    raise DualFrankaPerceptionError(f"missing base-frame transform {key}")


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homo = np.ones(4, dtype=np.float64)
    homo[:3] = point
    return (transform @ homo)[:3]


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homo = np.ones((points.shape[0], 4), dtype=np.float64)
    homo[:, :3] = points
    return (transform @ homo.T).T[:, :3]


def _tcp_xyz(pose: Any) -> np.ndarray | None:
    if not isinstance(pose, (list, tuple, np.ndarray)) or len(pose) < 3:
        return None
    return np.asarray(pose[:3], dtype=np.float64)


def _round(value: np.ndarray, ndigits: int = 5) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(value).reshape(-1)]
