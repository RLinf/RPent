"""Dual-Franka RGBD perception helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from robots.franka.runtime_config import get_calibration_path, load_mapping
from rpent.tools.state import EnvState
from rpent.tools.toolkit import readonly

ROBOT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "robot_config.yaml"


class DualFrankaPerceptionError(ValueError):
    """Raised when a dual-Franka perception artifact is missing or invalid."""


_PROJECTION_CAMERAS = {
    "base": {
        "raw_key": "base_0_rgb",
        "calibration_key": "base_camera",
        "display_name": "base RealSense",
    },
    "d455": {
        "raw_key": "d455_rgb",
        "calibration_key": "d455_camera",
        "display_name": "D455",
    },
}


def _resolve_step(state: EnvState, step: int | None) -> tuple[int, dict[str, Any]]:
    """Resolve a step selector (None or -1 = latest) to index and robot state."""
    record = state.get(-1 if step is None else step)
    return record.step_idx, record.state


@readonly
def back_project_base_pixel(
    *,
    row: int,
    col: int,
    target_name: str = "target",
    step: int | None = None,
    window_radius: int = 2,
    state: EnvState | None = None,
) -> dict[str, Any]:
    """Back-project a base-camera pixel into the shared right_base world frame."""
    return _back_project_camera_pixel(
        camera="base",
        row=row,
        col=col,
        target_name=target_name,
        step=step,
        window_radius=window_radius,
        state=state,
    )


@readonly
def back_project_d455_pixel(
    *,
    row: int,
    col: int,
    target_name: str = "target",
    step: int | None = None,
    window_radius: int = 2,
    state: EnvState | None = None,
) -> dict[str, Any]:
    """Back-project a D455 pixel into the shared right_base world frame."""
    return _back_project_camera_pixel(
        camera="d455",
        row=row,
        col=col,
        target_name=target_name,
        step=step,
        window_radius=window_radius,
        state=state,
    )


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
    camera_config = _PROJECTION_CAMERAS.get(camera)
    if camera_config is None:
        raise DualFrankaPerceptionError(f"unsupported projection camera: {camera!r}")
    if state is None:
        raise DualFrankaPerceptionError("state is required")
    step_idx, record_state = _resolve_step(state, step)
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

    state_blob = record_state or {}
    raw = state_blob.get("raw") or {}
    right_state = raw.get("right") or state_blob.get("right_arm") or {}
    left_state = raw.get("left") or state_blob.get("left_arm") or {}
    right_tcp = _tcp_xyz(right_state.get("tcp_pose"))
    left_tcp_local = _tcp_xyz(left_state.get("tcp_pose"))
    left_tcp = (
        transform_point_between_base_frames(left_tcp_local, target="right_base", source="left_base", calibration=calibration)
        if left_tcp_local is not None
        else None
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
    if right_tcp is not None:
        out["right_tcp_xyz"] = _round(right_tcp)
        out["delta_right_tcp_to_point_xyz"] = _round(point_right - right_tcp)
    if left_tcp is not None:
        out["left_tcp_xyz"] = _round(left_tcp)
        out["delta_left_tcp_to_point_xyz"] = _round(point_right - left_tcp)
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
    except Exception as exc:
        out["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _load_perception_config() -> dict[str, Any]:
    """Load the RPent perception section from ``robot_config.yaml``.

    Holds the machine config ``easy_handeye`` does not produce: the tabletop
    ``localization_validity`` bounds and the inter-base ``base_frames``.
    """
    raw = load_mapping(ROBOT_CONFIG_PATH)
    perception = raw.get("perception")
    if not isinstance(perception, dict):
        raise DualFrankaPerceptionError(
            "robot_config.yaml missing the 'perception' section"
        )
    return perception


def load_calibration_bundle(path: str | Path | None = None) -> dict[str, Any]:
    """Load the dual-Franka perception calibration.

    Merges the ``easy_handeye`` hand-eye transforms (``hand_eye_calibration.json``)
    with the RPent perception config (localization validity + base frames) from
    ``robot_config.yaml``, keeping the historical consumer shape:
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
    for camera_key, validity in (
        perception.get("localization_validity") or {}
    ).items():
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
    image_name = f"{camera_alias}.png"
    if not state.exists(image_name, step=step_idx):
        raise DualFrankaPerceptionError(
            f"{camera_alias} image artifact is missing"
        )
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

    annotated_name = state.save(
        "back_project_annotated.png", np.asarray(image), step=step_idx
    )
    if annotated_name is None:
        raise DualFrankaPerceptionError("failed to save annotated image")
    annotated_path = state.artifact_path(annotated_name, step=step_idx)

    report = {
        "ok": bool(projection.get("ok")),
        "snapshot_step": step_idx,
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
    report_name = state.save("back_project_report.json", report, step=step_idx)
    if report_name is None:
        raise DualFrankaPerceptionError("failed to save back-project report")
    return {
        "annotated_image": str(annotated_path),
        "report_json": str(state.artifact_path(report_name, step=step_idx)),
    }


def _depth_patch_stats(depth: np.ndarray, *, row: int, col: int, radius: int) -> dict[str, Any]:
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


def _median_depth(depth: np.ndarray, row: int, col: int, *, radius: int) -> tuple[float, int]:
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
            raise DualFrankaPerceptionError(f"expected 4x4 transform matrix, got {mat.shape}")
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
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
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

    # Backward-compatible fallback for the current dual-Franka calibration:
    # right_base origin expressed in left_base, with identity rotation.
    if target == "left_base" and source == "right_base":
        xyz = frames.get("right_base_in_left_base_xyz_m")
        if xyz is not None:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 3] = np.asarray(xyz, dtype=np.float64)
            return mat
    if target == "right_base" and source == "left_base":
        xyz = frames.get("right_base_in_left_base_xyz_m")
        if xyz is not None:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 3] = -np.asarray(xyz, dtype=np.float64)
            return mat
    raise DualFrankaPerceptionError(f"missing base-frame transform {key}")


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homo = np.ones(4, dtype=np.float64)
    homo[:3] = point
    return (transform @ homo)[:3]


def _tcp_xyz(pose: Any) -> np.ndarray | None:
    if not isinstance(pose, (list, tuple)) or len(pose) < 3:
        return None
    return np.asarray(pose[:3], dtype=np.float64)


def _round(value: np.ndarray, ndigits: int = 5) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(value).reshape(-1)]
