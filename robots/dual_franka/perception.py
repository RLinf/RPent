"""Dual-Franka RGBD perception helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rpent.tools.state import EnvState
from rpent.tools.toolkit import readonly

DEFAULT_CALIBRATION_BUNDLE_PATH = (
    Path(__file__).resolve().parent / "calibration" / "hand_eye_calibration.json"
)


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


def load_snapshot(
    output_dir: str | Path,
    step: int | None = None,
    *,
    state: EnvState | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    if state is not None:
        resolved_step = -1 if step is None else step
        record = state.get(resolved_step)
        artifacts: dict[str, Any] = {"images": {}, "depths": {}}
        for alias, image_name, depth_name in (
            ("base", "base.png", "base_depth.npy"),
            ("d455", "d455.png", "d455_depth.npy"),
        ):
            if state.exists(image_name, step=record.step_idx):
                artifacts["images"][alias] = str(
                    state.artifact_path(image_name, step=record.step_idx)
                )
            if state.exists(depth_name, step=record.step_idx):
                artifacts["depths"][alias] = str(
                    state.artifact_path(depth_name, step=record.step_idx)
                )
        if state.exists("camera_meta.json", step=record.step_idx):
            artifacts["camera_meta"] = state.load(
                "camera_meta.json", step=record.step_idx
            )
        return {
            "step_idx": record.step_idx,
            "state": record.state,
            "artifacts": artifacts,
        }
    manifest_path = out / "states.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(errors="replace"))
        records = manifest.get("steps") if isinstance(manifest, dict) else None
        if isinstance(records, list) and records:
            index = len(records) - 1 if step is None or step == -1 else int(step)
            if not 0 <= index < len(records):
                raise DualFrankaPerceptionError(
                    f"step {step} not found in {manifest_path}"
                )
            record = records[index]
            artifacts: dict[str, Any] = {"images": {}, "depths": {}}
            for alias, image_name, depth_name in (
                ("base", "base.png", "base_depth.npy"),
                ("d455", "d455.png", "d455_depth.npy"),
            ):
                image_path = out / image_name / f"{index:02d}{Path(image_name).suffix}"
                depth_path = out / depth_name / f"{index:02d}{Path(depth_name).suffix}"
                if image_path.exists():
                    artifacts["images"][alias] = str(image_path)
                if depth_path.exists():
                    artifacts["depths"][alias] = str(depth_path)
            meta_path = out / "camera_meta.json" / f"{index:02d}.json"
            if meta_path.exists():
                artifacts["camera_meta"] = json.loads(meta_path.read_text())
            return {
                "step_idx": index,
                "state": record.get("state") or {},
                "artifacts": artifacts,
            }
    if step is None:
        latest = out / "latest_state.json"
        if latest.exists():
            data = json.loads(latest.read_text(errors="replace"))
            if isinstance(data, dict) and "artifacts" in data:
                return data
    full_log = out / "full_log.jsonl"
    if not full_log.exists():
        raise DualFrankaPerceptionError(f"full_log.jsonl not found under {out}")
    matches: list[dict[str, Any]] = []
    for line in full_log.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("event") != "snapshot":
            continue
        if step is None or int(row.get("step_idx", -1)) == int(step):
            matches.append(row)
    if not matches:
        raise DualFrankaPerceptionError(
            f"snapshot step {step} not found under {out}" if step is not None else f"no snapshots found under {out}"
        )
    return matches[-1]


@readonly
def back_project_base_pixel(
    output_dir: str | Path | None = None,
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
        _resolve_output_dir(output_dir, state),
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
    output_dir: str | Path | None = None,
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
        _resolve_output_dir(output_dir, state),
        camera="d455",
        row=row,
        col=col,
        target_name=target_name,
        step=step,
        window_radius=window_radius,
        state=state,
    )


def _resolve_output_dir(
    output_dir: str | Path | None,
    state: EnvState | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if state is not None:
        return state.artifact_path("camera_meta.json", step=None).parent
    raise DualFrankaPerceptionError("output_dir or state is required")


def _back_project_camera_pixel(
    output_dir: str | Path,
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
    snapshot = load_snapshot(output_dir, step=step, state=state)
    artifacts = snapshot.get("artifacts") or {}
    depths = artifacts.get("depths") or {}
    depth_path = depths.get(camera)
    if not depth_path:
        raise DualFrankaPerceptionError(
            f"{camera_config['display_name']} depth artifact is missing. "
            "Restart the env server with the camera and depth enabled, then "
            "call view_driver_state/reset again."
        )
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
        snapshot,
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

    state_blob = snapshot.get("state") or {}
    raw = state_blob.get("raw") or {}
    right_state = raw.get("right") or state_blob.get("right_arm") or {}
    left_state = raw.get("left") or state_blob.get("left_arm") or {}
    right_tcp = _tcp_xyz(right_state.get("tcp_pose"))
    left_tcp_local = _tcp_xyz(left_state.get("tcp_pose"))
    left_tcp = (
        transform_point_between_base_frames(left_tcp_local, target="right_base", source="left_base")
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
        "step": snapshot.get("step_idx"),
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
            output_dir=Path(output_dir),
            snapshot=snapshot,
            depth=depth,
            projection=out,
            transform_right_camera=t_right_camera,
            camera_alias=camera,
            calibration_key=calibration_key,
        )
    except Exception as exc:
        out["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return out


def load_calibration_bundle(path: str | Path | None = None) -> dict[str, Any]:
    bundle_path = Path(path or DEFAULT_CALIBRATION_BUNDLE_PATH)
    data = json.loads(bundle_path.read_text(errors="replace"))
    if not isinstance(data, dict):
        raise DualFrankaPerceptionError(f"invalid calibration bundle: {bundle_path}")
    return data


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


def transform_vector_between_base_frames(
    vector: Any,
    *,
    target: str,
    source: str,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(3)
    if target == source:
        return vec
    t_target_source = _base_frame_transform(
        calibration or load_calibration_bundle(),
        target=target,
        source=source,
    )
    return t_target_source[:3, :3] @ vec


def base_rotation_matrix(
    *,
    target: str,
    source: str,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    if target == source:
        return np.eye(3, dtype=np.float64)
    t_target_source = _base_frame_transform(
        calibration or load_calibration_bundle(),
        target=target,
        source=source,
    )
    return t_target_source[:3, :3].copy()


def _camera_meta(
    snapshot: dict[str, Any],
    *,
    camera_alias: str,
    raw_key: str,
) -> dict[str, Any]:
    meta = ((snapshot.get("artifacts") or {}).get("camera_meta") or {})
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
    output_dir: Path,
    snapshot: dict[str, Any],
    depth: np.ndarray,
    projection: dict[str, Any],
    transform_right_camera: np.ndarray,
    camera_alias: str,
    calibration_key: str,
) -> dict[str, str]:
    """Persist a marked camera image and JSON report for one projection call."""
    import cv2

    artifacts = snapshot.get("artifacts") or {}
    images = artifacts.get("images") or {}
    image_path = images.get(camera_alias)
    if camera_alias == "base":
        image_path = image_path or images.get("extra_0")
    if not image_path:
        raise DualFrankaPerceptionError(
            f"{camera_alias} image artifact is missing"
        )

    pixel = projection.get("pixel") or []
    if len(pixel) != 2:
        raise DualFrankaPerceptionError(f"invalid projection pixel: {pixel!r}")
    row, col = int(pixel[0]), int(pixel[1])
    step = projection.get("step")
    radius = int(projection.get("depth_window_radius") or 0)
    diag_dir = output_dir / "localization_diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    stem = f"back_project_{camera_alias}_step_{step}_r{row}_c{col}"
    annotated_path = diag_dir / f"{stem}.png"
    report_path = diag_dir / f"{stem}.json"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise DualFrankaPerceptionError(f"failed to read base image: {image_path}")
    color = (0, 255, 0) if projection.get("selection_valid", True) else (0, 0, 255)
    cv2.drawMarker(
        image,
        (col, row),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(image, (col, row), 12, color, 2, lineType=cv2.LINE_AA)
    label = (
        f"r{row},c{col} cam={_short_xyz(projection.get('point_camera_xyz'))} "
        f"rb={_short_xyz(projection.get('point_xyz'))}"
    )
    label_x = min(col + 14, max(0, image.shape[1] - 390))
    label_y = max(20, row - 14)
    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(annotated_path), image):
        raise DualFrankaPerceptionError(f"failed to write {annotated_path}")

    report = {
        "ok": bool(projection.get("ok")),
        "snapshot_step": step,
        "image_path": str(image_path),
        "depth_path": str(projection.get("source_artifact")),
        "annotated_image": str(annotated_path),
        "calibration_path": str(DEFAULT_CALIBRATION_BUNDLE_PATH),
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
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "annotated_image": str(annotated_path),
        "report_json": str(report_path),
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
