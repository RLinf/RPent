"""Franka RGBD back-projection helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from rpent.tools.state import EnvState
from rpent.tools.toolkit import readonly

DEFAULT_CALIBRATION_BUNDLE_PATH = (
    Path(__file__).resolve().parent / "calibration" / "hand_eye_calibration.json"
)


class PerceptionError(ValueError):
    """Raised when a Franka perception artifact is missing or inconsistent."""


def load_camera_meta(
    output_dir: Path,
    *,
    state: EnvState | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Load RLinf camera metadata from latest/full Franka logs."""
    snapshot = load_snapshot(output_dir, step, state=state)
    meta = ((snapshot.get("artifacts") or {}).get("camera_meta"))
    if isinstance(meta, dict):
        return meta
    raise PerceptionError(
        "camera metadata not found in latest_state.json/full_log.jsonl "
        f"under {output_dir}"
    )


def load_snapshot(
    output_dir: Path,
    step: int | None = None,
    *,
    state: EnvState | None = None,
) -> dict[str, Any]:
    """Load a Franka snapshot from latest_state.json or full_log.jsonl."""
    if state is not None:
        resolved_step = -1 if step is None else step
        record = state.get(resolved_step)
        artifacts: dict[str, Any] = {"images": {}, "depths": {}}
        for key, image_name, depth_name in (
            ("main", "wrist.png", "wrist_depth.npy"),
            ("extra_0", "camera.png", "camera_depth.npy"),
        ):
            if state.exists(image_name, step=record.step_idx):
                artifacts["images"][key] = str(
                    state.artifact_path(image_name, step=record.step_idx)
                )
            if state.exists(depth_name, step=record.step_idx):
                artifacts["depths"][key] = str(
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

    manifest_path = output_dir / "states.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(errors="replace"))
        records = manifest.get("steps") if isinstance(manifest, dict) else None
        if isinstance(records, list) and records:
            index = len(records) - 1 if step is None or step == -1 else int(step)
            if not 0 <= index < len(records):
                raise PerceptionError(f"step {step} not found in {manifest_path}")
            record = records[index]
            artifacts: dict[str, Any] = {"images": {}, "depths": {}}
            for key, image_name, depth_name in (
                ("main", "wrist.png", "wrist_depth.npy"),
                ("extra_0", "camera.png", "camera_depth.npy"),
            ):
                image_path = output_dir / image_name / f"{index:02d}{Path(image_name).suffix}"
                depth_path = output_dir / depth_name / f"{index:02d}{Path(depth_name).suffix}"
                if image_path.exists():
                    artifacts["images"][key] = str(image_path)
                if depth_path.exists():
                    artifacts["depths"][key] = str(depth_path)
            meta_path = output_dir / "camera_meta.json" / f"{index:02d}.json"
            if meta_path.exists():
                artifacts["camera_meta"] = json.loads(meta_path.read_text())
            return {
                "step_idx": index,
                "state": record.get("state") or {},
                "artifacts": artifacts,
            }

    if step is None:
        latest = output_dir / "latest_state.json"
        if latest.exists():
            data = json.loads(latest.read_text(errors="replace"))
            if isinstance(data, dict) and "state" in data and "artifacts" in data:
                return data

    states_path = output_dir / "full_log.jsonl"
    if not states_path.exists():
        raise PerceptionError(f"full_log.jsonl not found under {output_dir}")
    matches: list[dict[str, Any]] = []
    for line in states_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if item.get("event") not in {None, "snapshot"}:
            continue
        if step is None or item.get("step_idx") == int(step):
            matches.append(item)
    if not matches:
        if step is None:
            raise PerceptionError(f"no snapshots found in {states_path}")
        raise PerceptionError(f"step {step} not found in {states_path}")
    return matches[-1]


def load_calibration_bundle(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load normalized hand-eye calibration from the project calibration bundle."""
    bundle_path = Path(path or DEFAULT_CALIBRATION_BUNDLE_PATH).expanduser().resolve()
    bundle = _load_calibration_bundle_file(bundle_path)
    external = _calibration_entry(bundle, "external")
    wrist = _calibration_entry(bundle, "wrist")
    return {
        "external": _normalize_calibration(bundle_path, external),
        "wrist": _normalize_calibration(bundle_path, wrist),
        "convention": (
            "The YAML transformation is interpreted as the camera pose in the "
            "YAML base-frame coordinate system. The resulting matrix maps "
            "camera-frame homogeneous points into that base frame."
        ),
    }


@readonly
def view_perception_setup(
    output_dir: Path | None = None,
    *,
    state: EnvState | None = None,
    step: int = -1,
) -> dict[str, Any]:
    """Return camera metadata plus normalized calibration summaries."""
    if output_dir is None:
        if state is None:
            raise PerceptionError("output_dir or state is required")
        output_dir = state.artifact_path("camera_meta.json", step=None).parent
    meta = load_camera_meta(output_dir, state=state, step=step)
    calibration = load_calibration_bundle()
    return {
        "camera_meta": meta,
        "calibration": {
            name: _compact_calibration(item)
            for name, item in calibration.items()
            if isinstance(item, dict)
        },
        "convention": calibration["convention"],
        "current_policy": (
            "back_project accepts a single pixel from one camera and returns "
            "the corresponding point in the robot base frame. The tool owns "
            "depth lookup, intrinsics, and hand-eye calibration; callers should "
            "not manually compute camera transforms."
        ),
    }


@readonly
def back_project(
    output_dir: Path | None = None,
    *,
    row: int,
    col: int,
    step: int | None = None,
    camera: str = "wrist",
    debug: bool = False,
    state: EnvState | None = None,
) -> dict[str, Any]:
    """Back-project one camera pixel into the Franka robot base frame."""
    if output_dir is None:
        if state is None:
            raise PerceptionError("output_dir or state is required")
        output_dir = state.artifact_path("camera_meta.json", step=None).parent
    snapshot = load_snapshot(output_dir, step, state=state)
    meta = load_camera_meta(output_dir, state=state, step=step)
    calibration = load_calibration_bundle()
    tcp_pose = _snapshot_tcp_pose(snapshot)
    t_base_tcp = pose7_to_matrix(tcp_pose)

    camera_alias = _normalize_camera_alias(camera)
    if camera_alias == "wrist":
        t_camera_target = calibration["wrist"]["matrix"]
        target_frame = "tcp"
    else:
        t_camera_target = calibration["external"]["matrix"]
        target_frame = "base"

    projection = _project_view_to_base(
        output_dir=output_dir,
        meta=meta,
        snapshot=snapshot,
        camera_alias=camera_alias,
        row=int(row),
        col=int(col),
        t_camera_target=t_camera_target,
        target_frame=target_frame,
        t_base_tcp=t_base_tcp,
    )
    overlay_path = _save_selected_pixel_overlay(
        output_dir=output_dir,
        snapshot=snapshot,
        meta=meta,
        camera_alias=camera_alias,
        row=int(row),
        col=int(col),
        projection=projection,
    )
    if "point_base" not in projection:
        out = {
            "error": projection.get("error", "back-projection failed"),
            "error_type": projection.get("error_type", "PerceptionError"),
            "camera": camera_alias,
            "pixel": [int(row), int(col)],
            "step": snapshot.get("step_idx"),
        }
        if overlay_path:
            out["selected_pixel_overlay"] = str(overlay_path)
        return out

    out = {
        "camera": camera_alias,
        "pixel": [int(row), int(col)],
        "point_base": projection["point_base"],
        "world_xyz": projection["point_base"],
        "coordinate_frame": "franka_base",
        "depth_m": projection.get("depth_m"),
        "step": snapshot.get("step_idx"),
        "source": "single_view_rgbd",
        "source_artifact": projection.get("depth_path"),
        "camera_key": projection.get("camera_key"),
        "camera_name": projection.get("camera_name"),
    }
    if overlay_path:
        out["selected_pixel_overlay"] = str(overlay_path)
    if debug:
        out["debug"] = {
            "point_camera": projection.get("point_camera"),
            "point_target": projection.get("point_target"),
            "target_frame": projection.get("target_frame"),
            "tcp_pose_xyzw": _round_vec(tcp_pose),
            "note": (
                "Wrist points are transformed through the current robot TCP "
                "pose. Third-person points use the fixed external camera "
                "calibration."
            ),
        }
    return out


def _save_selected_pixel_overlay(
    *,
    output_dir: Path,
    snapshot: dict[str, Any],
    meta: dict[str, Any],
    camera_alias: str,
    row: int,
    col: int,
    projection: dict[str, Any],
) -> Path | None:
    """Save an RGB image marking the pixel selected for back-projection."""
    try:
        camera_key = projection.get("camera_key")
        camera_name = projection.get("camera_name")
        if not camera_key or not camera_name:
            camera_key, camera_name = _resolve_camera_alias(meta, camera_alias)

        image_key = "main" if camera_key == "main" else "extra_0"
        image_path = (
            ((snapshot.get("artifacts") or {}).get("images") or {}).get(image_key)
        )
        if not image_path:
            return None

        source = Path(image_path)
        if not source.exists():
            return None

        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        x = max(0, min(width - 1, int(col)))
        y = max(0, min(height - 1, int(row)))
        radius = 9

        # White halo plus red mark stays visible on dark and bright objects.
        draw.ellipse(
            (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
            outline="white",
            width=4,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline="red",
            width=3,
        )
        draw.line((x - 16, y, x + 16, y), fill="red", width=3)
        draw.line((x, y - 16, x, y + 16), fill="red", width=3)

        status = "ok" if "point_base" in projection else "error"
        label = (
            f"{status} {camera_alias} step={snapshot.get('step_idx')} "
            f"row={row} col={col}"
        )
        label_bg = (0, 0, 0)
        draw.rectangle((4, 4, min(width - 1, 12 + len(label) * 7), 24), fill=label_bg)
        draw.text((8, 8), label, fill="white")

        out_dir = output_dir / "back_project_overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        step = int(snapshot.get("step_idx") or 0)
        stem = f"back_project_step{step:04d}_{camera_alias}_r{row}_c{col}"
        existing = sorted(out_dir.glob(stem + "_*.png"))
        out_path = out_dir / f"{stem}_{len(existing):02d}.png"
        image.save(out_path)
        return out_path
    except Exception:
        return None


@readonly
def back_project_correspondence(
    output_dir: Path | None = None,
    *,
    third_person_row: int | None = None,
    third_person_col: int | None = None,
    wrist_row: int | None = None,
    wrist_col: int | None = None,
    pixels: list[dict[str, Any]] | None = None,
    step: int | None = None,
    debug: bool = False,
    state: EnvState | None = None,
) -> dict[str, Any]:
    """Back-project target pixels to robot base frame.

    Wrist pixels are required. Matched third-person pixels are optional but
    recommended when the task exposes this tool: their independently projected
    base-frame point is compared against the wrist estimate to produce a
    confidence score, and high/medium-confidence pairs are fused.
    """
    if output_dir is None:
        if state is None:
            raise PerceptionError("output_dir or state is required")
        output_dir = state.artifact_path("camera_meta.json", step=None).parent
    snapshot = load_snapshot(output_dir, step, state=state)
    meta = load_camera_meta(output_dir, state=state, step=step)
    calibration = load_calibration_bundle()
    tcp_pose = _snapshot_tcp_pose(snapshot)
    t_base_tcp = pose7_to_matrix(tcp_pose)
    requests = _normalize_pixel_requests(
        third_person_row=third_person_row,
        third_person_col=third_person_col,
        wrist_row=wrist_row,
        wrist_col=wrist_col,
        pixels=pixels,
    )
    depth_cache: dict[str, tuple[np.ndarray, Path]] = {}
    k_inv_cache: dict[str, np.ndarray] = {}

    points = [
        _back_project_one_correspondence(
            output_dir=output_dir,
            meta=meta,
            snapshot=snapshot,
            calibration=calibration,
            t_base_tcp=t_base_tcp,
            tcp_pose=tcp_pose,
            request=request,
            debug=debug,
            depth_cache=depth_cache,
            k_inv_cache=k_inv_cache,
        )
        for request in requests
    ]

    if pixels is not None:
        valid_points = [point for point in points if "point_base" in point]
        reliable_points = [
            point
            for point in valid_points
            if point.get("confidence", {}).get("level") in {"high", "medium"}
        ]
        return {
            "points": points,
            "source": "multi_view_rgbd",
            "step": snapshot.get("step_idx"),
            "count": len(points),
            "valid_count": len(valid_points),
            "reliable_count": len(reliable_points),
            "aggregate": _aggregate_points(reliable_points or valid_points),
            "tcp_pose_source": "RLinf raw_base_state.tcp_pose",
        }
    return points[0]


def _normalize_camera_alias(camera: str) -> str:
    value = str(camera).strip().lower()
    if value in {"wrist", "main"}:
        return "wrist"
    if value in {"third_person", "third-person", "external", "extra_0", "agentview"}:
        return "third_person"
    raise PerceptionError(
        "unsupported camera; use 'wrist' or 'third_person'"
    )


def _back_project_one_correspondence(
    *,
    output_dir: Path,
    meta: dict[str, Any],
    snapshot: dict[str, Any],
    calibration: dict[str, Any],
    t_base_tcp: np.ndarray,
    tcp_pose: list[float],
    request: dict[str, int | None],
    debug: bool,
    depth_cache: dict[str, tuple[np.ndarray, Path]],
    k_inv_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    third_person_row = request["third_person_row"]
    third_person_col = request["third_person_col"]
    wrist_row = request["wrist_row"]
    wrist_col = request["wrist_col"]

    wrist = _project_view_to_base(
        output_dir=output_dir,
        meta=meta,
        snapshot=snapshot,
        camera_alias="wrist",
        row=wrist_row,
        col=wrist_col,
        t_camera_target=calibration["wrist"]["matrix"],
        target_frame="tcp",
        t_base_tcp=t_base_tcp,
        depth_cache=depth_cache,
        k_inv_cache=k_inv_cache,
    )
    third = None
    if third_person_row is not None and third_person_col is not None:
        third = _project_view_to_base(
            output_dir=output_dir,
            meta=meta,
            snapshot=snapshot,
            camera_alias="third_person",
            row=third_person_row,
            col=third_person_col,
            t_camera_target=calibration["external"]["matrix"],
            target_frame="base",
            t_base_tcp=t_base_tcp,
            depth_cache=depth_cache,
            k_inv_cache=k_inv_cache,
        )

    if "point_base" not in wrist:
        return {
            "error": wrist.get("error", "wrist back-projection failed"),
            "error_type": wrist.get("error_type", "PerceptionError"),
            "source": "multi_view_rgbd",
            "step": snapshot.get("step_idx"),
            "pixel_correspondence": _pixel_correspondence(
                third_person_row,
                third_person_col,
                wrist_row,
                wrist_col,
            ),
            "warnings": _projection_warnings(third, wrist),
            "diagnostics": {
                "fusion_enabled": False,
                "confidence": _confidence_unavailable("wrist projection failed"),
                "third_person": _compact_projection(third),
                "wrist": _compact_projection(wrist),
            },
        }

    point_base = wrist["point_base"]
    point_base_wrist = wrist["point_base"]
    point_base_third = None
    base_point_delta_m = None
    confidence = _confidence_unavailable("third-person correspondence not provided")
    source = "wrist_only"
    fusion_enabled = False
    if third and "point_base" in third:
        point_base_third = third["point_base"]
        wrist_arr = np.asarray(point_base_wrist, dtype=np.float64)
        third_arr = np.asarray(point_base_third, dtype=np.float64)
        base_point_delta_m = float(np.linalg.norm(wrist_arr - third_arr))
        confidence = _confidence_from_delta(base_point_delta_m)
        if confidence["level"] in {"high", "medium"}:
            point_base = _round_vec((wrist_arr + third_arr) * 0.5)
            source = "multi_view_fused"
            fusion_enabled = True
        else:
            source = "wrist_with_low_confidence_third_person_check"

    diagnostics = {
        "fusion_enabled": fusion_enabled,
        "confidence": confidence,
        "wrist": _compact_projection(wrist),
        "third_person": _compact_projection(third),
    }
    if base_point_delta_m is not None:
        diagnostics["base_point_delta_m"] = round(base_point_delta_m, 5)

    result = {
        "point_base": point_base,
        "source": source,
        "step": snapshot.get("step_idx"),
        "pixel_correspondence": _pixel_correspondence(
            third_person_row,
            third_person_col,
            wrist_row,
            wrist_col,
        ),
        "confidence": confidence,
        "tcp_pose_source": "RLinf raw_base_state.tcp_pose",
        "warnings": _projection_warnings(third, wrist),
    }
    if point_base_wrist is not None:
        result["point_base_wrist"] = point_base_wrist
    if point_base_third is not None:
        result["point_base_third_person"] = point_base_third
    if base_point_delta_m is not None:
        result["base_point_delta_m"] = round(base_point_delta_m, 5)
    if debug:
        result["tcp_pose_xyzw"] = _round_vec(tcp_pose)
        result["transforms"] = {
            "T_base_to_tcp": _round_matrix(t_base_tcp),
            "note": (
                "Wrist points are transformed through the current robot TCP "
                "pose. Third-person points are transformed through the fixed "
                "external camera calibration. point_base is fused only when "
                "the two base-frame estimates agree closely."
            ),
        }
        result["diagnostics"] = diagnostics
    return result


def _pixel_correspondence(
    third_person_row: int | None,
    third_person_col: int | None,
    wrist_row: int,
    wrist_col: int,
) -> dict[str, list[int] | None]:
    return {
        "third_person": (
            [int(third_person_row), int(third_person_col)]
            if third_person_row is not None and third_person_col is not None
            else None
        ),
        "wrist": [int(wrist_row), int(wrist_col)],
    }


def _project_view_to_base(
    *,
    output_dir: Path,
    meta: dict[str, Any],
    snapshot: dict[str, Any],
    camera_alias: str,
    row: int,
    col: int,
    t_camera_target: np.ndarray,
    target_frame: str,
    t_base_tcp: np.ndarray,
    depth_cache: dict[str, tuple[np.ndarray, Path]] | None = None,
    k_inv_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    try:
        camera_key, camera_name = _resolve_camera_alias(meta, camera_alias)
        camera_meta = _camera_meta_for_key(meta, camera_key, camera_name)
        k_matrix = np.asarray(camera_meta.get("intrinsic_K"), dtype=np.float64)
        if k_matrix.shape != (3, 3) or not np.isfinite(k_matrix).all():
            raise PerceptionError(f"intrinsic_K missing for camera {camera_name}")
        if k_inv_cache is not None and camera_key in k_inv_cache:
            k_inv = k_inv_cache[camera_key]
        else:
            k_inv = np.linalg.inv(k_matrix)
            if k_inv_cache is not None:
                k_inv_cache[camera_key] = k_inv
        depth, depth_path = _load_cached_depth(
            output_dir=output_dir,
            snapshot=snapshot,
            camera_key=camera_key,
            depth_cache=depth_cache,
        )
        if depth.ndim != 2:
            raise PerceptionError(f"expected 2D depth for {camera_key}, got {depth.shape}")
        h, w = depth.shape
        if row < 0 or row >= h or col < 0 or col >= w:
            raise PerceptionError(
                f"pixel ({row},{col}) out of bounds for {camera_key} depth {h}x{w}"
            )
        z = float(depth[int(row), int(col)])
        if not np.isfinite(z) or z <= 0.0 or z > 10.0:
            raise PerceptionError(
                f"invalid depth {z:.4f}m at {camera_key} pixel ({row},{col})"
            )
        pixel = np.array([float(col), float(row), 1.0], dtype=np.float64)
        point_camera = k_inv @ pixel * z
        point_target = t_camera_target @ np.array([*point_camera, 1.0])
        if target_frame == "base":
            point_base_h = point_target
        elif target_frame == "tcp":
            point_base_h = t_base_tcp @ point_target
        else:
            raise PerceptionError(f"unknown target frame {target_frame!r}")
        return {
            "camera_alias": camera_alias,
            "camera_key": camera_key,
            "camera_name": camera_name,
            "pixel": [int(row), int(col)],
            "depth_m": round(z, 5),
            "depth_path": str(depth_path),
            "point_camera": _round_vec(point_camera),
            "point_target": _round_vec(point_target[:3]),
            "point_base": _round_vec(point_base_h[:3]),
            "target_frame": target_frame,
        }
    except Exception as exc:
        return {
            "camera_alias": camera_alias,
            "pixel": [int(row), int(col)],
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def pose7_to_matrix(pose: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to a homogeneous transform."""
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (7,):
        raise PerceptionError(f"expected tcp_pose shape (7,), got {arr.shape}")
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_xyzw_to_matrix(arr[3:])
    t[:3, 3] = arr[:3]
    return t


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0:
        raise PerceptionError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(values: dict[str, Any]) -> np.ndarray:
    """Convert YAML qw/qx/qy/qz fields to a 3x3 rotation matrix."""
    return quat_xyzw_to_matrix(
        np.array(
            [
                float(values["qx"]),
                float(values["qy"]),
                float(values["qz"]),
                float(values["qw"]),
            ],
            dtype=np.float64,
        )
    )


def _normalize_pixel_requests(
    *,
    third_person_row: int | None,
    third_person_col: int | None,
    wrist_row: int | None,
    wrist_col: int | None,
    pixels: list[dict[str, Any]] | None,
) -> list[dict[str, int | None]]:
    if pixels is not None:
        if not isinstance(pixels, list) or not pixels:
            raise PerceptionError("pixels must be a non-empty list")
        return [_normalize_pixel_request(item, index=idx) for idx, item in enumerate(pixels)]

    values = {"wrist_row": wrist_row, "wrist_col": wrist_col}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise PerceptionError(
            "single-point call is missing required fields: " + ", ".join(missing)
        )
    if (third_person_row is None) != (third_person_col is None):
        raise PerceptionError(
            "third_person_row and third_person_col must be provided together"
        )
    return [
        {
            "third_person_row": (
                int(third_person_row) if third_person_row is not None else None
            ),
            "third_person_col": (
                int(third_person_col) if third_person_col is not None else None
            ),
            "wrist_row": int(wrist_row),
            "wrist_col": int(wrist_col),
        }
    ]


def _normalize_pixel_request(item: dict[str, Any], *, index: int) -> dict[str, int | None]:
    if not isinstance(item, dict):
        raise PerceptionError(f"pixels[{index}] must be an object")
    third = item.get("third_person")
    wrist = item.get("wrist")
    if third is None and (
        item.get("third_person_row") is not None
        or item.get("third_person_col") is not None
    ):
        third = [item.get("third_person_row"), item.get("third_person_col")]
    if wrist is None:
        wrist = [item.get("wrist_row"), item.get("wrist_col")]
    wrist_row, wrist_col = _coerce_pixel_pair(wrist, name=f"pixels[{index}].wrist")
    if third is None:
        third_row = None
        third_col = None
    else:
        third_row, third_col = _coerce_pixel_pair(
            third,
            name=f"pixels[{index}].third_person",
        )
    return {
        "third_person_row": third_row,
        "third_person_col": third_col,
        "wrist_row": wrist_row,
        "wrist_col": wrist_col,
    }


def _coerce_pixel_pair(value: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise PerceptionError(f"{name} must be [row, col]")
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise PerceptionError(f"{name} must contain integer row/col values") from exc


def _resolve_camera_alias(meta: dict[str, Any], alias: str) -> tuple[str, str | None]:
    observation_map = meta.get("observation_camera_map") or {}
    cameras = meta.get("cameras") or {}
    if alias in {"third_person", "external", "extra_0", "agentview"}:
        camera_key = "extra_0"
        camera_name = observation_map.get(camera_key)
        if camera_name is None:
            extras = [name for name in sorted(cameras) if name != observation_map.get("main")]
            camera_name = extras[0] if extras else None
        return camera_key, camera_name
    if alias in {"wrist", "main"}:
        camera_key = "main"
        camera_name = observation_map.get(camera_key)
        if camera_name is None:
            wrist_names = [name for name in sorted(cameras) if "wrist" in name.lower()]
            camera_name = wrist_names[0] if wrist_names else None
        return camera_key, camera_name
    raise PerceptionError(f"unsupported camera alias {alias!r}")


def _camera_meta_for_key(
    meta: dict[str, Any],
    camera_key: str,
    camera_name: str | None,
) -> dict[str, Any]:
    cameras = meta.get("cameras") or {}
    if camera_name and camera_name in cameras:
        return cameras[camera_name]
    if camera_key in cameras:
        return cameras[camera_key]
    raise PerceptionError(
        f"camera metadata not found for key={camera_key!r}, name={camera_name!r}"
    )


def _depth_artifact_path(
    output_dir: Path,
    snapshot: dict[str, Any],
    camera_key: str,
) -> Path:
    path = (((snapshot.get("artifacts") or {}).get("depths") or {}).get(camera_key))
    if not path and camera_key == "main":
        path = snapshot.get("depth")
    if not path and camera_key == "extra_0":
        path = snapshot.get("third_person_depth")
    if not path:
        step = int(snapshot.get("step_idx", 0))
        name = f"depth_{step:02d}.npy" if camera_key == "main" else f"{camera_key}_depth_{step:02d}.npy"
        path = output_dir / "depths" / name
    depth_path = _resolve_artifact_path(output_dir, path)
    if not depth_path.exists():
        raise PerceptionError(f"depth artifact not found: {depth_path}")
    return depth_path


def _load_cached_depth(
    *,
    output_dir: Path,
    snapshot: dict[str, Any],
    camera_key: str,
    depth_cache: dict[str, tuple[np.ndarray, Path]] | None,
) -> tuple[np.ndarray, Path]:
    if depth_cache is not None and camera_key in depth_cache:
        return depth_cache[camera_key]
    depth_path = _depth_artifact_path(output_dir, snapshot, camera_key)
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    if depth_cache is not None:
        depth_cache[camera_key] = (depth, depth_path)
    return depth, depth_path


def _resolve_artifact_path(output_dir: Path, path: str | Path) -> Path:
    """Resolve artifact paths saved as absolute or episode-relative."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        raw_path,
        output_dir / raw_path,
        output_dir / raw_path.name,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (output_dir / raw_path).resolve()


def _snapshot_tcp_pose(snapshot: dict[str, Any]) -> list[float]:
    state = snapshot.get("state") or {}
    raw = state.get("raw_base_state") or {}
    pose = raw.get("tcp_pose")
    if pose is None:
        raise PerceptionError("snapshot is missing state.raw_base_state.tcp_pose")
    return [float(v) for v in pose]


def _load_calibration_bundle_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PerceptionError(
            f"Franka calibration bundle not found: {path}. "
            "Run scripts/sync_franka_calibration.py to create it."
        )
    try:
        data = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError as exc:
        raise PerceptionError(f"invalid Franka calibration bundle {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PerceptionError(f"Franka calibration bundle must be a JSON object: {path}")
    return data


def _calibration_entry(bundle: dict[str, Any], name: str) -> dict[str, Any]:
    entry = bundle.get(name)
    if not isinstance(entry, dict):
        raise PerceptionError(f"Franka calibration bundle missing {name!r} entry")
    return entry


def _normalize_calibration(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    params = data.get("parameters") or {}
    transform = data.get("transformation") or {}
    required = {"x", "y", "z", "qx", "qy", "qz", "qw"}
    missing = required - set(transform)
    if missing:
        raise PerceptionError(f"{path} missing transform fields: {sorted(missing)}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(transform)
    matrix[:3, 3] = [
        float(transform["x"]),
        float(transform["y"]),
        float(transform["z"]),
    ]
    return {
        "path": str(path),
        "source_name": data.get("source_name"),
        "parameters": params,
        "matrix": matrix,
        "base_frame": (
            params.get("robot_effector_frame")
            if params.get("eye_on_hand")
            else params.get("robot_base_frame")
        ),
        "tracking_base_frame": params.get("tracking_base_frame"),
        "eye_on_hand": bool(params.get("eye_on_hand")),
    }


def _compact_calibration(item: dict[str, Any]) -> dict[str, Any]:
    matrix = item["matrix"]
    return {
        "path": item["path"],
        "eye_on_hand": item["eye_on_hand"],
        "base_frame": item["base_frame"],
        "tracking_base_frame": item["tracking_base_frame"],
        "translation": _round_vec(matrix[:3, 3], ndigits=6),
        "matrix_maps": "camera_frame_point -> base_frame_point",
    }


def _compact_projection(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    keys = [
        "camera_alias",
        "camera_key",
        "camera_name",
        "pixel",
        "depth_m",
        "point_base",
        "target_frame",
        "error",
        "error_type",
    ]
    return {key: result[key] for key in keys if key in result}


def _confidence_from_delta(delta_m: float) -> dict[str, Any]:
    delta = float(delta_m)
    if delta <= 0.015:
        level = "high"
    elif delta <= 0.03:
        level = "medium"
    elif delta <= 0.06:
        level = "low"
    else:
        level = "very_low"
    score = max(0.0, min(1.0, 1.0 - delta / 0.08))
    return {
        "score": round(score, 3),
        "level": level,
        "base_point_delta_m": round(delta, 5),
        "meaning": (
            "Confidence is based on disagreement between independently "
            "back-projected wrist and third-person base-frame points."
        ),
    }


def _confidence_unavailable(reason: str) -> dict[str, Any]:
    return {
        "score": None,
        "level": "unavailable",
        "reason": reason,
    }


def _aggregate_points(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not points:
        return None
    arr = np.asarray([point["point_base"] for point in points], dtype=np.float64)
    confidence_scores = [
        point.get("confidence", {}).get("score")
        for point in points
        if isinstance(point.get("confidence", {}).get("score"), int | float)
    ]
    deltas = [
        point.get("base_point_delta_m")
        for point in points
        if isinstance(point.get("base_point_delta_m"), int | float)
    ]
    return {
        "point_base_mean": _round_vec(arr.mean(axis=0)),
        "point_base_median": _round_vec(np.median(arr, axis=0)),
        "confidence_score_mean": (
            round(float(np.mean(confidence_scores)), 3)
            if confidence_scores
            else None
        ),
        "base_point_delta_mean_m": (
            round(float(np.mean(deltas)), 5) if deltas else None
        ),
    }


def _projection_warnings(
    third: dict[str, Any] | None,
    wrist: dict[str, Any],
) -> list[str]:
    warnings = []
    if third and "error" in third:
        warnings.append(f"third_person back-projection failed: {third['error']}")
    if "error" in wrist:
        warnings.append(f"wrist back-projection failed: {wrist['error']}")
    if third and "point_base" in third and "point_base" in wrist:
        delta = float(
            np.linalg.norm(
                np.asarray(third["point_base"]) - np.asarray(wrist["point_base"])
            )
        )
        if delta > 0.06:
            warnings.append(
                f"third_person/wrist base-frame estimates differ by {delta:.3f}m; "
                "treat this correspondence as low confidence"
            )
    return warnings


def _round_vec(vec: Any, ndigits: int = 5) -> list[float]:
    return [round(float(v), ndigits) for v in np.asarray(vec).reshape(-1).tolist()]


def _round_matrix(matrix: Any, ndigits: int = 6) -> list[list[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    return [
        [round(float(value), ndigits) for value in row]
        for row in arr.tolist()
    ]
